import SwiftUI

protocol KnowledgeGraphReading: AnyObject, Sendable {
    func fetchKGEntities(limit: Int) throws -> [BrainDatabase.KGEntityRow]
    func fetchKGRelations(limit: Int) throws -> [BrainDatabase.KGRelationRow]
}

extension BrainDatabase: KnowledgeGraphReading {}

@MainActor
final class KGViewModel: ObservableObject {
    @Published var nodes: [KGNode] = []
    @Published var edges: [KGEdge] = []
    @Published var isLoading = false
    @Published var selectedNodeId: String?
    @Published var selectedEntity: EntityCard?
    @Published var selectedEntityChunks: [BrainDatabase.KGChunkRow] = []
    @Published var selectedEntityChunkTotal: Int = 0
    @Published var selectedEntityFileTotal: Int = 0
    @Published var selectedEntityFiles: [BrainDatabase.SourceFileRow] = []
    @Published var selectedConversation: BrainDatabase.ExpandedConversation?
    @Published private(set) var isLoadingSelectedEntityChunks = false
    @Published private(set) var isLoadingSelectedEntityFiles = false
    @Published private(set) var selectedEntityCanLoadMoreChunks = false
    @Published private(set) var selectedEntityCanLoadMoreFiles = false
    @Published private(set) var degradationState: DegradationState = .healthy

    /// Set by KGCanvasView via GeometryReader — used for centering force
    var canvasCenter: CGPoint = CGPoint(x: 300, y: 250)

    private let database: BrainDatabase?
    private let graphReader: KnowledgeGraphReading
    private var layoutCanvasSize: CGSize?
    private var selectedEntityChunkCursor: BrainDatabase.ChunkCursor?
    private var selectedEntityFileCursor: BrainDatabase.SourceFileCursor?
    private var selectedEntityLoadTask: Task<Void, Never>?
    private var selectedEntityGeneration = 0
    private let selectedEntityChunkPageSize = 15
    private let selectedEntityFilePageSize = 15

    // Force simulation parameters
    private let repulsionStrength: CGFloat = 5000
    private let attractionStrength: CGFloat = 0.005
    private let centeringStrength: CGFloat = 0.01
    private let damping: CGFloat = 0.85
    private let idealEdgeLength: CGFloat = 150

    init(database: BrainDatabase) {
        self.database = database
        self.graphReader = database
    }

    init(graphReader: KnowledgeGraphReading) {
        self.database = nil
        self.graphReader = graphReader
    }

    // MARK: - Data Loading

    @discardableResult
    func loadGraph(
        retrySleep: (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) async -> Bool {
        guard !isLoading else { return false }
        isLoading = true
        defer { isLoading = false }

        // Retry once on transient ReadOnly/busy/locked failures from the writer
        // pidfile contention (PR #309). The Python enrich-supervisor + drain
        // hold the writer briefly; one retry after a short backoff usually
        // suffices. Persistent failures surface as a degradation badge.
        let attemptLimit = 2
        var lastError: Error?
        for attempt in 1...attemptLimit {
            do {
                let graph = try await Self.fetchGraphRows(reader: graphReader)
                applyGraph(entityRows: graph.entities, relationRows: graph.relations)
                degradationState = .healthy
                return true
            } catch {
                lastError = error
                if attempt < attemptLimit {
                    do {
                        try await retrySleep(.milliseconds(200))
                    } catch {
                        markDegraded(from: lastError)
                        return false
                    }
                }
            }
        }

        markDegraded(from: lastError)
        // Keep previously-loaded nodes/edges so the user sees last-known-good
        // data rather than a blank canvas — degraded ≠ hidden.
        return false
    }

    @discardableResult
    func loadGraphUntilSuccessful(
        retryDelay: Duration = .seconds(5),
        sleep: (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) async -> Bool {
        while !Task.isCancelled {
            if await loadGraph() {
                return true
            }
            do {
                try await sleep(retryDelay)
            } catch {
                return false
            }
        }
        return false
    }

    @discardableResult
    func loadGraphRepeatedly(
        refreshDelay: Duration = .seconds(30),
        retryDelay: Duration = .seconds(5),
        sleep: (Duration) async throws -> Void = { try await Task.sleep(for: $0) },
        onSuccessfulLoad: @MainActor () -> Void = {}
    ) async -> Bool {
        var loadedOnce = false
        while !Task.isCancelled {
            let loaded = await loadGraph()
            if loaded {
                loadedOnce = true
                onSuccessfulLoad()
            }
            do {
                try await sleep(loaded ? refreshDelay : retryDelay)
            } catch {
                return loadedOnce
            }
        }
        return loadedOnce
    }

    private func markDegraded(from error: Error?) {
        let reason = error.map { String(describing: $0) } ?? "unknown"
        NSLog("[BrainBar.KG] loadGraph failed: %@", reason)
        degradationState = .degraded(reason: reason)
    }

    private struct GraphRows: Sendable {
        let entities: [BrainDatabase.KGEntityRow]
        let relations: [BrainDatabase.KGRelationRow]
    }

    private nonisolated static func fetchGraphRows(reader: KnowledgeGraphReading) async throws -> GraphRows {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: GraphRows(
                        entities: try reader.fetchKGEntities(limit: 500),
                        relations: try reader.fetchKGRelations(limit: 5_000)
                    ))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func applyGraph(
        entityRows: [BrainDatabase.KGEntityRow],
        relationRows: [BrainDatabase.KGRelationRow]
    ) {
        let entityIds = Set(entityRows.map(\.id))
        let existingNodes = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })

        let incomingNodes = entityRows.map { row in
            let existingNode = existingNodes[row.id]
            return KGNode(
                id: row.id,
                name: row.name,
                entityType: row.entityType,
                importance: row.importance,
                position: existingNode?.position,
                velocity: existingNode?.velocity ?? .zero
            )
        }
        let seededNodes = KGAtlasLayout.seededNodes(incomingNodes, canvasSize: graphCanvasSize)
        nodes = zip(incomingNodes, seededNodes).map { incomingNode, seededNode in
            existingNodes[incomingNode.id] == nil ? seededNode : incomingNode
        }

        // Only include edges where both endpoints exist
        edges = relationRows.compactMap { row in
            guard entityIds.contains(row.sourceId), entityIds.contains(row.targetId) else {
                return nil
            }
            return KGEdge(
                sourceId: row.sourceId,
                targetId: row.targetId,
                relationType: row.relationType
            )
        }
    }

    private var graphCanvasSize: CGSize {
        layoutCanvasSize ?? CGSize(
            width: max(canvasCenter.x * 2, 640),
            height: max(canvasCenter.y * 2, 480)
        )
    }

    func updateCanvasSize(_ size: CGSize) {
        guard size != .zero else { return }
        guard layoutCanvasSize != size else {
            canvasCenter = CGPoint(x: size.width / 2, y: size.height / 2)
            return
        }
        layoutCanvasSize = size
        canvasCenter = CGPoint(x: size.width / 2, y: size.height / 2)
        guard !nodes.isEmpty else { return }
        nodes = KGAtlasLayout.seededNodes(nodes, canvasSize: size)
    }

    // MARK: - Selection

    func selectNode(id: String?) {
        selectedEntityLoadTask?.cancel()
        selectedEntityGeneration += 1
        let generation = selectedEntityGeneration
        selectedNodeId = id
        if let id {
            guard let database else { return }
            if let lookup = try? database.lookupEntity(query: nodeById(id)?.name ?? "") {
                selectedEntity = EntityCard(lookupPayload: lookup)
            }
            selectedEntityChunks = []
            selectedEntityFiles = []
            selectedEntityChunkTotal = 0
            selectedEntityFileTotal = 0
            selectedEntityChunkCursor = nil
            selectedEntityFileCursor = nil
            selectedEntityCanLoadMoreChunks = false
            selectedEntityCanLoadMoreFiles = false
            isLoadingSelectedEntityChunks = true
            isLoadingSelectedEntityFiles = true
            let chunkPageSize = selectedEntityChunkPageSize
            let filePageSize = selectedEntityFilePageSize
            selectedEntityLoadTask = Task { [weak self, database] in
                let result = await Self.fetchInitialSidebarRows(
                    database: database,
                    entityId: id,
                    chunkLimit: chunkPageSize,
                    fileLimit: filePageSize
                )
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    guard let self, self.selectedNodeId == id, self.selectedEntityGeneration == generation else { return }
                    self.isLoadingSelectedEntityChunks = false
                    self.isLoadingSelectedEntityFiles = false
                    switch result {
                    case .success(let rows):
                        self.selectedEntityChunkTotal = rows.chunkTotal
                        self.selectedEntityFileTotal = rows.fileTotal
                        self.selectedEntityChunks = rows.chunkPage.rows
                        self.selectedEntityFiles = rows.filePage.rows
                        self.selectedEntityChunkCursor = rows.chunkPage.nextCursor
                        self.selectedEntityFileCursor = rows.filePage.nextCursor
                        self.selectedEntityCanLoadMoreChunks = rows.chunkPage.nextCursor != nil
                        self.selectedEntityCanLoadMoreFiles = rows.filePage.nextCursor != nil
                    case .failure:
                        self.selectedEntityChunkTotal = 0
                        self.selectedEntityFileTotal = 0
                        self.selectedEntityChunks = []
                        self.selectedEntityFiles = []
                        self.selectedEntityChunkCursor = nil
                        self.selectedEntityFileCursor = nil
                        self.selectedEntityCanLoadMoreChunks = false
                        self.selectedEntityCanLoadMoreFiles = false
                    }
                }
            }
        } else {
            selectedEntity = nil
            selectedEntityChunks = []
            selectedEntityFiles = []
            selectedEntityChunkTotal = 0
            selectedEntityFileTotal = 0
            selectedEntityChunkCursor = nil
            selectedEntityFileCursor = nil
            selectedEntityCanLoadMoreChunks = false
            selectedEntityCanLoadMoreFiles = false
            isLoadingSelectedEntityChunks = false
            isLoadingSelectedEntityFiles = false
            selectedConversation = nil
        }
    }

    func loadMoreChunks() {
        guard let database, let entityId = selectedNodeId, let cursor = selectedEntityChunkCursor else { return }
        guard !isLoadingSelectedEntityChunks else { return }
        isLoadingSelectedEntityChunks = true
        let pageSize = selectedEntityChunkPageSize
        let generation = selectedEntityGeneration
        Task { [weak self, database] in
            let result = await Self.fetchChunkPage(
                database: database,
                entityId: entityId,
                cursor: cursor,
                limit: pageSize
            )
            await MainActor.run {
                guard let self, self.selectedNodeId == entityId, self.selectedEntityGeneration == generation else { return }
                self.isLoadingSelectedEntityChunks = false
                if case .success(let page) = result {
                    self.selectedEntityChunks.append(contentsOf: page.rows)
                    self.selectedEntityChunkCursor = page.nextCursor
                    self.selectedEntityCanLoadMoreChunks = page.nextCursor != nil
                } else {
                    self.selectedEntityChunkCursor = nil
                    self.selectedEntityCanLoadMoreChunks = false
                }
            }
        }
    }

    func loadMoreFiles() {
        guard let database, let entityId = selectedNodeId, let cursor = selectedEntityFileCursor else { return }
        guard !isLoadingSelectedEntityFiles else { return }
        isLoadingSelectedEntityFiles = true
        let pageSize = selectedEntityFilePageSize
        let generation = selectedEntityGeneration
        Task { [weak self, database] in
            let result = await Self.fetchFilePage(
                database: database,
                entityId: entityId,
                cursor: cursor,
                limit: pageSize
            )
            await MainActor.run {
                guard let self, self.selectedNodeId == entityId, self.selectedEntityGeneration == generation else { return }
                self.isLoadingSelectedEntityFiles = false
                if case .success(let page) = result {
                    self.selectedEntityFiles.append(contentsOf: page.rows)
                    self.selectedEntityFileCursor = page.nextCursor
                    self.selectedEntityCanLoadMoreFiles = page.nextCursor != nil
                } else {
                    self.selectedEntityFileCursor = nil
                    self.selectedEntityCanLoadMoreFiles = false
                }
            }
        }
    }

    func openConversation(chunkID: String) {
        guard let database else { return }
        selectedConversation = try? database.expandedConversation(id: chunkID)
    }

    // MARK: - Hit Testing

    func nodeAt(point: CGPoint) -> KGNode? {
        for node in nodes {
            let dx = point.x - node.position.x
            let dy = point.y - node.position.y
            let dist = sqrt(dx * dx + dy * dy)
            if dist <= node.radius + 4 { // 4pt tolerance
                return node
            }
        }
        return nil
    }

    // MARK: - Force-Directed Layout

    func tick() -> CGFloat {
        guard nodes.count > 1 else { return 0 }

        var forces = Array(repeating: CGVector.zero, count: nodes.count)
        let center = canvasCenter

        // Repulsion: all pairs (Coulomb's law)
        for i in 0..<nodes.count {
            for j in (i + 1)..<nodes.count {
                let dx = nodes[i].position.x - nodes[j].position.x
                let dy = nodes[i].position.y - nodes[j].position.y
                let distSq = max(dx * dx + dy * dy, 1)
                let force = repulsionStrength / distSq
                let dist = sqrt(distSq)
                let fx = (dx / dist) * force
                let fy = (dy / dist) * force
                forces[i].dx += fx
                forces[i].dy += fy
                forces[j].dx -= fx
                forces[j].dy -= fy
            }
        }

        // Attraction: edges (spring force)
        let nodeIndex = Dictionary(uniqueKeysWithValues: nodes.enumerated().map { ($1.id, $0) })
        for edge in edges {
            guard let si = nodeIndex[edge.sourceId], let ti = nodeIndex[edge.targetId] else { continue }
            let dx = nodes[ti].position.x - nodes[si].position.x
            let dy = nodes[ti].position.y - nodes[si].position.y
            let dist = max(sqrt(dx * dx + dy * dy), 1)
            let displacement = dist - idealEdgeLength
            let fx = (dx / dist) * displacement * attractionStrength
            let fy = (dy / dist) * displacement * attractionStrength
            forces[si].dx += fx
            forces[si].dy += fy
            forces[ti].dx -= fx
            forces[ti].dy -= fy
        }

        // Centering force
        for i in 0..<nodes.count {
            forces[i].dx += (center.x - nodes[i].position.x) * centeringStrength
            forces[i].dy += (center.y - nodes[i].position.y) * centeringStrength
        }

        // Apply forces with damping
        var totalKineticEnergy: CGFloat = 0
        for i in 0..<nodes.count {
            nodes[i].velocity.dx = (nodes[i].velocity.dx + forces[i].dx) * damping
            nodes[i].velocity.dy = (nodes[i].velocity.dy + forces[i].dy) * damping
            nodes[i].position.x += nodes[i].velocity.dx
            nodes[i].position.y += nodes[i].velocity.dy

            let speedSq = (nodes[i].velocity.dx * nodes[i].velocity.dx) + (nodes[i].velocity.dy * nodes[i].velocity.dy)
            totalKineticEnergy += 0.5 * speedSq
        }

        return totalKineticEnergy
    }

    // MARK: - Helpers

    private func nodeById(_ id: String) -> KGNode? {
        nodes.first { $0.id == id }
    }

    private struct SidebarRows: Sendable {
        let chunkTotal: Int
        let fileTotal: Int
        let chunkPage: BrainDatabase.ChunkPage
        let filePage: BrainDatabase.SourceFilePage
    }

    private nonisolated static func fetchInitialSidebarRows(
        database: BrainDatabase,
        entityId: String,
        chunkLimit: Int,
        fileLimit: Int
    ) async -> Result<SidebarRows, Error> {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let rows = try SidebarRows(
                        chunkTotal: database.fetchEntityChunkCount(entityId: entityId),
                        fileTotal: database.fetchEntitySourceFileCount(entityId: entityId),
                        chunkPage: database.fetchEntityChunksPage(entityId: entityId, after: nil, limit: chunkLimit),
                        filePage: database.fetchEntitySourceFiles(entityId: entityId, limit: fileLimit, after: nil)
                    )
                    continuation.resume(returning: .success(rows))
                } catch {
                    continuation.resume(returning: .failure(error))
                }
            }
        }
    }

    private nonisolated static func fetchChunkPage(
        database: BrainDatabase,
        entityId: String,
        cursor: BrainDatabase.ChunkCursor,
        limit: Int
    ) async -> Result<BrainDatabase.ChunkPage, Error> {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: .success(
                        try database.fetchEntityChunksPage(entityId: entityId, after: cursor, limit: limit)
                    ))
                } catch {
                    continuation.resume(returning: .failure(error))
                }
            }
        }
    }

    private nonisolated static func fetchFilePage(
        database: BrainDatabase,
        entityId: String,
        cursor: BrainDatabase.SourceFileCursor,
        limit: Int
    ) async -> Result<BrainDatabase.SourceFilePage, Error> {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: .success(
                        try database.fetchEntitySourceFiles(entityId: entityId, limit: limit, after: cursor)
                    ))
                } catch {
                    continuation.resume(returning: .failure(error))
                }
            }
        }
    }
}
