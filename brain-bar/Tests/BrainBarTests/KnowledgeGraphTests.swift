// KnowledgeGraphTests.swift — RED tests for Knowledge Graph viewer.
//
// TDD: These tests are written FIRST, before implementation.
// Covers: kg_entity_chunks table, fetchKGEntities, fetchKGRelations,
//         fetchEntityChunks, linkEntityChunk, KGNode, KGEdge, KGViewModel.

import XCTest
import SQLite3
@testable import BrainBar

// MARK: - Database KG Query Tests

final class KGDatabaseTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-kg-test-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    // MARK: - Schema

    func testKGEntityChunksTableExists() throws {
        let exists = try db.tableExists("kg_entity_chunks")
        XCTAssertTrue(exists, "kg_entity_chunks table must exist")
    }

    // MARK: - fetchKGEntities

    func testFetchKGEntitiesReturnsInsertedEntities() throws {
        try db.insertEntity(id: "person-abc", type: "person", name: "Alice")
        try db.insertEntity(id: "project-xyz", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "person-abc", targetId: "project-xyz", relationType: "builds")

        let entities = try db.fetchKGEntities()
        XCTAssertEqual(entities.count, 2)

        let names = Set(entities.map(\.name))
        XCTAssertTrue(names.contains("Alice"))
        XCTAssertTrue(names.contains("BrainLayer"))
    }

    func testFetchKGEntitiesRespectsLimit() throws {
        try db.insertEntity(id: "project-root", type: "project", name: "Root")
        for i in 0..<10 {
            try db.insertEntity(id: "e-\(i)", type: "person", name: "Entity \(i)")
            try db.insertRelation(sourceId: "e-\(i)", targetId: "project-root", relationType: "builds")
        }
        let entities = try db.fetchKGEntities(limit: 5)
        XCTAssertEqual(entities.count, 5)
    }

    func testFetchKGEntitiesReturnsTypeAndDescription() throws {
        try db.insertEntity(id: "tool-vim", type: "tool", name: "Vim", metadata: "{\"desc\":\"editor\"}")
        try db.insertEntity(id: "project-editor", type: "project", name: "Editor")
        try db.insertRelation(sourceId: "tool-vim", targetId: "project-editor", relationType: "supports")
        let entities = try db.fetchKGEntities()
        XCTAssertEqual(entities.first?.entityType, "tool")
        XCTAssertEqual(entities.first?.name, "Vim")
    }

    func testFetchKGEntitiesUsesImportanceColumnWhenMetadataImportanceMissing() throws {
        try db.insertEntity(id: "agent-a", type: "agent", name: "Agent A")
        try db.insertEntity(id: "tool-b", type: "tool", name: "Tool B")
        try db.insertRelation(sourceId: "agent-a", targetId: "tool-b", relationType: "uses")

        guard let handle = db.dbHandle else {
            XCTFail("Expected database handle")
            return
        }

        XCTAssertEqual(sqlite3_exec(handle, "UPDATE kg_entities SET importance = 8.0 WHERE id = 'agent-a'", nil, nil, nil), SQLITE_OK)
        XCTAssertEqual(sqlite3_exec(handle, "UPDATE kg_entities SET importance = 2.0 WHERE id = 'tool-b'", nil, nil, nil), SQLITE_OK)

        let entities = try db.fetchKGEntities()
        XCTAssertEqual(entities.map(\.name), ["Agent A", "Tool B"])
        XCTAssertEqual(entities.first?.importance, 8.0)
        XCTAssertEqual(entities.last?.importance, 2.0)
    }

    func testFetchKGEntitiesEmptyDB() throws {
        let entities = try db.fetchKGEntities()
        XCTAssertTrue(entities.isEmpty)
    }

    // MARK: - fetchKGRelations

    func testFetchKGRelationsReturnsInsertedRelations() throws {
        try db.insertEntity(id: "p1", type: "person", name: "Alice")
        try db.insertEntity(id: "p2", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "p1", targetId: "p2", relationType: "builds")

        let relations = try db.fetchKGRelations()
        XCTAssertEqual(relations.count, 1)
        XCTAssertEqual(relations.first?.sourceId, "p1")
        XCTAssertEqual(relations.first?.targetId, "p2")
        XCTAssertEqual(relations.first?.relationType, "builds")
    }

    func testFetchKGRelationsSurfacesExpirationMetadataWithoutFilteringExpiredRelations() throws {
        try db.insertEntity(id: "person-etan", type: "person", name: "Etan")
        try db.insertEntity(id: "company-domica", type: "company", name: "Domica")
        try db.insertEntity(id: "project-brainlayer", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "person-etan", targetId: "company-domica", relationType: "cto_of")
        try db.insertRelation(sourceId: "person-etan", targetId: "project-brainlayer", relationType: "builds")

        guard let handle = db.dbHandle else {
            XCTFail("Expected database handle")
            return
        }

        XCTAssertEqual(
            sqlite3_exec(
                handle,
                """
                UPDATE kg_relations
                SET expired_at = '2026-05-24T00:00:00Z',
                    valid_until = '2026-05-24T00:00:00Z'
                WHERE source_id = 'person-etan' AND target_id = 'company-domica'
                """,
                nil,
                nil,
                nil
            ),
            SQLITE_OK
        )

        let relations = try db.fetchKGRelations()

        XCTAssertEqual(relations.count, 2, "Expired relations remain first-class graph context")
        let expired = try XCTUnwrap(relations.first { $0.targetId == "company-domica" })
        XCTAssertNotNil(expired.expiredAt)
        XCTAssertNotNil(expired.validUntil)
    }

    func testFetchKGRelationsMultiple() throws {
        try db.insertEntity(id: "a", type: "person", name: "A")
        try db.insertEntity(id: "b", type: "project", name: "B")
        try db.insertEntity(id: "c", type: "tool", name: "C")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")
        try db.insertRelation(sourceId: "a", targetId: "c", relationType: "uses")

        let relations = try db.fetchKGRelations()
        XCTAssertEqual(relations.count, 2)
    }

    func testFetchKGRelationsRespectsLimit() throws {
        try db.insertEntity(id: "root", type: "project", name: "Root")
        for index in 0..<5 {
            try db.insertEntity(id: "e-\(index)", type: "person", name: "Entity \(index)")
            try db.insertRelation(sourceId: "e-\(index)", targetId: "root", relationType: "builds")
        }

        let relations = try db.fetchKGRelations(limit: 2)

        XCTAssertEqual(relations.count, 2)
    }

    func testFetchKGRelationsHandlesOversizedLimit() throws {
        try db.insertEntity(id: "root", type: "project", name: "Root")
        for index in 0..<5 {
            try db.insertEntity(id: "e-\(index)", type: "person", name: "Entity \(index)")
            try db.insertRelation(sourceId: "e-\(index)", targetId: "root", relationType: "builds")
        }

        let relations = try db.fetchKGRelations(limit: Int.max)

        XCTAssertEqual(relations.count, 5)
    }

    func testFetchKGRelationsEmptyDB() throws {
        let relations = try db.fetchKGRelations()
        XCTAssertTrue(relations.isEmpty)
    }

    func testLookupEntityPayloadIncludesRelationExpirationMetadata() throws {
        try db.insertEntity(id: "person-etan", type: "person", name: "Etan")
        try db.insertEntity(id: "company-domica", type: "company", name: "Domica")
        try db.insertRelation(sourceId: "person-etan", targetId: "company-domica", relationType: "cto_of")

        guard let handle = db.dbHandle else {
            XCTFail("Expected database handle")
            return
        }

        XCTAssertEqual(
            sqlite3_exec(
                handle,
                """
                UPDATE kg_relations
                SET expired_at = '2026-05-24T00:00:00Z',
                    valid_until = '2026-05-24T00:00:00Z'
                WHERE source_id = 'person-etan' AND target_id = 'company-domica'
                """,
                nil,
                nil,
                nil
            ),
            SQLITE_OK
        )

        let payload = try XCTUnwrap(db.lookupEntity(query: "Etan"))
        let card = EntityCard(lookupPayload: payload)
        let relation = try XCTUnwrap(card.relations.first)

        XCTAssertEqual(relation.targetName, "Domica")
        XCTAssertNotNil(relation.expiredAt)
        XCTAssertNotNil(relation.validUntil)
    }

    func testReadOnlyLegacyKGRelationSchemaDoesNotRequireExpirationColumns() throws {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try createLegacyKGDatabaseWithoutRelationExpirationColumns(path: tempDBPath)

        let reader = BrainDatabase(path: tempDBPath, openConfiguration: .init(readOnly: true))
        defer { reader.close() }

        let relations = try reader.fetchKGRelations()
        let payload = try XCTUnwrap(reader.lookupEntity(query: "Etan"))
        let card = EntityCard(lookupPayload: payload)

        XCTAssertEqual(relations.count, 1)
        XCTAssertNil(relations.first?.validUntil)
        XCTAssertNil(relations.first?.expiredAt)
        XCTAssertEqual(card.relations.first?.targetName, "Domica")
        XCTAssertNil(card.relations.first?.validUntil)
        XCTAssertNil(card.relations.first?.expiredAt)
    }

    // MARK: - linkEntityChunk + fetchEntityChunks

    func testLinkEntityChunkAndFetch() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try db.insertChunk(
            id: "chunk-1", content: "Alice wrote code",
            sessionId: "s1", project: "test", contentType: "ai_code", importance: 7
        )
        try db.linkEntityChunk(entityId: "e1", chunkId: "chunk-1", relevance: 0.9)

        let chunks = try db.fetchEntityChunks(entityId: "e1")
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.chunkID, "chunk-1")
        XCTAssertTrue((chunks.first?.snippet ?? "").contains("Alice"))
    }

    func testFetchEntityChunksRespectsLimit() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        for i in 0..<10 {
            try db.insertChunk(
                id: "c-\(i)", content: "Chunk \(i) content",
                sessionId: "s1", project: "test", contentType: "ai_code", importance: 5
            )
            try db.linkEntityChunk(entityId: "e1", chunkId: "c-\(i)", relevance: Double(10 - i) / 10.0)
        }
        let chunks = try db.fetchEntityChunks(entityId: "e1", limit: 3)
        XCTAssertEqual(chunks.count, 3)
    }

    func testFetchEntityChunksOrderedByRelevance() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try db.insertChunk(id: "low", content: "Low relevance", sessionId: "s1", project: "t", contentType: "ai_code", importance: 3)
        try db.insertChunk(id: "high", content: "High relevance", sessionId: "s1", project: "t", contentType: "ai_code", importance: 8)
        try db.linkEntityChunk(entityId: "e1", chunkId: "low", relevance: 0.2)
        try db.linkEntityChunk(entityId: "e1", chunkId: "high", relevance: 0.95)

        let chunks = try db.fetchEntityChunks(entityId: "e1")
        XCTAssertEqual(chunks.first?.chunkID, "high")
    }

    func testFetchEntityChunksNoLinks() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Lonely")
        let chunks = try db.fetchEntityChunks(entityId: "e1")
        XCTAssertTrue(chunks.isEmpty)
    }

    func testFetchEntityChunkCountAndCursorPagesTwentyFiveRows() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        for i in 0..<25 {
            try insertLinkedChunk(
                id: "c-\(i)",
                entityId: "e1",
                content: "Chunk \(i) content",
                sourceFile: "/tmp/source-\(i % 3).md",
                createdAt: String(format: "2026-05-24T12:%02d:00Z", i),
                relevance: Double(i) / 100.0
            )
        }

        let firstPage = try db.fetchEntityChunksPage(entityId: "e1", after: nil, limit: 10)
        let secondPage = try db.fetchEntityChunksPage(entityId: "e1", after: firstPage.nextCursor, limit: 10)

        XCTAssertEqual(try db.fetchEntityChunkCount(entityId: "e1"), 25)
        XCTAssertEqual(firstPage.rows.count, 10)
        XCTAssertEqual(secondPage.rows.count, 10)
        XCTAssertEqual(firstPage.rows.first?.chunkID, "c-24")
        XCTAssertEqual(secondPage.rows.first?.chunkID, "c-14")
        XCTAssertTrue(Set(firstPage.rows.map(\.chunkID)).isDisjoint(with: Set(secondPage.rows.map(\.chunkID))))
        XCTAssertNotNil(secondPage.nextCursor)
    }

    func testFetchEntityChunkCountIgnoresOrphanedChunkLinks() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try insertLinkedChunk(
            id: "real",
            entityId: "e1",
            content: "Resolvable chunk",
            sourceFile: "/tmp/source.md",
            createdAt: "2026-05-24T12:00:00Z",
            relevance: 0.9
        )
        try db.linkEntityChunk(entityId: "e1", chunkId: "missing", relevance: 0.8)

        let page = try db.fetchEntityChunksPage(entityId: "e1", after: nil, limit: 15)

        XCTAssertEqual(try db.fetchEntityChunkCount(entityId: "e1"), 1)
        XCTAssertEqual(page.rows.map(\.chunkID), ["real"])
        XCTAssertNil(page.nextCursor)
    }

    func testFetchEntitySourceFileCountReturnsDistinctFiles() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        for i in 0..<5 {
            try insertLinkedChunk(
                id: "shared-\(i)",
                entityId: "e1",
                content: "Shared file chunk \(i)",
                sourceFile: "/tmp/shared.md",
                createdAt: "2026-05-24T12:00:0\(i)Z",
                relevance: 0.8
            )
        }
        try insertLinkedChunk(
            id: "other-1",
            entityId: "e1",
            content: "Other file chunk",
            sourceFile: "/tmp/other.md",
            createdAt: "2026-05-24T12:01:00Z",
            relevance: 0.7
        )

        XCTAssertEqual(try db.fetchEntitySourceFileCount(entityId: "e1"), 2)
    }

    func testFetchEntitySourceFilesPagesAggregatedFiles() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        for i in 0..<3 {
            try insertLinkedChunk(
                id: "alpha-\(i)",
                entityId: "e1",
                content: "Alpha file chunk \(i)",
                sourceFile: "/tmp/alpha.md",
                createdAt: "2026-05-24T12:00:0\(i)Z",
                relevance: 0.6
            )
        }
        try insertLinkedChunk(
            id: "beta-1",
            entityId: "e1",
            content: "Beta file chunk",
            sourceFile: "/tmp/beta.md",
            createdAt: "2026-05-24T12:01:00Z",
            relevance: 0.9
        )
        try insertLinkedChunk(
            id: "gamma-1",
            entityId: "e1",
            content: "Gamma file chunk",
            sourceFile: "/tmp/gamma.md",
            createdAt: "2026-05-24T12:02:00Z",
            relevance: 0.4
        )

        let firstPage = try db.fetchEntitySourceFiles(entityId: "e1", limit: 2, after: nil)
        let secondPage = try db.fetchEntitySourceFiles(entityId: "e1", limit: 2, after: firstPage.nextCursor)

        XCTAssertEqual(firstPage.rows.map(\.sourceFile), ["/tmp/beta.md", "/tmp/alpha.md"])
        XCTAssertEqual(firstPage.rows.map(\.chunkCount), [1, 3])
        XCTAssertEqual(secondPage.rows.map(\.sourceFile), ["/tmp/gamma.md"])
        XCTAssertNil(secondPage.nextCursor)
    }

    private func insertLinkedChunk(
        id: String,
        entityId: String,
        content: String,
        sourceFile: String,
        createdAt: String,
        relevance: Double
    ) throws {
        try db.insertChunk(
            id: id,
            content: content,
            sessionId: "s-\(id)",
            project: "test",
            contentType: "ai_code",
            importance: 5
        )
        try updateChunk(id: id, sourceFile: sourceFile, createdAt: createdAt)
        try db.linkEntityChunk(entityId: entityId, chunkId: id, relevance: relevance)
    }

    private func updateChunk(id: String, sourceFile: String, createdAt: String) throws {
        guard let handle = db.dbHandle else {
            XCTFail("Expected database handle")
            return
        }
        let sql = "UPDATE chunks SET source_file = ?, created_at = ? WHERE id = ?"
        var stmt: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(handle, sql, -1, &stmt, nil), SQLITE_OK)
        defer { sqlite3_finalize(stmt) }
        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, sourceFile, -1, transient)
        sqlite3_bind_text(stmt, 2, createdAt, -1, transient)
        sqlite3_bind_text(stmt, 3, id, -1, transient)
        XCTAssertEqual(sqlite3_step(stmt), SQLITE_DONE)
    }
}

private func createLegacyKGDatabaseWithoutRelationExpirationColumns(path: String) throws {
    var handle: OpaquePointer?
    let openRC = sqlite3_open_v2(
        path,
        &handle,
        SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX,
        nil
    )
    guard openRC == SQLITE_OK, let handle else {
        throw NSError(domain: "KnowledgeGraphTests", code: Int(openRC))
    }
    defer { sqlite3_close(handle) }

    let sql = """
        CREATE TABLE kg_entities (
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            description TEXT,
            importance REAL DEFAULT 0.5
        );
        CREATE TABLE kg_relations (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            properties TEXT DEFAULT '{}',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_id, target_id, relation_type)
        );
        INSERT INTO kg_entities (id, entity_type, name, metadata, description)
        VALUES
            ('person-etan', 'person', 'Etan', '{}', NULL),
            ('company-domica', 'company', 'Domica', '{}', NULL);
        INSERT INTO kg_relations (id, source_id, target_id, relation_type)
        VALUES ('person-etan-cto_of-company-domica', 'person-etan', 'company-domica', 'cto_of');
    """
    let execRC = sqlite3_exec(handle, sql, nil, nil, nil)
    guard execRC == SQLITE_OK else {
        throw NSError(domain: "KnowledgeGraphTests", code: Int(execRC))
    }
}

// MARK: - KGNode / KGEdge Model Tests

final class KGModelTests: XCTestCase {

    func testKGNodeIdentifiable() {
        let node = KGNode(id: "n1", name: "Alice", entityType: "person", importance: 7.0)
        XCTAssertEqual(node.id, "n1")
        XCTAssertEqual(node.name, "Alice")
        XCTAssertEqual(node.entityType, "person")
    }

    func testKGNodeDefaultPosition() {
        let node = KGNode(id: "n1", name: "A", entityType: "person", importance: 5.0)
        // Position should be initialized (not zero — randomized)
        // We just verify the struct is constructable with defaults
        XCTAssertNotNil(node.position)
    }

    func testKGNodeRadiusScalesWithImportance() {
        let low = KGNode(id: "lo", name: "Lo", entityType: "person", importance: 1.0)
        let high = KGNode(id: "hi", name: "Hi", entityType: "person", importance: 9.0)
        XCTAssertGreaterThan(high.radius, low.radius)
    }

    func testKGEdgeProperties() {
        let edge = KGEdge(sourceId: "a", targetId: "b", relationType: "builds")
        XCTAssertEqual(edge.sourceId, "a")
        XCTAssertEqual(edge.targetId, "b")
        XCTAssertEqual(edge.relationType, "builds")
    }

    func testKGEdgeIdentifiable() {
        let edge = KGEdge(sourceId: "a", targetId: "b", relationType: "builds")
        // id should be deterministic from source+target+type
        let edge2 = KGEdge(sourceId: "a", targetId: "b", relationType: "builds")
        XCTAssertEqual(edge.id, edge2.id)
    }

    func testKGNodeColorForType() {
        let person = KGNode(id: "1", name: "P", entityType: "person", importance: 5)
        let project = KGNode(id: "2", name: "Q", entityType: "project", importance: 5)
        // Different types should produce different colors
        XCTAssertNotEqual(person.color, project.color)
    }
}

// MARK: - KGViewModel Tests

@MainActor
final class KGViewModelTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-kgvm-test-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    func testLoadGraphPopulatesNodesAndEdges() async throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        await vm.loadGraph()

        XCTAssertEqual(vm.nodes.count, 2)
        XCTAssertEqual(vm.edges.count, 1)
    }

    func testLoadGraphRefreshPreservesExistingLayoutForStableNodes() async throws {
        // Regression guard: Cursor Bugbot PR #315 flagged repeated graph refresh
        // resetting node positions and discarding in-progress force layout state.
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        vm.nodes[0].position = CGPoint(x: 123, y: 234)
        vm.nodes[0].velocity = CGVector(dx: 5, dy: 7)
        vm.nodes[1].position = CGPoint(x: 456, y: 345)
        vm.nodes[1].velocity = CGVector(dx: -3, dy: 2)

        await vm.loadGraph()

        let nodeA = try XCTUnwrap(vm.nodes.first { $0.id == "a" })
        let nodeB = try XCTUnwrap(vm.nodes.first { $0.id == "b" })
        XCTAssertEqual(nodeA.position, CGPoint(x: 123, y: 234))
        XCTAssertEqual(nodeA.velocity, CGVector(dx: 5, dy: 7))
        XCTAssertEqual(nodeB.position, CGPoint(x: 456, y: 345))
        XCTAssertEqual(nodeB.velocity, CGVector(dx: -3, dy: 2))
    }

    func testUpdateCanvasSizeReseedsLoadedGraphFromActualDrawableSize() async throws {
        // Regression guard: Cursor Bugbot PR #315 flagged a race where data can
        // load before the inner Canvas reports its real size, leaving nodes
        // seeded against the default center until a later size change.
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        let defaultPositions = Dictionary(uniqueKeysWithValues: vm.nodes.map { ($0.id, $0.position) })

        vm.updateCanvasSize(CGSize(width: 1_200, height: 800))

        XCTAssertEqual(vm.canvasCenter, CGPoint(x: 600, y: 400))
        let nodeA = try XCTUnwrap(vm.nodes.first { $0.id == "a" })
        let nodeB = try XCTUnwrap(vm.nodes.first { $0.id == "b" })
        XCTAssertNotEqual(nodeA.position, defaultPositions["a"])
        XCTAssertNotEqual(nodeB.position, defaultPositions["b"])
        XCTAssertEqual(nodeA.position, CGPoint(x: 264, y: 192))
        XCTAssertEqual(nodeB.position, CGPoint(x: 600, y: 192))
    }

    func testLoadGraphKeepsMainActorAvailableWhileLoading() async throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        var mainActorRanDuringLoad = false
        let marker = Task { @MainActor in
            mainActorRanDuringLoad = true
        }

        await vm.loadGraph()

        await marker.value
        XCTAssertTrue(mainActorRanDuringLoad, "loadGraph should suspend off the MainActor while database work runs")
    }

    func testLoadGraphEmptyDB() async throws {
        let vm = KGViewModel(database: db)
        await vm.loadGraph()

        XCTAssertTrue(vm.nodes.isEmpty)
        XCTAssertTrue(vm.edges.isEmpty)
    }

    func testSelectNodeUpdatesSelectedEntity() async throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")
        let vm = KGViewModel(database: db)
        await vm.loadGraph()

        vm.selectNode(id: "a")
        XCTAssertEqual(vm.selectedNodeId, "a")
        XCTAssertNotNil(vm.selectedEntity)
        XCTAssertEqual(vm.selectedEntity?.name, "Alice")
    }

    func testSelectNodeNilDeselects() async throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")
        let vm = KGViewModel(database: db)
        await vm.loadGraph()

        vm.selectNode(id: "a")
        vm.selectNode(id: nil)
        XCTAssertNil(vm.selectedNodeId)
        XCTAssertNil(vm.selectedEntity)
    }

    func testForceTickMovesNodes() async throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        await vm.loadGraph()

        let positionsBefore = vm.nodes.map(\.position)
        vm.tick() // one simulation step
        let positionsAfter = vm.nodes.map(\.position)

        // At least one node should have moved
        let moved = zip(positionsBefore, positionsAfter).contains { $0 != $1 }
        XCTAssertTrue(moved, "Force tick should move at least one node")
    }

    func testNodeHitTestFindsCorrectNode() async throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")
        let vm = KGViewModel(database: db)
        await vm.loadGraph()

        // Place node at known position
        vm.nodes[0].position = CGPoint(x: 100, y: 100)

        let hit = vm.nodeAt(point: CGPoint(x: 102, y: 98)) // within radius
        XCTAssertEqual(hit?.id, "a")

        let miss = vm.nodeAt(point: CGPoint(x: 500, y: 500)) // far away
        XCTAssertNil(miss)
    }

    func testSelectedEntityChunksPopulated() async throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try db.insertEntity(id: "e2", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "e1", targetId: "e2", relationType: "builds")
        try db.insertChunk(
            id: "c1", content: "Alice built BrainLayer",
            sessionId: "s1", project: "test", contentType: "ai_code", importance: 7
        )
        try db.linkEntityChunk(entityId: "e1", chunkId: "c1", relevance: 0.9)

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        vm.selectNode(id: "e1")
        await waitForSelectedEntityChunks(vm)

        XCTAssertFalse(vm.selectedEntityChunks.isEmpty)
        XCTAssertTrue(vm.selectedEntityChunks.first?.snippet.contains("Alice") ?? false)
    }

    func testSelectNodePopulatesTotalsFirstChunkPageAndFilesWithoutBlockingMainActor() async throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try db.insertEntity(id: "e2", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "e1", targetId: "e2", relationType: "builds")
        for i in 0..<25 {
            try db.insertChunk(
                id: "c-\(i)",
                content: "Alice memory \(i)",
                sessionId: "s-\(i)",
                project: "test",
                contentType: "ai_code",
                importance: 5
            )
            try updateChunk(id: "c-\(i)", sourceFile: "/tmp/file-\(i % 4).md", createdAt: String(format: "2026-05-24T12:%02d:00Z", i))
            try db.linkEntityChunk(entityId: "e1", chunkId: "c-\(i)", relevance: Double(i) / 100.0)
        }

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        var mainActorRanAfterSelection = false
        let marker = Task { @MainActor in
            mainActorRanAfterSelection = true
        }

        vm.selectNode(id: "e1")
        XCTAssertTrue(vm.isLoadingSelectedEntityFiles)
        await marker.value
        await waitForSelectedEntityLoad(vm)

        XCTAssertTrue(mainActorRanAfterSelection, "selectNode should not perform sidebar database reads synchronously on the MainActor")
        XCTAssertEqual(vm.selectedEntityChunkTotal, 25)
        XCTAssertEqual(vm.selectedEntityFileTotal, 4)
        XCTAssertEqual(vm.selectedEntityChunks.count, 15)
        XCTAssertEqual(vm.selectedEntityFiles.count, 4)
        XCTAssertFalse(vm.isLoadingSelectedEntityFiles)
    }

    func testLoadMoreChunksAppendsNextPageWithoutDuplicates() async throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try db.insertEntity(id: "e2", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "e1", targetId: "e2", relationType: "builds")
        for i in 0..<25 {
            try db.insertChunk(
                id: "more-\(i)",
                content: "Alice paged memory \(i)",
                sessionId: "more-\(i)",
                project: "test",
                contentType: "ai_code",
                importance: 5
            )
            try updateChunk(id: "more-\(i)", sourceFile: "/tmp/file-\(i % 2).md", createdAt: String(format: "2026-05-24T13:%02d:00Z", i))
            try db.linkEntityChunk(entityId: "e1", chunkId: "more-\(i)", relevance: Double(i) / 100.0)
        }

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        vm.selectNode(id: "e1")
        await waitForSelectedEntityLoad(vm)

        vm.loadMoreChunks()
        await waitForSelectedEntityChunkCount(vm, count: 25)

        XCTAssertEqual(vm.selectedEntityChunks.count, 25)
        XCTAssertEqual(Set(vm.selectedEntityChunks.map(\.chunkID)).count, 25)
        XCTAssertEqual(vm.selectedEntityChunks.first?.chunkID, "more-24")
        XCTAssertEqual(vm.selectedEntityChunks.last?.chunkID, "more-0")
    }

    func testLoadMoreChunkFailureDisablesAutomaticFooterRetry() async throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        for i in 0..<25 {
            try db.insertChunk(
                id: "fail-\(i)",
                content: "Alice failure memory \(i)",
                sessionId: "fail-\(i)",
                project: "test",
                contentType: "ai_code",
                importance: 5
            )
            try updateChunk(id: "fail-\(i)", sourceFile: "/tmp/file-\(i % 2).md", createdAt: String(format: "2026-05-24T14:%02d:00Z", i))
            try db.linkEntityChunk(entityId: "e1", chunkId: "fail-\(i)", relevance: Double(i) / 100.0)
        }

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        vm.selectNode(id: "e1")
        await waitForSelectedEntityLoad(vm)
        XCTAssertTrue(vm.selectedEntityCanLoadMoreChunks)

        db.close()
        vm.loadMoreChunks()
        await waitForSelectedEntityChunkLoadingToFinish(vm)

        XCTAssertEqual(vm.selectedEntityChunks.count, 15)
        XCTAssertFalse(vm.selectedEntityCanLoadMoreChunks)
        XCTAssertFalse(vm.isLoadingSelectedEntityChunks)
    }

    private func updateChunk(id: String, sourceFile: String, createdAt: String) throws {
        guard let handle = db.dbHandle else {
            XCTFail("Expected database handle")
            return
        }
        let sql = "UPDATE chunks SET source_file = ?, created_at = ? WHERE id = ?"
        var stmt: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(handle, sql, -1, &stmt, nil), SQLITE_OK)
        defer { sqlite3_finalize(stmt) }
        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, sourceFile, -1, transient)
        sqlite3_bind_text(stmt, 2, createdAt, -1, transient)
        sqlite3_bind_text(stmt, 3, id, -1, transient)
        XCTAssertEqual(sqlite3_step(stmt), SQLITE_DONE)
    }

    private func waitForSelectedEntityLoad(_ vm: KGViewModel, iterations: Int = 200) async {
        for _ in 0..<iterations {
            if vm.selectedEntityChunkTotal == 25 && vm.selectedEntityChunks.count == 15 {
                return
            }
            await Task.yield()
        }
    }

    private func waitForSelectedEntityChunks(_ vm: KGViewModel, iterations: Int = 200) async {
        for _ in 0..<iterations {
            if !vm.selectedEntityChunks.isEmpty {
                return
            }
            await Task.yield()
        }
    }

    private func waitForSelectedEntityChunkCount(_ vm: KGViewModel, count: Int, iterations: Int = 200) async {
        for _ in 0..<iterations {
            if vm.selectedEntityChunks.count == count {
                return
            }
            await Task.yield()
        }
    }

    private func waitForSelectedEntityChunkLoadingToFinish(_ vm: KGViewModel, iterations: Int = 200) async {
        for _ in 0..<iterations {
            if !vm.isLoadingSelectedEntityChunks {
                return
            }
            await Task.yield()
        }
    }
}

final class BrainBarInjectionsPlaceholderTests: XCTestCase {
    deinit {}

    func testInjectionsTabShowsClearMessageWhenStoreNil() {
        let subtitle = BrainBarPlaceholderCopy.injectionFeedNotWired

        XCTAssertEqual(subtitle, "Injection feed not yet wired in this build.")
        XCTAssertFalse(subtitle.localizedCaseInsensitiveContains("unavailable"))
    }
}

@MainActor
final class KGCanvasSimulationTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-kgcanvas-test-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        super.tearDown()
    }

    func testStableGraphStopsWithinTwoSecondsWorthOfFrames() async {
        let vm = KGViewModel(database: db)
        vm.canvasCenter = CGPoint(x: 300, y: 250)
        vm.nodes = makeStableFixture(center: vm.canvasCenter)
        vm.edges = []

        let controller = KGSimulationController(
            frameDuration: .milliseconds(33),
            sleep: { _ in await Task.yield() }
        )

        var tickCount = 0
        controller.start {
            tickCount += 1
            return vm.tick()
        }

        await waitForSimulationToStop(controller)

        XCTAssertFalse(controller.timerActive, "Stable graph should auto-stop once kinetic energy is low")
        XCTAssertLessThanOrEqual(
            tickCount,
            60,
            "Stopping within 60 frames matches ~2s on the real 30fps timer"
        )
    }

    func testRestartAfterIdleStartsSimulationAgain() async {
        let controller = KGSimulationController(
            frameDuration: .milliseconds(33),
            sleep: { _ in await Task.yield() }
        )

        var tickCount = 0
        let tick: @MainActor () -> CGFloat = {
            tickCount += 1
            return 0
        }

        controller.start(tick: tick)
        await waitForSimulationToStop(controller)

        XCTAssertFalse(controller.timerActive, "Controller should go idle after crossing the energy threshold")

        controller.start(tick: tick)
        await waitForSimulationToStop(controller)

        XCTAssertEqual(tickCount, 2, "A previously idle simulation should restart on the next appearance")
    }

    private func waitForSimulationToStop(_ controller: KGSimulationController, iterations: Int = 200) async {
        for _ in 0..<iterations where controller.timerActive {
            await Task.yield()
        }
    }

    private func makeStableFixture(center: CGPoint) -> [KGNode] {
        let radius: CGFloat = 66.0
        return [
            KGNode(
                id: "a",
                name: "Alice",
                entityType: "person",
                importance: 5,
                position: CGPoint(x: center.x + radius, y: center.y),
                velocity: .zero
            ),
            KGNode(
                id: "b",
                name: "BrainLayer",
                entityType: "project",
                importance: 5,
                position: CGPoint(
                    x: center.x - radius / 2,
                    y: center.y + (sqrt(3) * radius / 2)
                ),
                velocity: .zero
            ),
            KGNode(
                id: "c",
                name: "Codex",
                entityType: "agent",
                importance: 5,
                position: CGPoint(
                    x: center.x - radius / 2,
                    y: center.y - (sqrt(3) * radius / 2)
                ),
                velocity: .zero
            ),
        ]
    }
}
