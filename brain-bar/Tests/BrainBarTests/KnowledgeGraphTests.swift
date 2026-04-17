// KnowledgeGraphTests.swift — RED tests for Knowledge Graph viewer.
//
// TDD: These tests are written FIRST, before implementation.
// Covers: kg_entity_chunks table, fetchKGEntities, fetchKGRelations,
//         fetchEntityChunks, linkEntityChunk, KGNode, KGEdge, KGViewModel.

import XCTest
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

    func testFetchKGRelationsMultiple() throws {
        try db.insertEntity(id: "a", type: "person", name: "A")
        try db.insertEntity(id: "b", type: "project", name: "B")
        try db.insertEntity(id: "c", type: "tool", name: "C")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")
        try db.insertRelation(sourceId: "a", targetId: "c", relationType: "uses")

        let relations = try db.fetchKGRelations()
        XCTAssertEqual(relations.count, 2)
    }

    func testFetchKGRelationsEmptyDB() throws {
        let relations = try db.fetchKGRelations()
        XCTAssertTrue(relations.isEmpty)
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

    func testLoadGraphPopulatesNodesAndEdges() throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        vm.loadGraph()

        XCTAssertEqual(vm.nodes.count, 2)
        XCTAssertEqual(vm.edges.count, 1)
    }

    func testLoadGraphEmptyDB() throws {
        let vm = KGViewModel(database: db)
        vm.loadGraph()

        XCTAssertTrue(vm.nodes.isEmpty)
        XCTAssertTrue(vm.edges.isEmpty)
    }

    func testSelectNodeUpdatesSelectedEntity() throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")
        let vm = KGViewModel(database: db)
        vm.loadGraph()

        vm.selectNode(id: "a")
        XCTAssertEqual(vm.selectedNodeId, "a")
        XCTAssertNotNil(vm.selectedEntity)
        XCTAssertEqual(vm.selectedEntity?.name, "Alice")
    }

    func testSelectNodeNilDeselects() throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")
        let vm = KGViewModel(database: db)
        vm.loadGraph()

        vm.selectNode(id: "a")
        vm.selectNode(id: nil)
        XCTAssertNil(vm.selectedNodeId)
        XCTAssertNil(vm.selectedEntity)
    }

    func testForceTickMovesNodes() throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        vm.loadGraph()

        let positionsBefore = vm.nodes.map(\.position)
        vm.tick() // one simulation step
        let positionsAfter = vm.nodes.map(\.position)

        // At least one node should have moved
        let moved = zip(positionsBefore, positionsAfter).contains { $0 != $1 }
        XCTAssertTrue(moved, "Force tick should move at least one node")
    }

    func testNodeHitTestFindsCorrectNode() throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")
        let vm = KGViewModel(database: db)
        vm.loadGraph()

        // Place node at known position
        vm.nodes[0].position = CGPoint(x: 100, y: 100)

        let hit = vm.nodeAt(point: CGPoint(x: 102, y: 98)) // within radius
        XCTAssertEqual(hit?.id, "a")

        let miss = vm.nodeAt(point: CGPoint(x: 500, y: 500)) // far away
        XCTAssertNil(miss)
    }

    func testSelectedEntityChunksPopulated() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try db.insertEntity(id: "e2", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "e1", targetId: "e2", relationType: "builds")
        try db.insertChunk(
            id: "c1", content: "Alice built BrainLayer",
            sessionId: "s1", project: "test", contentType: "ai_code", importance: 7
        )
        try db.linkEntityChunk(entityId: "e1", chunkId: "c1", relevance: 0.9)

        let vm = KGViewModel(database: db)
        vm.loadGraph()
        vm.selectNode(id: "e1")

        XCTAssertFalse(vm.selectedEntityChunks.isEmpty)
        XCTAssertTrue(vm.selectedEntityChunks.first?.snippet.contains("Alice") ?? false)
    }
}
