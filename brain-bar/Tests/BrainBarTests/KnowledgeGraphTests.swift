// KnowledgeGraphTests.swift — RED tests for Knowledge Graph viewer.
//
// TDD: These tests are written FIRST, before implementation.
// Covers: kg_entity_chunks table, fetchKGEntities, fetchKGRelations,
//         fetchEntityChunks, linkEntityChunk, KGNode, KGEdge, KGViewModel.

import XCTest
import SQLite3
@testable import BrainBar

private final class DashboardChangeNotificationProbe {
    var count = 0

    deinit {}
}

private func dashboardChangeNotificationProbeCallback(
    center: CFNotificationCenter?,
    observer: UnsafeMutableRawPointer?,
    name: CFNotificationName?,
    object: UnsafeRawPointer?,
    userInfo: CFDictionary?
) {
    guard let observer else { return }
    let probe = Unmanaged<DashboardChangeNotificationProbe>.fromOpaque(observer).takeUnretainedValue()
    probe.count += 1
}

// MARK: - Sidebar UI Logic Tests

final class KGSidebarViewTests: XCTestCase {
    func testLinkedChunksEmptyStateCopyIsAvailableToSidebar() {
        XCTAssertEqual(KGSidebarView.noLinkedChunksMessage, "No linked chunks are stored yet.")
    }

    func testLoadMoreFooterIdentityChangesAfterRowsAppend() {
        XCTAssertNotEqual(
            KGSidebarView.chunkLoadMoreTriggerID(visibleCount: 15),
            KGSidebarView.chunkLoadMoreTriggerID(visibleCount: 30)
        )
        XCTAssertNotEqual(
            KGSidebarView.fileLoadMoreTriggerID(visibleCount: 15),
            KGSidebarView.fileLoadMoreTriggerID(visibleCount: 30)
        )
    }
}

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

    func testFetchKGEntitiesAggregatesAliasGroupChunkCount() throws {
        try db.insertEntity(id: "person-etan-heyman", type: "person", name: "Etan Heyman")
        try db.insertEntity(id: "person-etan", type: "person", name: "Etan")
        try db.insertEntity(id: "project-brainlayer", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "person-etan-heyman", targetId: "project-brainlayer", relationType: "builds")
        try insertAlias(alias: "Etan", entityId: "person-etan-heyman")
        try insertLinkedChunk(
            id: "canonical-chunk",
            entityId: "person-etan-heyman",
            content: "Canonical Etan chunk",
            sourceFile: "/tmp/canonical.md",
            createdAt: "2026-05-24T12:00:00Z",
            relevance: 0.9
        )
        try insertLinkedChunk(
            id: "alias-chunk",
            entityId: "person-etan",
            content: "Alias Etan chunk",
            sourceFile: "/tmp/alias.md",
            createdAt: "2026-05-24T12:01:00Z",
            relevance: 0.8
        )

        let entities = try db.fetchKGEntities()
        let etan = try XCTUnwrap(entities.first { $0.id == "person-etan-heyman" })

        XCTAssertEqual(etan.linkedChunkCount, 2)
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

    func testFetchEntitySidebarCountsAndPagesAggregateAliasGroup() throws {
        try db.insertEntity(id: "person-etan-heyman", type: "person", name: "Etan Heyman")
        try db.insertEntity(id: "person-etan", type: "person", name: "Etan")
        try insertAlias(alias: "Etan", entityId: "person-etan-heyman")
        try insertLinkedChunk(
            id: "canonical",
            entityId: "person-etan-heyman",
            content: "Canonical chunk",
            sourceFile: "/tmp/canonical.md",
            createdAt: "2026-05-24T12:00:00Z",
            relevance: 0.9
        )
        try insertLinkedChunk(
            id: "alias",
            entityId: "person-etan",
            content: "Alias chunk",
            sourceFile: "/tmp/alias.md",
            createdAt: "2026-05-24T12:01:00Z",
            relevance: 0.8
        )

        let page = try db.fetchEntityChunksPage(entityId: "person-etan-heyman", after: nil, limit: 10)
        let files = try db.fetchEntitySourceFiles(entityId: "person-etan-heyman", limit: 10, after: nil)

        XCTAssertEqual(try db.fetchEntityChunkCount(entityId: "person-etan-heyman"), 2)
        XCTAssertEqual(try db.fetchEntitySourceFileCount(entityId: "person-etan-heyman"), 2)
        XCTAssertEqual(page.rows.map(\.chunkID), ["canonical", "alias"])
        XCTAssertEqual(files.rows.map(\.sourceFile), ["/tmp/canonical.md", "/tmp/alias.md"])
    }

    func testAliasAggregationDoesNotMergeAmbiguousAliasSurfaces() throws {
        try db.insertEntity(id: "person-alex", type: "person", name: "Alex")
        try db.insertEntity(id: "person-alex-one", type: "person", name: "Alex One")
        try db.insertEntity(id: "person-alex-two", type: "person", name: "Alex Two")
        try insertAlias(alias: "Alex", entityId: "person-alex-one")
        try insertAlias(alias: "Alex", entityId: "person-alex-two")
        try insertLinkedChunk(
            id: "alias-only",
            entityId: "person-alex",
            content: "Alias row chunk",
            sourceFile: "/tmp/alex.md",
            createdAt: "2026-05-24T12:00:00Z",
            relevance: 0.9
        )
        try insertLinkedChunk(
            id: "canonical-one",
            entityId: "person-alex-one",
            content: "First canonical chunk",
            sourceFile: "/tmp/one.md",
            createdAt: "2026-05-24T12:01:00Z",
            relevance: 0.8
        )
        try insertLinkedChunk(
            id: "canonical-two",
            entityId: "person-alex-two",
            content: "Second canonical chunk",
            sourceFile: "/tmp/two.md",
            createdAt: "2026-05-24T12:02:00Z",
            relevance: 0.7
        )

        let page = try db.fetchEntityChunksPage(entityId: "person-alex", after: nil, limit: 10)

        XCTAssertEqual(try db.fetchEntityChunkCount(entityId: "person-alex"), 1)
        XCTAssertEqual(page.rows.map(\.chunkID), ["alias-only"])
    }

    func testCanonicalEntityDoesNotMergeIntoAnotherEntityUniqueAlias() throws {
        try db.insertEntity(id: "person-alex", type: "person", name: "Alex")
        try db.insertEntity(id: "person-alexander", type: "person", name: "Alexander")
        try insertAlias(alias: "Alex", entityId: "person-alexander")
        try insertLinkedChunk(
            id: "alex-canonical",
            entityId: "person-alex",
            content: "Real Alex canonical chunk",
            sourceFile: "/tmp/alex.md",
            createdAt: "2026-05-24T12:00:00Z",
            relevance: 0.9
        )
        try insertLinkedChunk(
            id: "alexander-canonical",
            entityId: "person-alexander",
            content: "Alexander canonical chunk",
            sourceFile: "/tmp/alexander.md",
            createdAt: "2026-05-24T12:01:00Z",
            relevance: 0.8
        )

        let page = try db.fetchEntityChunksPage(entityId: "person-alex", after: nil, limit: 10)

        XCTAssertEqual(try db.fetchEntityChunkCount(entityId: "person-alex"), 1)
        XCTAssertEqual(page.rows.map(\.chunkID), ["alex-canonical"])
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

    func testFetchEntitySourceFilesAndCountIgnoreEmptyFiles() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try insertLinkedChunk(
            id: "file-1",
            entityId: "e1",
            content: "Real file chunk",
            sourceFile: "/tmp/real.md",
            createdAt: "2026-05-24T12:00:00Z",
            relevance: 0.9
        )
        try insertLinkedChunk(
            id: "empty-file",
            entityId: "e1",
            content: "Empty source chunk",
            sourceFile: "/tmp/to-empty.md",
            createdAt: "2026-05-24T12:01:00Z",
            relevance: 0.8
        )
        try insertLinkedChunk(
            id: "another-empty-file",
            entityId: "e1",
            content: "Another empty source chunk",
            sourceFile: "/tmp/another-empty.md",
            createdAt: "2026-05-24T12:02:00Z",
            relevance: 0.7
        )
        try setChunkSourceFile(id: "empty-file", sourceFile: "")
        try setChunkSourceFile(id: "another-empty-file", sourceFile: "")

        let page = try db.fetchEntitySourceFiles(entityId: "e1", limit: 10, after: nil)

        XCTAssertEqual(try db.fetchEntitySourceFileCount(entityId: "e1"), 1)
        XCTAssertEqual(page.rows.map(\.sourceFile), ["/tmp/real.md"])
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

    func testFetchEntitySidebarSnapshotReturnsTotalsAndFirstPagesTogether() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        for i in 0..<5 {
            try insertLinkedChunk(
                id: "c-\(i)",
                entityId: "e1",
                content: "Snapshot chunk \(i)",
                sourceFile: i < 3 ? "/tmp/a.md" : "/tmp/b.md",
                createdAt: String(format: "2026-05-24T12:%02d:00Z", i),
                relevance: Double(i) / 10.0
            )
        }

        let snapshot = try db.fetchEntitySidebarSnapshot(entityId: "e1", chunkLimit: 3, fileLimit: 1)

        XCTAssertEqual(snapshot.chunkTotal, 5)
        XCTAssertEqual(snapshot.fileTotal, 2)
        XCTAssertEqual(snapshot.chunkPage.rows.map(\.chunkID), ["c-4", "c-3", "c-2"])
        XCTAssertNotNil(snapshot.chunkPage.nextCursor)
        XCTAssertEqual(snapshot.filePage.rows.map(\.sourceFile), ["/tmp/b.md"])
        XCTAssertNotNil(snapshot.filePage.nextCursor)
    }

    func testFetchEntitySidebarSnapshotDoesNotPostDashboardChangeNotification() throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try insertLinkedChunk(
            id: "c-1",
            entityId: "e1",
            content: "Snapshot notification chunk",
            sourceFile: "/tmp/a.md",
            createdAt: "2026-05-24T12:00:00Z",
            relevance: 0.5
        )

        let probe = DashboardChangeNotificationProbe()
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterAddObserver(
            center,
            Unmanaged.passUnretained(probe).toOpaque(),
            dashboardChangeNotificationProbeCallback,
            BrainDatabase.dashboardDidChangeNotification as CFString,
            nil,
            .deliverImmediately
        )
        defer {
            CFNotificationCenterRemoveObserver(
                center,
                Unmanaged.passUnretained(probe).toOpaque(),
                CFNotificationName(BrainDatabase.dashboardDidChangeNotification as CFString),
                nil
            )
        }

        _ = try db.fetchEntitySidebarSnapshot(entityId: "e1", chunkLimit: 3, fileLimit: 1)
        RunLoop.current.run(until: Date().addingTimeInterval(0.1))

        XCTAssertEqual(probe.count, 0)
    }

    func testConcurrentEntitySidebarSnapshotsSerializeOnOneConnection() async throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        for i in 0..<200 {
            try insertLinkedChunk(
                id: "parallel-\(i)",
                entityId: "e1",
                content: "Parallel snapshot chunk \(i)",
                sourceFile: "/tmp/parallel-\(i % 5).md",
                createdAt: String(format: "2026-05-24T12:%02d:00Z", i % 60),
                relevance: Double(i) / 1_000.0
            )
        }

        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<50 {
                group.addTask { [db] in
                    _ = try db!.fetchEntitySidebarSnapshot(entityId: "e1", chunkLimit: 15, fileLimit: 5)
                }
            }
            try await group.waitForAll()
        }
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

    private func insertAlias(alias: String, entityId: String) throws {
        guard let handle = db.dbHandle else {
            XCTFail("Expected database handle")
            return
        }
        let sql = "INSERT INTO kg_entity_aliases (alias, entity_id) VALUES (?, ?)"
        var stmt: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(handle, sql, -1, &stmt, nil), SQLITE_OK)
        defer { sqlite3_finalize(stmt) }
        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, alias, -1, transient)
        sqlite3_bind_text(stmt, 2, entityId, -1, transient)
        XCTAssertEqual(sqlite3_step(stmt), SQLITE_DONE)
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

    private func setChunkSourceFile(id: String, sourceFile: String) throws {
        guard let handle = db.dbHandle else {
            XCTFail("Expected database handle")
            return
        }
        let sql = "UPDATE chunks SET source_file = ? WHERE id = ?"
        var stmt: OpaquePointer?
        XCTAssertEqual(sqlite3_prepare_v2(handle, sql, -1, &stmt, nil), SQLITE_OK)
        defer { sqlite3_finalize(stmt) }
        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
        sqlite3_bind_text(stmt, 1, sourceFile, -1, transient)
        sqlite3_bind_text(stmt, 2, id, -1, transient)
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

    func testKGNodeRadiusScalesWithAliasAggregatedChunkCount() {
        let sparse = KGNode(id: "sparse", name: "Sparse", entityType: "person", importance: 5.0, linkedChunkCount: 0)
        let dense = KGNode(id: "dense", name: "Dense", entityType: "person", importance: 5.0, linkedChunkCount: 100)

        XCTAssertGreaterThan(dense.radius, sparse.radius)
    }

    func testKGNodeRadiusIgnoresNegativeLinkedChunkCount() {
        let zero = KGNode(id: "zero", name: "Zero", entityType: "person", importance: 5.0, linkedChunkCount: 0)
        let negative = KGNode(id: "negative", name: "Negative", entityType: "person", importance: 5.0, linkedChunkCount: -10)

        XCTAssertEqual(negative.radius, zero.radius)
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

    func testGraphDefaultsToTieredAltitudeOptionB() {
        let vm = KGViewModel(database: db)

        XCTAssertEqual(vm.layoutMode, .tieredAltitude)
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
        vm.setLayoutMode(.importance)
        await vm.loadGraph()
        let defaultPositions = Dictionary(uniqueKeysWithValues: vm.nodes.map { ($0.id, $0.position) })

        vm.updateCanvasSize(CGSize(width: 1_200, height: 800))

        XCTAssertEqual(vm.canvasCenter, CGPoint(x: 600, y: 400))
        let nodeA = try XCTUnwrap(vm.nodes.first { $0.id == "a" })
        let nodeB = try XCTUnwrap(vm.nodes.first { $0.id == "b" })
        XCTAssertNotEqual(nodeA.position, defaultPositions["a"])
        XCTAssertNotEqual(nodeB.position, defaultPositions["b"])
        XCTAssertEqual(nodeA.position.x, 312, accuracy: 0.001)
        XCTAssertEqual(nodeA.position.y, 448, accuracy: 0.001)
        XCTAssertEqual(nodeB.position.x, 600, accuracy: 0.001)
        XCTAssertEqual(nodeB.position.y, 240, accuracy: 0.001)
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

    func testLoadGraphIfNeededReusesCachedGraphOnReactivation() async throws {
        let reader = RecordingKnowledgeGraphReader(
            entities: [
                BrainDatabase.KGEntityRow(
                    id: "a",
                    name: "Alice",
                    entityType: "person",
                    description: nil,
                    importance: 5,
                    linkedChunkCount: 0
                ),
                BrainDatabase.KGEntityRow(
                    id: "b",
                    name: "BrainLayer",
                    entityType: "project",
                    description: nil,
                    importance: 5,
                    linkedChunkCount: 0
                ),
            ],
            relations: [
                BrainDatabase.KGRelationRow(
                    id: "rel-a-b",
                    sourceId: "a",
                    targetId: "b",
                    relationType: "builds",
                    validUntil: nil,
                    expiredAt: nil
                ),
            ]
        )
        let vm = KGViewModel(graphReader: reader)

        let firstLoad = await vm.loadGraphIfNeeded()
        let secondLoad = await vm.loadGraphIfNeeded()

        XCTAssertTrue(firstLoad)
        XCTAssertTrue(secondLoad)
        XCTAssertEqual(reader.entityFetchCount, 1)
        XCTAssertEqual(reader.relationFetchCount, 1)
        XCTAssertEqual(vm.nodes.count, 2)
        XCTAssertEqual(vm.edges.count, 1)
    }

    func testLoadGraphRepeatedlyReusesCachedGraphBeforeRefreshOnReactivation() async throws {
        let reader = RecordingKnowledgeGraphReader(
            entities: [
                BrainDatabase.KGEntityRow(
                    id: "a",
                    name: "Alice",
                    entityType: "person",
                    description: nil,
                    importance: 5,
                    linkedChunkCount: 0
                ),
                BrainDatabase.KGEntityRow(
                    id: "b",
                    name: "BrainLayer",
                    entityType: "project",
                    description: nil,
                    importance: 5,
                    linkedChunkCount: 0
                ),
            ],
            relations: [
                BrainDatabase.KGRelationRow(
                    id: "rel-a-b",
                    sourceId: "a",
                    targetId: "b",
                    relationType: "builds",
                    validUntil: nil,
                    expiredAt: nil
                ),
            ]
        )
        let vm = KGViewModel(graphReader: reader)
        let initialLoad = await vm.loadGraph()
        XCTAssertTrue(initialLoad)

        var successCallbackCount = 0
        var sleepCount = 0
        let loadedOnce = await vm.loadGraphRepeatedly(
            refreshDelay: .milliseconds(1),
            retryDelay: .milliseconds(1),
            sleep: { _ in
                sleepCount += 1
                throw CancellationError()
            },
            onSuccessfulLoad: {
                successCallbackCount += 1
            }
        )

        XCTAssertTrue(loadedOnce)
        XCTAssertEqual(successCallbackCount, 1)
        XCTAssertEqual(sleepCount, 1)
        XCTAssertEqual(reader.entityFetchCount, 1)
        XCTAssertEqual(reader.relationFetchCount, 1)
    }

    func testOwnerEntityRemainsAnchoredDuringSimulation() async throws {
        let reader = RecordingKnowledgeGraphReader(
            entities: [
                BrainDatabase.KGEntityRow(
                    id: "owner",
                    name: "Etan Heyman",
                    entityType: "person",
                    description: nil,
                    importance: 0.5,
                    linkedChunkCount: 10
                ),
                BrainDatabase.KGEntityRow(
                    id: "project",
                    name: "brainlayer",
                    entityType: "project",
                    description: nil,
                    importance: 9,
                    linkedChunkCount: 100
                ),
            ],
            relations: [
                BrainDatabase.KGRelationRow(
                    id: "owner-project",
                    sourceId: "owner",
                    targetId: "project",
                    relationType: "builds",
                    validUntil: nil,
                    expiredAt: nil
                ),
            ]
        )
        let vm = KGViewModel(graphReader: reader)
        vm.setLayoutMode(.importance)
        vm.updateCanvasSize(CGSize(width: 860, height: 500))

        let loaded = await vm.loadGraph()
        XCTAssertTrue(loaded)
        let initialOwner = try XCTUnwrap(vm.nodes.first { $0.id == "owner" })
        _ = vm.tick()
        let movedOwner = try XCTUnwrap(vm.nodes.first { $0.id == "owner" })

        XCTAssertEqual(initialOwner.position.x, movedOwner.position.x, accuracy: 0.001)
        XCTAssertEqual(initialOwner.position.y, movedOwner.position.y, accuracy: 0.001)
        XCTAssertGreaterThan(movedOwner.position.y, 300)
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

    func testSelectNodeWithMissingNodeDoesNotLookupArbitraryEntity() async throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        vm.selectNode(id: "a")
        XCTAssertEqual(vm.selectedEntity?.name, "Alice")

        vm.nodes.removeAll()
        vm.selectNode(id: "a")

        XCTAssertNil(vm.selectedEntity)
        XCTAssertFalse(vm.isLoadingSelectedEntityChunks)
        XCTAssertTrue(vm.selectedEntityChunks.isEmpty)
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

    func testSelectNodeClearsOpenConversationOverlay() async throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try db.insertEntity(id: "e2", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "e1", targetId: "e2", relationType: "builds")
        try db.insertChunk(
            id: "c1", content: "Alice built BrainLayer",
            sessionId: "s1", project: "test", contentType: "ai_code", importance: 7
        )

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        vm.openConversation(chunkID: "c1")
        XCTAssertNotNil(vm.selectedConversation)

        vm.selectNode(id: "e2")

        XCTAssertNil(vm.selectedConversation)
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

    func testReduceMotionTickFreezesLayoutAndReportsZeroEnergy() async throws {
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        vm.nodes[0].velocity = CGVector(dx: 10, dy: 3)
        vm.nodes[1].velocity = CGVector(dx: -4, dy: -7)

        let positionsBefore = vm.nodes.map(\.position)
        let energy = vm.tick(reduceMotionEnabled: true)

        XCTAssertEqual(energy, 0, "Reduce Motion should make force layout report settled energy")
        XCTAssertEqual(vm.nodes.map(\.position), positionsBefore, "Reduce Motion should freeze node positions")
        XCTAssertTrue(vm.nodes.allSatisfy { $0.velocity == .zero }, "Reduce Motion should zero any residual velocity")
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
        try db.insertEntity(id: "e2", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "e1", targetId: "e2", relationType: "mentions")
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
        XCTAssertTrue(vm.selectedEntityChunkSidebarLoadFailed)
        XCTAssertFalse(vm.selectedEntityFileSidebarLoadFailed)
    }

    func testLoadMoreFileFailureDoesNotMarkChunksFailed() async throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try db.insertEntity(id: "e2", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "e1", targetId: "e2", relationType: "mentions")
        for i in 0..<25 {
            try db.insertChunk(
                id: "file-fail-\(i)",
                content: "Alice source file memory \(i)",
                sessionId: "file-fail-\(i)",
                project: "test",
                contentType: "ai_code",
                importance: 5
            )
            try updateChunk(id: "file-fail-\(i)", sourceFile: "/tmp/source-\(i).md", createdAt: String(format: "2026-05-24T15:%02d:00Z", i))
            try db.linkEntityChunk(entityId: "e1", chunkId: "file-fail-\(i)", relevance: Double(i) / 100.0)
        }

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        vm.selectNode(id: "e1")
        await waitForSelectedEntityLoad(vm)
        XCTAssertTrue(vm.selectedEntityCanLoadMoreFiles)

        db.close()
        vm.loadMoreFiles()
        await waitForSelectedEntityFileLoadingToFinish(vm)

        XCTAssertEqual(vm.selectedEntityChunks.count, 15)
        XCTAssertEqual(vm.selectedEntityFiles.count, 15)
        XCTAssertFalse(vm.selectedEntityCanLoadMoreFiles)
        XCTAssertFalse(vm.isLoadingSelectedEntityFiles)
        XCTAssertFalse(vm.selectedEntityChunkSidebarLoadFailed)
        XCTAssertTrue(vm.selectedEntityFileSidebarLoadFailed)
    }

    func testInitialSidebarFetchFailureIsExplicitlyFlagged() async throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")
        try db.insertEntity(id: "e2", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "e1", targetId: "e2", relationType: "mentions")

        let vm = KGViewModel(database: db)
        await vm.loadGraph()
        db.close()

        vm.selectNode(id: "e1")
        await waitForSelectedEntityChunkLoadingToFinish(vm)

        XCTAssertTrue(vm.selectedEntityChunkSidebarLoadFailed)
        XCTAssertTrue(vm.selectedEntityFileSidebarLoadFailed)
        XCTAssertEqual(vm.selectedEntityChunkTotal, 0)
        XCTAssertEqual(vm.selectedEntityFileTotal, 0)
        XCTAssertTrue(vm.selectedEntityChunks.isEmpty)
        XCTAssertTrue(vm.selectedEntityFiles.isEmpty)
    }

    func testFetchEntitySidebarSnapshotHonorsCancellationBeforeDatabaseWork() async throws {
        try db.insertEntity(id: "e1", type: "person", name: "Alice")

        XCTAssertThrowsError(try db.fetchEntitySidebarSnapshot(
            entityId: "e1",
            chunkLimit: 15,
            fileLimit: 15,
            shouldCancel: { true }
        )) { error in
            XCTAssertTrue(error is CancellationError)
        }
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

    private func waitForSelectedEntityFileLoadingToFinish(_ vm: KGViewModel, iterations: Int = 200) async {
        for _ in 0..<iterations {
            if !vm.isLoadingSelectedEntityFiles {
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
        vm.setLayoutMode(.importance)
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

    func testInactiveSimulationDoesNotTickUntilActivated() async {
        let controller = KGSimulationController(
            frameDuration: .milliseconds(33),
            sleep: { _ in await Task.yield() }
        )

        var tickCount = 0
        controller.setActive(false)
        controller.start {
            tickCount += 1
            return 1
        }

        await Task.yield()
        await Task.yield()

        XCTAssertFalse(controller.timerActive)
        XCTAssertEqual(tickCount, 0)

        controller.setActive(true)
        controller.start {
            tickCount += 1
            return 0
        }
        await waitForSimulationToStop(controller)

        XCTAssertEqual(tickCount, 1)
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

private final class RecordingKnowledgeGraphReader: KnowledgeGraphReading, @unchecked Sendable {
    private let entities: [BrainDatabase.KGEntityRow]
    private let relations: [BrainDatabase.KGRelationRow]
    private(set) var entityFetchCount = 0
    private(set) var relationFetchCount = 0

    init(
        entities: [BrainDatabase.KGEntityRow],
        relations: [BrainDatabase.KGRelationRow]
    ) {
        self.entities = entities
        self.relations = relations
    }

    deinit {}

    func fetchKGEntities(limit: Int) throws -> [BrainDatabase.KGEntityRow] {
        entityFetchCount += 1
        return Array(entities.prefix(limit))
    }

    func fetchKGRelations(limit: Int) throws -> [BrainDatabase.KGRelationRow] {
        relationFetchCount += 1
        return Array(relations.prefix(limit))
    }
}
