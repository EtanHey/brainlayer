// MCP entity lookup and digest schema contracts.
import XCTest
import SQLite3
import AppKit
@testable import BrainBar

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

    func testEntityCacheExcludesArchivedEntities() throws {
        guard let handle = db.dbHandle else {
            return XCTFail("Expected database handle")
        }
        XCTAssertEqual(
            sqlite3_exec(handle, "ALTER TABLE kg_entities ADD COLUMN status TEXT DEFAULT 'active'", nil, nil, nil),
            SQLITE_OK
        )
        try db.insertEntity(id: "person-active", type: "person", name: "Andrew Kelley")
        try db.insertEntity(id: "person-archived", type: "person", name: "But Ben")
        try db.insertEntity(id: "digest-entity-both-julius", type: "concept", name: "Both Julius")
        XCTAssertEqual(
            sqlite3_exec(handle, "UPDATE kg_entities SET status = 'archived' WHERE id = 'person-archived'", nil, nil, nil),
            SQLITE_OK
        )

        let cache = EntityCache()
        cache.load(from: handle)

        XCTAssertEqual(cache.detectEntities(in: "Andrew Kelley and But Ben").map(\.name), ["Andrew Kelley"])
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
        XCTAssertEqual(relation.targetEntityId, "company-domica")
        XCTAssertEqual(relation.direction, "outgoing")
        XCTAssertNotNil(relation.expiredAt)
        XCTAssertNotNil(relation.validUntil)
    }

    func testReadOnlyLegacyKGRelationSchemaDoesNotRequireExpirationColumns() throws {
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try createLegacyKGDatabaseWithoutRelationExpirationColumns(path: tempDBPath)

        let reader = BrainDatabase(path: tempDBPath, openConfiguration: .init(readOnly: true))
        defer { reader.close() }

        let payload = try XCTUnwrap(reader.lookupEntity(query: "Etan"))
        let card = EntityCard(lookupPayload: payload)

        XCTAssertEqual(card.relations.first?.targetName, "Domica")
        XCTAssertEqual(card.relations.first?.targetEntityId, "company-domica")
        XCTAssertEqual(card.relations.first?.direction, "outgoing")
        XCTAssertNil(card.relations.first?.validUntil)
        XCTAssertNil(card.relations.first?.expiredAt)
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
