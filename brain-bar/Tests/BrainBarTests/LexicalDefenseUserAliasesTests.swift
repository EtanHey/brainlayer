// Optional runtime user alias source for the native search fallback (#995). Synthetic fixtures only.

import XCTest
@testable import BrainBar

final class LexicalDefenseUserAliasesTests: XCTestCase {
    private var directory: URL!
    private var db: BrainDatabase!
    private var dbPath: String!

    override func setUp() {
        super.setUp()
        directory = URL(fileURLWithPath: brainBarTestTemporaryPath())
            .appendingPathComponent("brainbar-user-aliases-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        dbPath = directory.appendingPathComponent("brainbar.db").path
        db = BrainDatabase(path: dbPath)
    }

    override func tearDown() {
        db.close()
        try? FileManager.default.removeItem(at: directory)
        super.tearDown()
    }

    private func write(_ json: String, to name: String = "aliases.json") throws -> URL {
        let url = directory.appendingPathComponent(name)
        try Data(json.utf8).write(to: url)
        return url
    }

    func testDefaultURLIsUserConfigAndEnvironmentOverrides() {
        let home = URL(fileURLWithPath: "/synthetic-home")
        XCTAssertEqual(
            LexicalDefenseUserAliases.defaultURL(environment: [:], homeDirectory: home).path,
            "/synthetic-home/.config/brainlayer/lexical_defense.local.json"
        )
        XCTAssertEqual(
            LexicalDefenseUserAliases.defaultURL(
                environment: [LexicalDefenseUserAliases.environmentKey: "/synthetic/a.json"], homeDirectory: home
            ).path,
            "/synthetic/a.json"
        )
    }

    func testNativeSearchExpandsUserAliasOnlyWhenTheFileIsPresent() throws {
        try db.insertChunk(
            id: "user-alias-hit",
            content: "Zorbleth signed off on the synthetic rollout.",
            sessionId: "session-user-alias",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 8
        )
        let url = directory.appendingPathComponent("aliases.json")
        db.userLexicalAliases = LexicalDefenseUserAliases(url: url)
        XCTAssertTrue(try db.search(query: "qwvx", limit: 10).isEmpty, "absent file must not change search")

        _ = try write(#"{"version": 1, "entries": [{"canonical": "Zorbleth", "aliases": ["qwvx"]}]}"#)
        let results = try db.search(query: "qwvx", limit: 10)

        XCTAssertEqual(results.compactMap { $0["chunk_id"] as? String }, ["user-alias-hit"])
    }

    func testEditsApplyOnNextLookupAndDeletionRestoresBaseline() throws {
        let url = try write(#"{"version": 1, "entries": [{"canonical": "Zorbleth", "aliases": ["qwvx"]}]}"#)
        let aliases = LexicalDefenseUserAliases(url: url)
        XCTAssertEqual(aliases.replacements(for: "QWVX"), ["Zorbleth", "qwvx"])

        _ = try write(#"{"version": 1, "entries": [{"canonical": "Zorbleth", "aliases": ["qwvx", "zblt"]}]}"#)
        XCTAssertEqual(aliases.replacements(for: "zblt"), ["Zorbleth", "qwvx", "zblt"])

        try FileManager.default.removeItem(at: url)
        XCTAssertEqual(aliases.replacements(for: "qwvx"), [])
    }

    func testMalformedFilesAreRejectedWhole() throws {
        let long = String(repeating: "q", count: LexicalDefenseUserAliases.maxSurfaceChars + 1)
        let many = (0...LexicalDefenseUserAliases.maxSurfacesPerEntry).map { "\"qw\($0)\"" }.joined(separator: ",")
        let entries = (0...LexicalDefenseUserAliases.maxEntries).map { "{\"canonical\": \"Zorb\($0)\"}" }.joined(separator: ",")
        let cases: [String: String] = [
            "not_json": "{not json",
            "wrong_version": #"{"version": 2, "entries": []}"#,
            "bool_version": #"{"version": true, "entries": []}"#,
            "unknown_top_key": #"{"version": 1, "entries": [], "extra": 1}"#,
            "entry_not_object": #"{"version": 1, "entries": ["Zorbleth"]}"#,
            "missing_canonical": #"{"version": 1, "entries": [{"aliases": ["qwvx"]}]}"#,
            "unknown_entry_key": #"{"version": 1, "entries": [{"canonical": "Zorbleth", "category": "x"}]}"#,
            "surface_not_string": #"{"version": 1, "entries": [{"canonical": "Zorbleth", "aliases": [7]}]}"#,
            "punctuation_only": #"{"version": 1, "entries": [{"canonical": "Zorbleth", "aliases": ["-.-"]}]}"#,
            "surface_too_long": "{\"version\": 1, \"entries\": [{\"canonical\": \"\(long)\"}]}",
            "too_many_surfaces": "{\"version\": 1, \"entries\": [{\"canonical\": \"Zorbleth\", \"aliases\": [\(many)]}]}",
            "too_many_entries": "{\"version\": 1, \"entries\": [\(entries)]}",
            // Schema-valid JSON padded with whitespace: only the byte bound can reject it.
            "oversized": #"{"version": 1, "entries": [{"canonical": "Zorbleth", "aliases": ["qwvx"]}]}"#
                + String(repeating: " ", count: LexicalDefenseUserAliases.maxBytes),
        ]
        for (name, json) in cases {
            XCTAssertThrowsError(try LexicalDefenseUserAliases.parse(Data(json.utf8)), name)
            let url = try write(json, to: "\(name).json")
            let aliases = LexicalDefenseUserAliases(url: url)
            XCTAssertEqual(aliases.replacements(for: "qwvx"), [], name)
        }
        XCTAssertThrowsError(try LexicalDefenseUserAliases.parse(Data([0xFF, 0xFE, 0x7B, 0x7D])), "not_utf8")
    }

    func testConflictsSkipTheLaterEntryAndExpansionIsOneHop() throws {
        let parsed = try LexicalDefenseUserAliases.parse(
            Data(#"""
            {"version": 1, "entries": [
              {"canonical": "Zorbleth", "aliases": ["qwvx"]},
              {"canonical": "qwvx", "aliases": ["Plimsar"]},
              {"canonical": "Plimsar", "aliases": ["Zorbleth"]},
              {"canonical": "Quillmoor", "aliases": ["claimed-term"]},
              {"canonical": "Vantrel", "split_forms": ["van trel"]}
            ]}
            """#.utf8),
            claimed: ["claimedterm"]
        )

        XCTAssertEqual(parsed.skipped, ["qwvx", "Plimsar", "Quillmoor"])
        XCTAssertEqual(parsed.replacements["zorbleth"], ["Zorbleth", "qwvx"])
        XCTAssertNil(parsed.replacements["plimsar"])
        XCTAssertEqual(parsed.replacements["vantrel"], ["Vantrel", "van trel"])
    }
}
