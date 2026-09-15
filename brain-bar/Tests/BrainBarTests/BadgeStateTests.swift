import Foundation
import XCTest
@testable import BrainBar

final class BadgeStateTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_789_459_200)

    private var producerFixtureURL: URL {
        fixtureURL(named: "badge-state-v1.json")
    }

    private var pendingFixtureURL: URL {
        fixtureURL(named: "badge-state-pending-v1.json")
    }

    private func fixtureURL(named name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("tests/fixtures/badge-state/" + name)
    }

    func testMissingBadgeStateWithoutPendingMarkerFailsVisibleWithBadgeOn() throws {
        let url = try mutatedFixture(source: pendingFixtureURL) { _ in }
        try FileManager.default.removeItem(at: url)

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertTrue(presentation.reason.localizedCaseInsensitiveContains("without a pending_first_run marker"))
    }

    func testPendingFirstRunDeadlineAndSleepAwareGraceMatrix() throws {
        let url = try mutatedFixture(source: pendingFixtureURL) { _ in }
        defer { try? FileManager.default.removeItem(at: url) }
        let cadence = ObservabilityCadence.known(300)

        XCTAssertFalse(BadgeStateReader.read(url: url, now: now, cadence: cadence).badgeOn)
        XCTAssertTrue(BadgeStateReader.read(url: url, now: now.addingTimeInterval(601), cadence: cadence).badgeOn)

        var history = BadgeReadHistory()
        _ = history.pendingFirstRunGrace(now: now, cadence: cadence)
        let wake = now.addingTimeInterval(3_600)
        let grace = history.pendingFirstRunGrace(now: wake, cadence: cadence)
        XCTAssertFalse(
            BadgeStateReader.read(url: url, now: wake, cadence: cadence, pendingFirstRunGraceUntil: grace).badgeOn
        )
        let awake = wake.addingTimeInterval(301)
        XCTAssertTrue(
            BadgeStateReader.read(
                url: url,
                now: awake,
                cadence: cadence,
                pendingFirstRunGraceUntil: history.pendingFirstRunGrace(now: awake, cadence: cadence)
            ).badgeOn
        )
    }

    func testUnknownCadenceFailsVisibleWithBadgeOn() throws {
        let url = try mutatedFixture { _ in }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(
            url: url,
            now: now,
            cadence: .init(interval: 300, assumption: "cadence unknown, assuming 300s")
        )

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertTrue(presentation.reason.localizedCaseInsensitiveContains("freshness unknown"))
    }

    func testStaleBadgeStateFailsVisibleWithBadgeOn() throws {
        let url = try mutatedFixture { $0["generated_at"] = timestamp(now.addingTimeInterval(-601)) }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertTrue(presentation.reason.localizedCaseInsensitiveContains("stale"))
    }

    func testCorruptBadgeStateFailsVisibleWithBadgeOn() throws {
        let url = temporaryURL()
        var data = try Data(contentsOf: producerFixtureURL)
        data.replaceSubrange(data.startIndex ... data.startIndex, with: Data("!".utf8))
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertTrue(presentation.reason.localizedCaseInsensitiveContains("unreadable"))
    }

    func testUnknownSchemaFailsVisibleWithBadgeOn() throws {
        let url = try mutatedFixture { $0["schema_version"] = 2 }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertTrue(presentation.reason.contains("schema_version 2"))
    }

    func testFutureBadgeStateFailsVisibleWithBadgeOn() throws {
        let url = try mutatedFixture { $0["generated_at"] = timestamp(now.addingTimeInterval(1)) }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertTrue(presentation.reason.localizedCaseInsensitiveContains("future"))
    }

    func testInconsistentBadgeStateFailsVisibleWithBadgeOn() throws {
        let url = try mutatedFixture {
            var alerts = try XCTUnwrap($0["alerts"] as? [String: Any])
            alerts["badge_on"] = false
            $0["alerts"] = alerts
        }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertTrue(presentation.reason.localizedCaseInsensitiveContains("inconsistent"))
    }

    func testMeasuredHealthyDocumentTurnsBadgeOff() throws {
        let url = try mutatedFixture {
            var alerts = try XCTUnwrap($0["alerts"] as? [String: Any])
            alerts["badge_on"] = false
            alerts["active"] = []
            $0["alerts"] = alerts
        }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertFalse(presentation.badgeOn)
        XCTAssertEqual(presentation.reason, "")
    }

    func testMeasuredDataLossDocumentTurnsBadgeOn() throws {
        let url = try mutatedFixture { _ in }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertEqual(presentation.activeCodes, ["jsonl_backup_attempt_failed"])
    }

    func testBadgeStateURLIsDatabaseRelativeWithEnvironmentOverride() {
        let databasePath = "/tmp/brainlayer-test/brainlayer.db"

        XCTAssertEqual(
            BadgeStateReader.url(dbPath: databasePath, environment: [:]).path,
            "/tmp/brainlayer-test/badge-state.json"
        )
        XCTAssertEqual(
            BadgeStateReader.url(
                dbPath: databasePath,
                environment: ["BRAINLAYER_BADGE_STATE_PATH": "/tmp/override-badge.json"]
            ).path,
            "/tmp/override-badge.json"
        )
    }

    private func temporaryURL() -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("badge-state-\(UUID().uuidString).json")
    }

    private func timestamp(_ date: Date) -> String {
        ISO8601DateFormatter().string(from: date)
    }

    private func mutatedFixture(
        source: URL? = nil,
        _ mutate: (inout [String: Any]) throws -> Void
    ) throws -> URL {
        var payload = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: source ?? producerFixtureURL)) as? [String: Any]
        )
        try mutate(&payload)
        let url = temporaryURL()
        try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys]).write(to: url)
        return url
    }
}
