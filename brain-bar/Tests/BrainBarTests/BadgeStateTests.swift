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

        let history = BadgeReadHistory()
        _ = BadgeStateReader.read(url: url, now: now, cadence: cadence, history: history)
        let wake = now.addingTimeInterval(3_600)
        XCTAssertFalse(BadgeStateReader.read(url: url, now: wake, cadence: cadence, history: history).badgeOn)
        let awake = wake.addingTimeInterval(301)
        XCTAssertTrue(BadgeStateReader.read(url: url, now: awake, cadence: cadence, history: history).badgeOn)
        XCTAssertTrue(
            BadgeStateReader.read(url: url, now: awake.addingTimeInterval(1_200), cadence: cadence, history: history)
                .badgeOn
        )

        let healthyURL = try mutatedFixture {
            var alerts = try XCTUnwrap($0["alerts"] as? [String: Any])
            alerts["badge_on"] = false
            alerts["active"] = []
            $0["alerts"] = alerts
        }
        defer { try? FileManager.default.removeItem(at: healthyURL) }
        let healthyHistory = BadgeReadHistory()
        XCTAssertFalse(BadgeStateReader.read(url: healthyURL, now: now, cadence: cadence, history: healthyHistory).badgeOn)
        XCTAssertFalse(BadgeStateReader.read(url: healthyURL, now: wake, cadence: cadence, history: healthyHistory).badgeOn)
        XCTAssertTrue(
            BadgeStateReader.read(url: healthyURL, now: awake, cadence: cadence, history: healthyHistory).badgeOn
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

    func testFuturePendingFirstRunFailsVisibleBeforePendingGrace() throws {
        let url = try mutatedFixture(source: pendingFixtureURL) {
            $0["generated_at"] = timestamp(now.addingTimeInterval(3_600))
            var alerts = try XCTUnwrap($0["alerts"] as? [String: Any])
            alerts["expected_first_run_by"] = timestamp(now.addingTimeInterval(4_200))
            $0["alerts"] = alerts
        }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertTrue(presentation.reason.localizedCaseInsensitiveContains("future"))
    }

    func testPendingFirstRunDeadlineCannotExceedGeneratedAtPlusTenMinutes() throws {
        let url = try mutatedFixture(source: pendingFixtureURL) {
            $0["generated_at"] = timestamp(now)
            var alerts = try XCTUnwrap($0["alerts"] as? [String: Any])
            alerts["expected_first_run_by"] = "2099-01-01T00:00:00Z"
            $0["alerts"] = alerts
        }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertTrue(presentation.badgeOn)
        XCTAssertTrue(presentation.reason.localizedCaseInsensitiveContains("deadline"))
    }

    func testProducerPendingFirstRunDeadlineAllowsFractionalSecondBeyondTenMinutes() throws {
        let generatedAtString = "2026-09-22T22:28:02Z"
        let now = try XCTUnwrap(ISO8601DateFormatter().date(from: generatedAtString)).addingTimeInterval(600)
        let url = try mutatedFixture(source: pendingFixtureURL) {
            $0["generated_at"] = generatedAtString
            var alerts = try XCTUnwrap($0["alerts"] as? [String: Any])
            alerts["expected_first_run_by"] = "2026-09-22T22:38:02.122198Z"
            $0["alerts"] = alerts
        }
        defer { try? FileManager.default.removeItem(at: url) }

        let presentation = BadgeStateReader.read(url: url, now: now, cadence: .known(300))

        XCTAssertFalse(presentation.badgeOn)
        XCTAssertEqual(presentation.reason, "awaiting first health-check run")
    }

    func testFutureBadgeStateFailsVisibleAfterSleepGrace() throws {
        let url = try mutatedFixture {
            $0["generated_at"] = timestamp(now.addingTimeInterval(3_601))
            var alerts = try XCTUnwrap($0["alerts"] as? [String: Any])
            alerts["badge_on"] = false
            alerts["active"] = []
            $0["alerts"] = alerts
        }
        defer { try? FileManager.default.removeItem(at: url) }
        let history = BadgeReadHistory()
        let cadence = ObservabilityCadence.known(300)
        _ = BadgeStateReader.read(url: url, now: now, cadence: cadence, history: history)

        let presentation = BadgeStateReader.read(
            url: url, now: now.addingTimeInterval(3_600), cadence: cadence, history: history
        )
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

    func testBadgeStateOverrideExpandsHomeDirectory() {
        let name = "badge-state-\(UUID().uuidString).json"
        let actual = BadgeStateReader.url(
            dbPath: "/tmp/brainlayer.db",
            environment: ["BRAINLAYER_BADGE_STATE_PATH": "~/\(name)"]
        )
        XCTAssertEqual(actual.path, FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(name).path)
    }

    func testBadgeStateTildeExpansionIsExplicitForMacOS14() {
        let name = "badge-state-\(UUID().uuidString).json"
        XCTAssertEqual(
            BadgeStateReader.expandedTildePath("~/\(name)"),
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(name).path
        )
    }

    func testBadgeStateURLResolvesSymlinkedDatabase() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("badge-state-path-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let realDirectory = root.appendingPathComponent("real", isDirectory: true)
        let linkDirectory = root.appendingPathComponent("link", isDirectory: true)
        try FileManager.default.createDirectory(at: realDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: linkDirectory, withIntermediateDirectories: true)
        let realDB = realDirectory.appendingPathComponent("brainlayer.db")
        let linkedDB = linkDirectory.appendingPathComponent("brainlayer.db")
        try Data().write(to: realDB)
        try FileManager.default.createSymbolicLink(at: linkedDB, withDestinationURL: realDB)

        XCTAssertEqual(
            BadgeStateReader.url(dbPath: linkedDB.path, environment: [:]).path,
            realDirectory.appendingPathComponent("badge-state.json").path
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
