import XCTest
@testable import BrainBar

/// #966: one watcher-health truth, built from watcher-health.json beside the resolved DB plus
/// launchd state. Every attention state carries what, since when, and what to do.
final class WatcherHealthStatusTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)

    private func file(
        ageSeconds: TimeInterval = 70,
        alerts: [String] = [],
        dbProbeFailed: Bool = false,
        lagBytes: Int = 0,
        failures: Int = 0,
        failureSince: Date? = nil,
        quarantined: Int = 0,
        quarantineSince: Date? = nil
    ) -> WatcherHealthFileRead {
        .readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-ageSeconds),
            pollCount: 72,
            alertReasons: alerts,
            dbProbeFailed: dbProbeFailed,
            maxOffsetLagBytes: lagBytes,
            fileIngestionFailureCount: failures,
            earliestFileIngestionFailureAt: failureSince,
            quarantinedRecordCount: quarantined,
            earliestQuarantinedRecordAt: quarantineSince
        ))
    }

    private func issues(_ status: WatcherHealthStatus, file: StaticString = #filePath, line: UInt = #line) -> [WatcherHealthIssue] {
        guard case let .degraded(issues, _) = status else {
            XCTFail("expected degraded, got \(status)", file: file, line: line)
            return []
        }
        return issues
    }

    // MARK: running

    func testRunningWithFreshHeartbeatAndNoAlertsIsRunning() {
        let status = WatcherHealthStatus.derive(launchd: .running, file: file(), now: now)
        XCTAssertEqual(status, .running(heartbeatAt: now.addingTimeInterval(-70)))
        XCTAssertEqual(status.title, "Watcher running")
        XCTAssertFalse(status.needsAttention)
        XCTAssertNil(status.reasonText(now: now))
    }

    func testNormalHeartbeatSpacingIsNotStale() {
        // The watcher writes once per poll: ~60-95 s apart is normal (measured 91 s).
        for age in [60.0, 95.0, 180.0, 299.0] {
            XCTAssertEqual(
                WatcherHealthStatus.derive(launchd: .running, file: file(ageSeconds: age), now: now),
                .running(heartbeatAt: now.addingTimeInterval(-age)),
                "heartbeat \(Int(age)) s old must still read as running"
            )
        }
    }

    // MARK: running but degraded, one concrete reason each

    func testStaleHeartbeatIsDegradedSinceTheLastHeartbeat() throws {
        let lastBeat = now.addingTimeInterval(-900)
        let status = WatcherHealthStatus.derive(launchd: .running, file: file(ageSeconds: 900), now: now)
        let found = issues(status)
        let first = try XCTUnwrap(found.first)
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(found.first?.since, lastBeat)
        XCTAssertTrue(found.first?.what.localizedCaseInsensitiveContains("heartbeat") == true, "\(found)")
        XCTAssertTrue(status.needsAttention)
        XCTAssertEqual(status.title, "Watcher needs attention")
        XCTAssertEqual(status.reasonText(now: now), "\(first.what) · since 15m ago · \(first.action)")
    }

    func testStaleFileReportsOnlyStalenessNotItsOldAlerts() throws {
        // Alerts in a stale file describe the past; only the stale heartbeat is current.
        let status = WatcherHealthStatus.derive(
            launchd: .running,
            file: file(ageSeconds: 3_600, alerts: ["offset_lag"], lagBytes: 10_000_000),
            now: now
        )
        let found = issues(status)
        let first = try XCTUnwrap(found.first)
        XCTAssertEqual(found.count, 1)
        XCTAssertTrue(first.what.localizedCaseInsensitiveContains("heartbeat"), "\(found)")
    }

    func testCoverageDropIsDegradedAsOfTheHeartbeat() throws {
        let status = WatcherHealthStatus.derive(launchd: .running, file: file(alerts: ["coverage_drop"]), now: now)
        let found = issues(status)
        let first = try XCTUnwrap(found.first)
        XCTAssertEqual(found.count, 1)
        XCTAssertNil(first.since, "the watcher does not publish when coverage started dropping")
        XCTAssertTrue(first.what.localizedCaseInsensitiveContains("landing"), first.what)
        XCTAssertEqual(status.reasonText(now: now), "\(first.what) · as of 1m ago · \(first.action)")
    }

    func testOffsetLagNamesHowFarBehind() throws {
        let status = WatcherHealthStatus.derive(
            launchd: .running,
            file: file(alerts: ["offset_lag"], lagBytes: 3 * 1_048_576),
            now: now
        )
        let found = issues(status)
        let first = try XCTUnwrap(found.first)
        XCTAssertEqual(found.count, 1)
        XCTAssertTrue(first.what.contains("3 MB"), first.what)
    }

    func testFileIngestionFailureIsDegradedSinceTheFirstFailure() throws {
        let firstFailure = now.addingTimeInterval(-7_200)
        let status = WatcherHealthStatus.derive(
            launchd: .running,
            file: file(alerts: ["file_ingestion_failure"], failures: 2, failureSince: firstFailure),
            now: now
        )
        let found = issues(status)
        let first = try XCTUnwrap(found.first)
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(first.since, firstFailure)
        XCTAssertTrue(first.what.hasPrefix("2 transcript files"), first.what)
    }

    func testQuarantinedRecordIsDegradedSinceTheFirstQuarantine() throws {
        let firstQuarantine = now.addingTimeInterval(-600)
        let status = WatcherHealthStatus.derive(
            launchd: .running,
            file: file(alerts: ["quarantined_record"], quarantined: 1, quarantineSince: firstQuarantine),
            now: now
        )
        let found = issues(status)
        let first = try XCTUnwrap(found.first)
        XCTAssertEqual(found.count, 1)
        XCTAssertEqual(first.since, firstQuarantine)
        XCTAssertTrue(first.what.hasPrefix("1 transcript record quarantined"), first.what)
    }

    func testDatabaseProbeFailureIsDegraded() throws {
        let status = WatcherHealthStatus.derive(launchd: .running, file: file(dbProbeFailed: true), now: now)
        let found = issues(status)
        let first = try XCTUnwrap(found.first)
        XCTAssertEqual(found.count, 1)
        XCTAssertTrue(first.what.localizedCaseInsensitiveContains("database"), first.what)
    }

    func testUnrecognisedAlertIsNamedNotDropped() throws {
        let status = WatcherHealthStatus.derive(launchd: .running, file: file(alerts: ["disk_full"]), now: now)
        let found = issues(status)
        let first = try XCTUnwrap(found.first)
        XCTAssertEqual(found.count, 1)
        XCTAssertTrue(first.what.contains("disk_full"), first.what)
    }

    func testMultipleIssuesSummariseTheFirstAndCountTheRest() {
        let status = WatcherHealthStatus.derive(
            launchd: .running,
            file: file(alerts: ["coverage_drop", "offset_lag"], dbProbeFailed: true, lagBytes: 2 * 1_048_576),
            now: now
        )
        let found = issues(status)
        XCTAssertEqual(found.count, 3)
        XCTAssertTrue(status.reasonText(now: now)?.hasSuffix("(+2 more)") == true, status.reasonText(now: now) ?? "nil")
    }

    func testEveryIssueCarriesWhatAndAnAction() {
        let status = WatcherHealthStatus.derive(
            launchd: .running,
            file: file(
                alerts: ["coverage_drop", "offset_lag", "file_ingestion_failure", "quarantined_record", "other"],
                dbProbeFailed: true,
                failures: 1,
                failureSince: now,
                quarantined: 1,
                quarantineSince: now
            ),
            now: now
        )
        let found = issues(status)
        XCTAssertEqual(found.count, 6)
        for issue in found {
            XCTAssertFalse(issue.what.isEmpty)
            XCTAssertFalse(issue.action.isEmpty, "\(issue.what) has no action")
        }
    }

    // MARK: stopped

    func testLaunchdNotRunningIsStoppedEvenWithAFreshHeartbeat() {
        let status = WatcherHealthStatus.derive(
            launchd: .notRunning("com.brainlayer.watch is not loaded"),
            file: file(),
            now: now
        )
        guard case let .stopped(reason) = status else { return XCTFail("expected stopped, got \(status)") }
        XCTAssertTrue(reason.contains("com.brainlayer.watch is not loaded"), reason)
        XCTAssertEqual(status.title, "Watcher stopped")
        XCTAssertTrue(status.needsAttention)
        XCTAssertEqual(status.reasonText(now: now), reason)
    }

    // MARK: unknown, never a false running or down

    func testRunningButHealthFileMissingIsUnknownWithThePath() {
        let status = WatcherHealthStatus.derive(
            launchd: .running,
            file: .missing(path: "/data/watcher-health.json"),
            now: now
        )
        guard case let .unknown(reason) = status else { return XCTFail("expected unknown, got \(status)") }
        XCTAssertTrue(reason.contains("/data/watcher-health.json"), reason)
        XCTAssertEqual(status.title, "Watcher status unknown")
        XCTAssertFalse(status.needsAttention, "unknown is not a claim that the watcher is down")
        XCTAssertEqual(status.reasonText(now: now), reason)
    }

    func testRunningButHealthFileUnreadableIsUnknownWithTheReason() {
        let status = WatcherHealthStatus.derive(
            launchd: .running,
            file: .unreadable(path: "/data/watcher-health.json", reason: "not JSON"),
            now: now
        )
        guard case let .unknown(reason) = status else { return XCTFail("expected unknown, got \(status)") }
        XCTAssertTrue(reason.contains("not JSON"), reason)
    }

    func testHealthNotYetReadIsUnknown() {
        guard case .unknown = WatcherHealthStatus.derive(launchd: .running, file: nil, now: now) else {
            return XCTFail("an unsampled health file must not read as running")
        }
    }

    func testLaunchdUnavailableWithFreshHeartbeatTrustsTheHeartbeat() {
        XCTAssertEqual(
            WatcherHealthStatus.derive(launchd: .unavailable("launchctl timed out"), file: file(), now: now),
            .running(heartbeatAt: now.addingTimeInterval(-70))
        )
    }

    func testLaunchdUnavailableWithoutAFreshHeartbeatIsUnknown() {
        for read in [file(ageSeconds: 900), WatcherHealthFileRead.missing(path: "/x/watcher-health.json")] {
            let status = WatcherHealthStatus.derive(launchd: .unavailable("launchctl timed out"), file: read, now: now)
            guard case let .unknown(reason) = status else { return XCTFail("expected unknown, got \(status)") }
            XCTAssertTrue(reason.contains("launchctl timed out"), reason)
        }
    }

    // MARK: launchd evidence from each surface's probe

    func testDashboardProcessProbeMapsToLaunchdEvidence() {
        XCTAssertEqual(WatcherLaunchdEvidence(process: .running(pid: 42)), .running)
        guard case .notRunning = WatcherLaunchdEvidence(process: .absent) else { return XCTFail("absent must be notRunning") }
        XCTAssertEqual(WatcherLaunchdEvidence(process: .failure("launchctl timed out")), .unavailable("launchctl timed out"))
        guard case .unavailable = WatcherLaunchdEvidence(process: nil) else { return XCTFail("nil must be unavailable") }
    }

    func testSettingsLaunchdObservationMapsToLaunchdEvidence() {
        let enabled = BrainLayerLaunchdJobSetting(enabled: true, loadState: .running)
        XCTAssertEqual(WatcherLaunchdEvidence(setting: enabled, loadState: .running), .running)
        guard case .notRunning = WatcherLaunchdEvidence(setting: enabled, loadState: .loaded) else { return XCTFail("loaded-not-running") }
        guard case .notRunning = WatcherLaunchdEvidence(setting: enabled, loadState: .unloaded) else { return XCTFail("unloaded") }
        guard case .unavailable = WatcherLaunchdEvidence(setting: enabled, loadState: .probeError("x")) else { return XCTFail("probe error") }
        guard case .unavailable = WatcherLaunchdEvidence(setting: enabled, loadState: nil) else { return XCTFail("no observation") }
        let disabled = BrainLayerLaunchdJobSetting(enabled: false, loadState: .unloaded)
        guard case let .notRunning(detail) = WatcherLaunchdEvidence(setting: disabled, loadState: .unloaded) else {
            return XCTFail("disabled job must be notRunning")
        }
        XCTAssertTrue(detail.localizedCaseInsensitiveContains("disabled"), detail)
    }

    // MARK: reader

    func testReaderResolvesBesideTheDatabaseUnlessOverridden() {
        XCTAssertEqual(
            WatcherHealthReader.url(dbPath: "/data/brainlayer.db", environment: [:]).path,
            "/data/watcher-health.json"
        )
        XCTAssertEqual(
            WatcherHealthReader.url(
                dbPath: "/data/brainlayer.db",
                environment: ["BRAINLAYER_WATCHER_HEALTH_PATH": "/elsewhere/h.json"]
            ).path,
            "/elsewhere/h.json"
        )
    }

    func testReaderDistinguishesMissingUnreadableAndReadable() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let missing = directory.appendingPathComponent("watcher-health.json")
        XCTAssertEqual(WatcherHealthReader.read(url: missing), .missing(path: missing.path))

        try Data("not json".utf8).write(to: missing)
        guard case .unreadable = WatcherHealthReader.read(url: missing) else { return XCTFail("garbage must be unreadable") }

        try Data(#"{"poll_count": 3}"#.utf8).write(to: missing)
        guard case let .unreadable(_, reason) = WatcherHealthReader.read(url: missing) else {
            return XCTFail("a file without updated_at cannot prove liveness")
        }
        XCTAssertTrue(reason.contains("updated_at"), reason)

        let payload = #"""
        {"updated_at": "2026-09-29T21:34:07.711548+00:00", "poll_count": 72, "alerting": true,
         "alert_reasons": ["file_ingestion_failure", "quarantined_record"], "db_probe_failed": false,
         "max_offset_lag_bytes": 12, "file_ingestion_failure_count": 2,
         "file_ingestion_failures": [{"observed_at": "2026-09-29T20:00:00+00:00"},
                                     {"observed_at": "2026-09-29T19:00:00+00:00"}],
         "quarantined_record_count_total": 1,
         "quarantined_records": [{"observed_at": "2026-09-29T21:00:00.500000+00:00"}]}
        """#
        try Data(payload.utf8).write(to: missing)
        guard case let .readable(parsed) = WatcherHealthReader.read(url: missing) else {
            return XCTFail("valid health file must be readable")
        }
        let iso = ISO8601DateFormatter()
        XCTAssertEqual(parsed.updatedAt.timeIntervalSince1970, 1_790_717_647.711, accuracy: 0.01)
        XCTAssertEqual(parsed.pollCount, 72)
        XCTAssertEqual(parsed.alertReasons, ["file_ingestion_failure", "quarantined_record"])
        XCTAssertEqual(parsed.maxOffsetLagBytes, 12)
        XCTAssertEqual(parsed.fileIngestionFailureCount, 2)
        XCTAssertEqual(parsed.earliestFileIngestionFailureAt, iso.date(from: "2026-09-29T19:00:00Z"))
        XCTAssertEqual(parsed.quarantinedRecordCount, 1)
        XCTAssertEqual(parsed.earliestQuarantinedRecordAt!.timeIntervalSince1970, 1_790_715_600.5, accuracy: 0.01)
    }
}
