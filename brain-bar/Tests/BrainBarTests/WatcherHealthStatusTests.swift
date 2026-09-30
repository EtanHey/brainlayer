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

    // MARK: #1013 review round 1

    /// B1 (round 2): a heartbeat proves the watcher was alive AT that time, so it bounds the stop
    /// from above, never below. launchd records no stop time: the text says the stop time is
    /// unknown and gives the last heartbeat as a fact, with no "for at least" claim.
    func testStoppedGivesTheLastHeartbeatWithoutClaimingAStopDuration() {
        let status = WatcherHealthStatus.derive(
            launchd: .notRunning("com.brainlayer.watch is not loaded"),
            file: file(ageSeconds: 900),
            now: now
        )
        let text = status.reasonText(now: now) ?? ""
        XCTAssertEqual(
            text,
            "Watcher is not running (com.brainlayer.watch is not loaded) · stop time unknown; last heartbeat 15m ago · "
                + "Restart it from Settings → Jobs → Ingest, or check ~/Library/Logs/brainlayer/watch.err.log"
        )
        XCTAssertFalse(text.contains("at least"), text)
        XCTAssertFalse(text.contains("since"), text)
    }

    /// B1: with no readable heartbeat there is nothing to report but the unknown stop time.
    func testStoppedWithoutAHeartbeatSaysStopTimeUnknown() {
        for read in [WatcherHealthFileRead.missing(path: "/x/watcher-health.json"), nil] {
            let text = WatcherHealthStatus.derive(launchd: .notRunning("com.brainlayer.watch is not running"), file: read, now: now)
                .reasonText(now: now) ?? ""
            XCTAssertTrue(text.contains(" · stop time unknown — no readable heartbeat · "), text)
        }
    }

    /// B2: the producer lists at most 100 failures/quarantines. When the list is shorter than the
    /// count, the listed earliest is only a lower bound on the true start, and the text says so.
    func testCappedDetailListsGiveALowerBoundNotAFalseStart() throws {
        let listedEarliest = now.addingTimeInterval(-7_200)
        let capped = WatcherHealthStatus.derive(launchd: .running, file: .readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-70),
            pollCount: 72,
            alertReasons: ["quarantined_record"],
            quarantinedRecordCount: 150,
            quarantinedRecordsListed: 100,
            earliestQuarantinedRecordAt: listedEarliest
        )), now: now)
        let text = try XCTUnwrap(capped.reasonText(now: now))
        XCTAssertTrue(text.contains("150 transcript records quarantined · since at least 2h ago (100 of 150 listed) · "), text)

        let complete = WatcherHealthStatus.derive(launchd: .running, file: .readable(WatcherHealthFile(
            updatedAt: now.addingTimeInterval(-70),
            pollCount: 72,
            alertReasons: ["file_ingestion_failure"],
            fileIngestionFailureCount: 2,
            fileIngestionFailuresListed: 2,
            earliestFileIngestionFailureAt: listedEarliest
        )), now: now)
        XCTAssertTrue(complete.reasonText(now: now)?.contains("could not be ingested · since 2h ago · ") == true)
    }

    /// B2: the reader records how many details were listed, so truncation is visible to the model.
    func testReaderCountsListedDetails() {
        let failures = (0..<100).map { _ in #"{"observed_at": "2026-09-29T20:00:00+00:00"}"# }.joined(separator: ",")
        let payload = #"{"updated_at": "2026-09-29T21:34:07+00:00", "poll_count": 7, "alert_reasons": ["file_ingestion_failure"], "file_ingestion_failure_count": 101, "file_ingestion_failures": ["# + failures + "]}"
        guard case let .readable(parsed) = WatcherHealthReader.parse(Data(payload.utf8), path: "/x") else {
            return XCTFail("valid snapshot must be readable")
        }
        XCTAssertEqual(parsed.fileIngestionFailureCount, 101)
        XCTAssertEqual(parsed.fileIngestionFailuresListed, 100)
    }

    /// B3: the watcher always writes updated_at, poll_count and alert_reasons together. A fresh
    /// object missing any of them is not proof of a healthy watcher: it is unreadable, so Unknown.
    func testPartialSnapshotIsNeverRunning() {
        let partial = Data(#"{"updated_at": "2026-09-29T21:34:07+00:00"}"#.utf8)
        guard case let .unreadable(_, reason) = WatcherHealthReader.parse(partial, path: "/x/watcher-health.json") else {
            return XCTFail("a partial snapshot must be unreadable")
        }
        XCTAssertEqual(reason, "missing required field(s): poll_count, alert_reasons")
        guard case .unknown = WatcherHealthStatus.derive(
            launchd: .running,
            file: WatcherHealthReader.parse(partial, path: "/x/watcher-health.json"),
            now: now
        ) else { return XCTFail("a partial snapshot must read as unknown, never running") }
    }

    /// B3 (round 2): a snapshot whose required or alert fields carry an unexpected type cannot
    /// prove a healthy watcher. `"alert_reasons": [17]` once parsed as zero alerts and read Running.
    func testMalformedFieldTypesAreUnreadableNeverRunning() {
        let fresh = ISO8601DateFormatter().string(from: now.addingTimeInterval(-70))
        let base = #""updated_at": "\#(fresh)", "poll_count": 7, "alert_reasons": []"#
        let cases: [(json: String, reason: String)] = [
            (#"{"updated_at": "\#(fresh)", "poll_count": 7, "alert_reasons": [17]}"#, "alert_reasons must be a list of strings"),
            (#"{"updated_at": "\#(fresh)", "poll_count": 7, "alert_reasons": ["db_probe_failed", null]}"#, "alert_reasons must be a list of strings"),
            (#"{"updated_at": "\#(fresh)", "poll_count": 7, "alert_reasons": "db_probe_failed"}"#, "alert_reasons must be a list of strings"),
            (#"{"updated_at": 1800000000, "poll_count": 7, "alert_reasons": []}"#, "updated_at must be an ISO-8601 timestamp"),
            (#"{"updated_at": "\#(fresh)", "poll_count": "7", "alert_reasons": []}"#, "poll_count must be an integer"),
            (#"{"updated_at": "\#(fresh)", "poll_count": true, "alert_reasons": []}"#, "poll_count must be an integer"),
            (#"{"updated_at": "\#(fresh)", "poll_count": 7.5, "alert_reasons": []}"#, "poll_count must be an integer"),
            (#"{\#(base), "db_probe_failed": "yes"}"#, "db_probe_failed must be a boolean"),
            (#"{\#(base), "db_probe_failed": 1}"#, "db_probe_failed must be a boolean"),
            (#"{\#(base), "max_offset_lag_bytes": "big"}"#, "max_offset_lag_bytes must be an integer"),
            (#"{\#(base), "file_ingestion_failure_count": 1.5}"#, "file_ingestion_failure_count must be an integer"),
            (#"{\#(base), "quarantined_record_count_total": true}"#, "quarantined_record_count_total must be an integer"),
            (#"{\#(base), "file_ingestion_failures": {}}"#, "file_ingestion_failures must be a list"),
            (#"{\#(base), "quarantined_records": "none"}"#, "quarantined_records must be a list"),
        ]
        for (json, expected) in cases {
            let read = WatcherHealthReader.parse(Data(json.utf8), path: "/x/watcher-health.json")
            guard case let .unreadable(_, reason) = read else {
                XCTFail("\(json) must be unreadable, got \(read)")
                continue
            }
            XCTAssertEqual(reason, expected, json)
            let status = WatcherHealthStatus.derive(launchd: .running, file: read, now: now)
            guard case .unknown = status else {
                XCTFail("\(json) must read as unknown, never \(status)")
                continue
            }
        }
        // The well-typed shape with every optional field present still parses.
        let valid = #"{\#(base), "db_probe_failed": false, "max_offset_lag_bytes": 0, "file_ingestion_failure_count": 0, "quarantined_record_count_total": 0, "file_ingestion_failures": [], "quarantined_records": []}"#
        guard case .readable = WatcherHealthReader.parse(Data(valid.utf8), path: "/x") else {
            return XCTFail("a well-typed snapshot must stay readable")
        }
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
