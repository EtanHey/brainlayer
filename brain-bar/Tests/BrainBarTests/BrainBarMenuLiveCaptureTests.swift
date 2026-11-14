import Foundation
import XCTest
@testable import BrainBar

final class BrainBarMenuLiveCaptureTests: XCTestCase {
    typealias C = BrainBarMenuCapture
    // Every stored field, including nested fields. Source paths name values, not invented fallbacks.
    static let fields: [String: String] = [
        "now": "mapped from caller.now", "timeZone": "mapped from caller.timeZone",
        "verdict": "mapped from publishedStatus.healthVerdict ?? staleHealthVerdict",
        "verdictAwakeAgeS": "mapped from caller sleepHistory.elapsed; unavailable -> infinity",
        "isChecking": "mapped from publishedStatus.isChecking",
        "wakeCatchUpPending": "mapped from caller wakeCatchUp.isPending(uptime: awakeUptime)",
        "backupsMeasured": "mapped from measured section only with no backupTruth.statusNotice",
        "lastKnownBackupAt": "not carried by the app yet",
        "watcher": "mapped from watcherHealth readable file; otherwise nil",
        "watcher.readable": "mapped from readable file case as constant true; missing/unreadable gives nil watcher",
        "watcher.fresh": "mapped from caller sleepHistory.elapsed since file.updatedAt, 0...300 seconds",
        "watcher.alertReasons": "mapped from WatcherHealthFile.alertReasons",
        "watcher.dbProbeFailed": "mapped from WatcherHealthFile.dbProbeFailed",
        "upload": "mapped from measured backups.localSnapshot in_progress with both process fields",
        "upload.startedAt": "mapped from localSnapshot.uploadProcessStartedAt",
        "upload.live": "mapped from localSnapshot.uploadProcessLive",
        "offsiteRecovering": "not carried by the app yet",
        "offsiteRecovering.progress": "not carried by the app yet",
        "offsiteRecovering.eta": "not carried by the app yet",
        "offsiteRecovering.startedAt": "not carried by the app yet",
        "offsiteRecovering.updatedAt": "not carried by the app yet",
        "offsiteRecovering.cadenceS": "not carried by the app yet",
        "lagEpisodeStart": "not carried by the app yet",
        "maintenanceNightFailures": "mapped from caller-known complete nightly/weekly streak; unknown -> nil",
        "driveAuthModel": "mapped from DriveAuthModel.status.state (wording only)",
        "backupDailyFailures": "mapped from observations.backupDaily.calendarConsecutiveFailures; unknown remains nil",
        "backupDailyNextRunAt": "mapped from observations.backupDaily.nextRunAt",
        "jsonlBackupFailures": "mapped from observations.jsonlBackup.calendarConsecutiveFailures; unknown remains nil",
        "keywordCoverage": "mapped from collector.stats.ftsCoveragePercent / 100 when signalCoverageIsAvailable",
    ]

    static func fieldNames() -> Set<String> {
        let date = Date(timeIntervalSince1970: 0.0)
        var c = C(now: date, timeZone: TimeZone(secondsFromGMT: 0)!)
        c.watcher = .init()
        c.upload = .init(startedAt: date, live: true)
        c.offsiteRecovering = .init(startedAt: date, updatedAt: date, cadenceS: 30.0)
        var names: Set<String> = []
        func walk(_ value: Any, _ prefix: String) {
            for child in Mirror(reflecting: value).children {
                let path = prefix + child.label!
                names.insert(path)
                let nested = String(reflecting: type(of: child.value)).contains(String(reflecting: C.self) + ".")
                guard nested else { continue } // External value types (Date, verdict, etc.) keep their own contracts.
                var value = child.value
                if Mirror(reflecting: value).displayStyle == .optional {
                    guard let wrapped = Mirror(reflecting: value).children.first else {
                        XCTFail("Populate nested capture fixture at \(path)"); continue
                    }
                    value = wrapped.value
                }
                if Mirror(reflecting: value).displayStyle == .struct { walk(value, path + ".") }
            }
        }
        walk(c, "")
        return names
    }

    func testEveryFieldHasAnExplicitDecision() {
        XCTAssertEqual(Self.fieldNames(), Set(Self.fields.keys))
        XCTAssertTrue(Self.fields.values.allSatisfy { $0.hasPrefix("mapped from ") || $0 == "not carried by the app yet" })
    }

    func testAnIncompleteFieldTableIsRejected() {
        var incomplete = Self.fields
        incomplete.removeValue(forKey: "watcher.readable")
        XCTAssertNotEqual(Self.fieldNames(), Set(incomplete.keys))
    }

    /// Build the same value types used by StatusSnapshotTests; no models that poll, run commands or show UI.
    static func adapt(_ source: C, status: BrainBarStatusSnapshot? = nil, awake: TimeInterval? = 0.0,
                      watcherAge: TimeInterval? = 0.0, backupsNotice: String? = nil, upkeepKnown: Bool = true, uploadStart: Date? = nil, uploadLive: Bool? = nil,
                      editObservations: ([BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]) -> [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation]? = { $0 }) -> C {
        var backups = BrainBarStatusSnapshotTests.backups()
        if let upload = source.upload {
            backups.localSnapshot = .init(lastAt: nil, destination: nil, localVerified: true, uploaded: false,
                uploadState: "in_progress", uploadError: nil, uploadProcessLive: uploadLive ?? upload.live,
                uploadProcessStartedAt: uploadStart ?? upload.startedAt, localIntegrity: nil, localIntegrityPragma: nil)
        }
        var observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] = [
            .backupDaily: BrainBarStatusSnapshotTests.step(next: source.backupDailyNextRunAt),
            .jsonlBackup: BrainBarStatusSnapshotTests.step(), .maintenanceNightly: BrainBarStatusSnapshotTests.step(),
        ]
        observations[.backupDaily]?.calendarConsecutiveFailures = source.backupDailyFailures
        observations[.jsonlBackup]?.calendarConsecutiveFailures = source.jsonlBackupFailures
        if let count = source.maintenanceNightFailures {
            observations[.maintenanceNightly]?.calendarNightOutcome = .init(scheduledDate: "fixture", at: source.now,
                failed: count > 0, abandoned: false, consecutiveFailures: count)
        }
        let file = source.watcher.map { w -> WatcherHealthFileRead in
            if !w.readable { return .unreadable(path: "fixture", reason: "fixture") }
            return .readable(.init(updatedAt: source.now, pollCount: 1, alertReasons: w.alertReasons, dbProbeFailed: w.dbProbeFailed))
        }
        let verdict = source.verdict.map { v in
            BrainBarHealthVerdict(dot: v.checks.contains { $0.state == .red } ? .red
                : v.checks.contains { $0.state == .amber } ? .amber : .green,
                checks: v.checks, notes: v.notes, evaluatedAt: v.evaluatedAt)
        }
        return BrainBarMenuLiveCapture.capture(
            status: status ?? .init(issues: [], isChecking: source.isChecking, healthVerdict: verdict),
            verdictAwakeAgeS: awake, wakeCatchUpPending: source.wakeCatchUpPending,
            driveAuth: source.driveAuthModel.map { .init(state: $0, reason: nil, expiresAt: nil) },
            backups: source.backupsMeasured ? .measured(backups) : .unavailable("fixture"), backupsNotice: backupsNotice,
            watcher: file, watcherAwakeAgeS: watcherAge, observations: editObservations(observations), maintenanceNightStreakKnown: upkeepKnown,
            stats: .init(chunkCount: 100, enrichedChunkCount: 0, pendingEnrichmentCount: 0,
                enrichmentPercent: 0.0, enrichmentRatePerMinute: 0.0, databaseSizeBytes: 0,
                recentActivityBuckets: [], recentEnrichmentBuckets: [], ftsEligibleChunkCount: 100,
                ftsIndexedChunkCount: Int((source.keywordCoverage ?? 0.0) * 100.0),
                signalCoverageIsAvailable: source.keywordCoverage != nil),
            now: source.now, timeZone: source.timeZone)
    }

    func testSeventeenContractStatesThroughLiveTypes() throws {
        var receipts: [[String: Any]] = []
        for row in try BrainBarMenuPresentationTests.loadCases() {
            let id = row["id"] as! String
            var source = try BrainBarMenuCases.capture(row["inputs"] as! [String: Any])
            if ["S4", "S8"].contains(id) { source.watcher = nil; source.backupsMeasured = false }
            let c = Self.adapt(source, status: id == "S4" ? .checking : nil)
            var expected = BrainBarMenuPresentationTests.expected(row)
            // Missing seams have deliberately poorer, but truthful, whole presentations.
            if id == "S2-after-seam" {
                let today = try BrainBarMenuPresentationTests.loadCases().first { $0["id"] as? String == "S2-today" }!
                expected = BrainBarMenuPresentationTests.expected(today)
            }
            if id == "S4" {
                let launch = try BrainBarMenuPresentationTests.loadCases().first { $0["id"] as? String == "S8" }!
                expected = BrainBarMenuPresentationTests.expected(launch)
            }
            let actual = BrainBarMenuPresenter.present(c)
            XCTAssertEqual(actual, expected, id)
            receipts.append(["state": id, "headline": actual.headline,
                "rows": actual.rows.map { ["kind": $0.kind.rawValue, "strings": $0.strings] as [String: Any] }])
            XCTAssertNil(c.offsiteRecovering); XCTAssertNil(c.lastKnownBackupAt)
            XCTAssertNil(c.lagEpisodeStart); XCTAssertEqual(c.keywordCoverage, source.keywordCoverage)
        }
        if let path = ProcessInfo.processInfo.environment["BRAINBAR_CAPTURE_RECEIPTS"] {
            try JSONSerialization.data(withJSONObject: receipts, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: path))
        }
    }

    func testMappingPreservesMeasuredFieldsAndUnknowns() throws {
        let cases = try BrainBarMenuPresentationTests.loadCases()
        for row in cases {
            let source = try BrainBarMenuCases.capture(row["inputs"] as! [String: Any])
            let c = Self.adapt(source)
            XCTAssertEqual(c.now, source.now); XCTAssertEqual(c.timeZone, source.timeZone)
            XCTAssertEqual(c.verdict?.checks, source.verdict?.checks); XCTAssertEqual(c.driveAuthModel, source.driveAuthModel)
            XCTAssertEqual(c.backupDailyFailures, source.backupDailyFailures)
            XCTAssertEqual(c.backupDailyNextRunAt, source.backupDailyNextRunAt)
            XCTAssertEqual(c.jsonlBackupFailures, source.jsonlBackupFailures)
            XCTAssertEqual(c.maintenanceNightFailures, source.maintenanceNightFailures)
            XCTAssertEqual(c.upload?.startedAt, source.upload?.startedAt)
            XCTAssertEqual(c.upload?.live, source.upload?.live)
            XCTAssertEqual(c.watcher?.alertReasons, source.watcher?.alertReasons)
            XCTAssertEqual(c.watcher?.dbProbeFailed, source.watcher?.dbProbeFailed)
        }
        let now = Date(timeIntervalSince1970: 1_800_000_000.0), zone = TimeZone(secondsFromGMT: 0)!
        let empty = BrainBarMenuLiveCapture.capture(status: .checking, verdictAwakeAgeS: nil,
            wakeCatchUpPending: false, driveAuth: nil, backups: .notRead, backupsNotice: nil, watcher: nil,
            watcherAwakeAgeS: nil, observations: nil, maintenanceNightStreakKnown: false, now: now, timeZone: zone)
        XCTAssertFalse(empty.backupsMeasured); XCTAssertNil(empty.watcher); XCTAssertNil(empty.upload)
        XCTAssertNil(empty.driveAuthModel); XCTAssertNil(empty.backupDailyNextRunAt)
        XCTAssertEqual(empty.verdictAwakeAgeS, .infinity)
        XCTAssertNil(empty.backupDailyFailures); XCTAssertNil(empty.jsonlBackupFailures)
        XCTAssertNil(empty.maintenanceNightFailures)
    }

    func testAnUploadNeedsItsOwnMeasuredProcessFields() {
        let now = Date(timeIntervalSince1970: 1_800_000_000.0)
        let samples: [(Date?, Bool?, String)] = [(nil, true, "in_progress"),
                                                 (now, nil, "in_progress"), (now, true, "pending")]
        for (start, live, state) in samples {
            var backups = BrainBarStatusSnapshotTests.backups()
            backups.localSnapshot = .init(lastAt: now, destination: nil, localVerified: true, uploaded: false,
                uploadState: state, uploadError: nil, uploadProcessLive: live,
                uploadProcessStartedAt: start, localIntegrity: nil, localIntegrityPragma: nil)
            let c = BrainBarMenuLiveCapture.capture(status: .checking, verdictAwakeAgeS: nil,
                wakeCatchUpPending: false, driveAuth: nil, backups: .measured(backups), backupsNotice: nil,
                watcher: .missing(path: "fixture"), watcherAwakeAgeS: nil, observations: nil, maintenanceNightStreakKnown: false,
                now: now, timeZone: TimeZone(secondsFromGMT: 0)!)
            XCTAssertNil(c.upload, "A snapshot date or now must never stand in for a process start")
            XCTAssertNil(c.watcher)
        }
    }

    func testAuthModelIsWordingOnlyAndStaleRedSurvives() throws {
        let rows = try BrainBarMenuPresentationTests.loadCases()
        var source = try BrainBarMenuCases.capture(rows.first { $0["id"] as? String == "S1" }!["inputs"] as! [String: Any])
        source.driveAuthModel = .invalid
        XCTAssertEqual(BrainBarMenuPresenter.present(Self.adapt(source)),
                       BrainBarMenuPresentationTests.expected(rows.first { $0["id"] as? String == "S1" }!))
        source = try BrainBarMenuCases.capture(rows.first { $0["id"] as? String == "S3" }!["inputs"] as! [String: Any])
        let v = source.verdict!
        source.verdict = .init(dot: .red, checks: v.checks, notes: v.notes, evaluatedAt: v.evaluatedAt)
        let stale = BrainBarStatusSnapshot(issues: [], isChecking: false, staleHealthVerdict: source.verdict)
        let c = Self.adapt(source, status: stale, awake: 901.0)
        let actual = BrainBarMenuPresenter.present(c)
        XCTAssertEqual(actual, .init(form: .needs, headline: "2 things need you", count: 2, icon: .marked,
            rows: [.init(kind: .headline, strings: ["2 things need you"], mark: .marked),
                   .init(kind: .separator),
                   .init(kind: .needsYou, strings: ["Something BrainBar checks needs attention.",
                       "BrainBar can’t fix this itself.", "Give the report to an agent."],
                       action: .init(kind: .copyReport, label: "Copy report for an agent"), mark: .alertCircle),
                   .init(kind: .needsYou, strings: ["Something BrainBar checks needs attention.",
                       "BrainBar can’t fix this itself.", "Give the report to an agent."],
                       action: .init(kind: .copyReport, label: "Copy report for an agent"), mark: .alertCircle, dividerBefore: true),
                   .init(kind: .statusLine, strings: ["Checking…"], state: .cantTell),
                   .init(kind: .statusLine, strings: ["Checking…"], state: .cantTell),
                   .init(kind: .statusLine, strings: ["Checking…"], state: .cantTell),
                   .init(kind: .separator),
                   .init(kind: .standardItem, strings: ["Settings…"], keyEquivalent: "⌘,"),
                   .init(kind: .standardItem, strings: ["Quit BrainBar"], keyEquivalent: "⌘Q", optionAlternate: "Restart BrainBar")]))
    }

    static func checking(_ headline: String, fine: String) -> BrainBarMenuPresentation {
        .init(form: .calm, headline: headline, count: 0, icon: .resting, rows: [
            .init(kind: .headline, strings: [headline], mark: .wait), .init(kind: .separator),
            .init(kind: .statusLine, strings: ["Checking…"], mark: .wait, state: .cantTell),
            .init(kind: .everythingElse, strings: [fine], mark: .ok), .init(kind: .separator),
            .init(kind: .standardItem, strings: ["Settings…"], keyEquivalent: "⌘,"),
            .init(kind: .standardItem, strings: ["Quit BrainBar"], keyEquivalent: "⌘Q", optionAlternate: "Restart BrainBar")])
    }

    func testStaleOrFutureBackupsEnvelopeCannotReassure() throws {
        let source = try BrainBarMenuCases.capture(BrainBarMenuPresentationTests.loadCases().first {
            $0["id"] as? String == "S1"
        }!["inputs"] as! [String: Any])
        for notice in ["Backup status is old", "Backup status is from the future"] {
            let c = Self.adapt(source, backupsNotice: notice)
            XCTAssertFalse(c.backupsMeasured)
            XCTAssertEqual(BrainBarMenuPresenter.present(c), Self.checking("Can’t confirm backups yet",
                fine: "Saving, search and upkeep are fine."))
        }
    }

    func testUnknownUpkeepIsNotProvenFine() throws {
        let row = try BrainBarMenuPresentationTests.loadCases().first { $0["id"] as? String == "S1" }!
        var source = try BrainBarMenuCases.capture(row["inputs"] as! [String: Any])
        source.maintenanceNightFailures = 3 // An untrusted prior value must not be treated as a known streak.
        let c = Self.adapt(source, upkeepKnown: false)
        XCTAssertNil(c.maintenanceNightFailures)
        XCTAssertEqual(BrainBarMenuPresenter.present(c), Self.checking("Still checking upkeep",
            fine: "Saving, search and backups are fine."))
    }

    func testAllThreeStreaksStayUnknownAcrossUnavailableHistory() throws {
        let row = try BrainBarMenuPresentationTests.loadCases().first { $0["id"] as? String == "S1" }!
        let source = try BrainBarMenuCases.capture(row["inputs"] as! [String: Any])
        let jobs: [BrainLayerLaunchdJob] = [.backupDaily, .jsonlBackup, .maintenanceNightly]
        for job in jobs {
            for reason in ["read error", "damaged latest record", "history over limit", "missing observation"] {
                let c = Self.adapt(source, editObservations: { original in
                    var observations = original
                    switch reason {
                    case "read error": observations[job] = .stateOnly(.probeError("History unreadable"))
                    case "missing observation": observations.removeValue(forKey: job)
                    case "history over limit":
                        observations[job]?.calendarConsecutiveFailures = nil
                        observations[job]?.calendarNightOutcome?.streakComplete = false
                    default:
                        observations[job]?.calendarConsecutiveFailures = nil
                        observations[job]?.calendarNightOutcome = nil
                    }
                    return observations
                })
                XCTAssertNil(job == .backupDaily ? c.backupDailyFailures
                    : job == .jsonlBackup ? c.jsonlBackupFailures : c.maintenanceNightFailures, "\(job) \(reason)")
                XCTAssertEqual(BrainBarMenuPresenter.present(c), Self.checking(
                    job == .maintenanceNightly ? "Still checking upkeep" : "Can’t confirm backups yet",
                    fine: job == .maintenanceNightly ? "Saving, search and backups are fine." : "Saving, search and upkeep are fine."), "\(job) \(reason)")
            }
        }
        let firstSample = Self.adapt(source, upkeepKnown: false, editObservations: { _ in nil })
        XCTAssertNil(firstSample.backupDailyFailures); XCTAssertNil(firstSample.jsonlBackupFailures)
        XCTAssertNil(firstSample.maintenanceNightFailures)
        var expected = Self.checking("Can’t confirm backups yet", fine: "Saving and search are fine.")
        expected.rows.insert(.init(kind: .statusLine, strings: ["Checking…"], mark: .wait, state: .cantTell), at: 3)
        XCTAssertEqual(BrainBarMenuPresenter.present(firstSample), expected)
    }

    func testSleepStaleMissingAndUnavailableClocks() throws {
        let row = try BrainBarMenuPresentationTests.loadCases().first { $0["id"] as? String == "S1" }!
        var source = try BrainBarMenuCases.capture(row["inputs"] as! [String: Any])
        let expected = BrainBarMenuPresentationTests.expected(row)
        let evaluated = source.verdict!.evaluatedAt
        let history = BrainBarSleepHistory(uptime: { 0.0 })
        history.sleeping(at: evaluated.addingTimeInterval(10.0))
        history.woke(at: evaluated.addingTimeInterval(3610.0))
        source.now = evaluated.addingTimeInterval(3620.0)
        let awake = history.elapsed(since: evaluated, now: source.now)
        XCTAssertEqual(awake, 20.0)
        let afterSleep = Self.adapt(source, awake: awake)
        XCTAssertEqual(afterSleep.verdictAwakeAgeS, 20.0)
        XCTAssertEqual(BrainBarMenuPresenter.present(afterSleep), expected)
        let checking = BrainBarMenuPresenter.present(Self.adapt(source, status: .checking, awake: nil))
        let launch = try BrainBarMenuPresentationTests.loadCases().first { $0["id"] as? String == "S8" }!
        XCTAssertEqual(checking, BrainBarMenuPresentationTests.expected(launch))
        let stale = BrainBarStatusSnapshot(issues: [], isChecking: false, staleHealthVerdict: source.verdict)
        let staleAges: [TimeInterval?] = [901.0, nil, -1.0]
        for age in staleAges {
            let c = Self.adapt(source, status: stale, awake: age)
            XCTAssertEqual(c.verdict, source.verdict)
            XCTAssertEqual(BrainBarMenuPresenter.present(c), checking)
        }
        let watcherAges: [TimeInterval?] = [nil, -1.0, 301.0]
        for age in watcherAges {
            let c = Self.adapt(source, watcherAge: age)
            XCTAssertEqual(c.watcher?.fresh, false)
            XCTAssertEqual(BrainBarMenuPresenter.present(c), Self.checking("Still checking new conversations",
                fine: "Search, backups and upkeep are fine."))
        }
    }
}
