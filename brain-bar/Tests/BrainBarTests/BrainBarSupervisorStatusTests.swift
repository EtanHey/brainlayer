import Darwin
import Foundation
import XCTest
@testable import BrainBar

final class BrainBarSupervisorStatusTests: XCTestCase {
    private func journal(_ phase: String) throws -> URL {
        let canonical = try XCTUnwrap(realpath(brainBarTestTemporaryDirectory().path, nil))
        defer { free(canonical) }
        let root = URL(fileURLWithPath: String(cString: canonical))
            .appendingPathComponent("brainbar-owner-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                               attributes: [.posixPermissions: 0o700])
        addTeardownBlock { try FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("migration.json")
        try Data("{\"phase\":\"\(phase)\"}".utf8).write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
        return url
    }

    func testInstallerOwnershipPhasesAndRollback() throws {
        for phase in ["migrating", "migrated", "reverting"] {
            XCTAssertEqual(BrainBarJobOwnership.read(journalURL: try journal(phase)), .supervisor)
        }
        for phase in ["reverted", "rolled_back"] {
            XCTAssertEqual(BrainBarJobOwnership.read(journalURL: try journal(phase)), .legacy)
        }
        let missing = try journal("migrated").deletingLastPathComponent().appendingPathComponent("missing")
        XCTAssertEqual(BrainBarJobOwnership.read(journalURL: missing), .legacy)
    }

    func testRealShapedLargeMigrationJournalKeepsOwnershipAndStillBoundsReads() throws {
        let url = try journal("migrated")
        // The live journal stores rollback file payloads and a serialized manifest, not only phase.
        // Match its key structure with harmless placeholders rather than copying owner contents.
        let payload = String(repeating: "fixture rollback bytes ", count: 1000)
        let row: [String: Any] = ["schema_version": 1, "phase": "migrated", "components": [:],
            "files": ["com.brainlayer.watch.plist": payload, "com.brainlayer.drain.plist": payload],
            "states": [:], "keg": "../Cellar/brainlayer/fixture", "archive": "fixture/archive",
            "group_custody": [:], "published_files": ["fixture-supervisor.plist": payload,
                "com.brainlayer.maintenance.plist": payload], "published_sha256": [:],
            "configuration_sha256": [:], "daemon_pinned": payload, "manifest_bytes": payload,
            "manifest_sha256": String(repeating: "0", count: 64)]
        let data = try JSONSerialization.data(withJSONObject: row)
        XCTAssertGreaterThan(data.count, 117_580)
        try data.write(to: url)
        XCTAssertEqual(BrainBarJobOwnership.read(journalURL: url), .supervisor)
        let oversized = try JSONSerialization.data(withJSONObject: ["phase": "migrated",
            "manifest_bytes": String(repeating: "x", count: 4 * 1024 * 1024)])
        try oversized.write(to: url)
        guard case .unknown = BrainBarJobOwnership.read(journalURL: url) else {
            return XCTFail("Even large rollback journals must have a finite bound")
        }
    }

    func testInterruptedSkippedCalendarRowsKeepPreviousResultsWithoutPoisoningOtherSteps() throws {
        let directory = try journal("migrated").deletingLastPathComponent()
        try Data("""
        {"event":"finished","step":"jsonl-backup","ts":1784466000,"status":"ok","returncode":0}
        {"event":"finished","step":"backup-daily","ts":1784466100,"status":"failed","returncode":1}
        {"event":"finished","step":"jsonl-backup","ts":1784466200,"status":"skipped","returncode":null}
        {"event":"finished","step":"decay","ts":1784466201,"status":"skipped","returncode":null}

        """.utf8).write(to: directory.appendingPathComponent("runs.jsonl"))
        let history = BrainBarCalendarHistory.read(directory: directory)
        XCTAssertNil(history.error)
        XCTAssertEqual(history.runs["jsonl-backup"]?.code, 0)
        XCTAssertEqual(history.runs["jsonl-backup"]?.at, Date(timeIntervalSince1970: 1784466000))
        XCTAssertEqual(history.runs["backup-daily"]?.code, 1)
        XCTAssertNil(history.runs["decay"])
    }

    @MainActor
    func testFailedCalendarStepCannotBeGreenOnJobsOrBackups() throws {
        for (status, code) in [("failed", Int32(1)), ("failed", 0), ("ok", 9)] {
            let history = BrainBarCalendarHistory(runs: Dictionary(uniqueKeysWithValues:
                ["jsonl-backup", "index", "maintenance-nightly"].map {
                    ($0, .init(at: Date(timeIntervalSince1970: 1784466000), code: code, status: status))
                }))
            let observations = Self.supervised(history: history).sampleActivity()
            guard case .failed = observations[.jsonlBackup]?.loadState else {
                XCTFail("A failed result or nonzero exit must not be loaded/green"); continue
            }
            XCTAssertEqual(observations[.jsonlBackup]?.backupScheduleLine?.tone, .red)
            for group in [BrainLayerLaunchdJobGroup.ingest, .maintenance, .backups] {
                XCTAssertEqual(group.status(settings: BrainLayerConfig.defaultConfig.launchdJobs,
                    observations: observations, formatDate: { $0.description }).health, .unhealthy)
            }
        }
    }

    func testCalendar75WithoutVerifiedDeliberateReasonIsAttentionNotFailedOrGreen() {
        let history = BrainBarCalendarHistory(runs: ["jsonl-backup":
            .init(at: Date(), code: 75, status: "deferred")])
        let observations = Self.supervised(history: history).sampleActivity()
        XCTAssertEqual(observations[.jsonlBackup]?.loadState, .loaded)
        XCTAssertEqual(observations[.jsonlBackup]?.backupScheduleLine?.tone, .neutral)
        XCTAssertTrue(observations[.jsonlBackup]?.backupScheduleLine?.text.contains("Deferred") == true)
        XCTAssertEqual(BrainLayerLaunchdJobGroup.backups.status(settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: observations, formatDate: { $0.description }).health.title, "Deferred")
    }

    func testRecordedWindowReasonAndWeeklySubstitutionReachJobs() throws {
        let directory = try journal("migrated").deletingLastPathComponent()
        let file = directory.appendingPathComponent("runs.jsonl")
        for (reason, expected) in [
            ("index-overran-window", "Indexing ran past the 04:00–06:00 window"),
            ("quiet-window-missed", "The 04:00–06:00 maintenance window was missed"),
        ] {
            try Data("""
            {"event":"finished","step":"maintenance-weekly","ts":1784466060,"status":"deferred","returncode":75,"reason":"\(reason)","replaces":"maintenance-nightly"}

            """.utf8).write(to: file)
            let observations = Self.supervised(history: BrainBarCalendarHistory.read(directory: directory)).sampleActivity()
            let status = BrainLayerLaunchdJobGroup.maintenance.status(settings: BrainLayerConfig.defaultConfig.launchdJobs,
                observations: observations, formatDate: { _ in "today" })
            XCTAssertEqual(status.health, .deferred)
            XCTAssertTrue(status.attentionReason?.contains(expected) == true)
            XCTAssertFalse(status.attentionReason?.contains("reason unavailable") == true)
            XCTAssertTrue(status.note?.contains("Nightly skipped: weekly scheduled instead") == true)
        }
        try Data("""
        {"event":"spawn","step":"maintenance-weekly","ts":1784466000,"pid":4242}
        {"event":"finished","step":"maintenance-weekly","ts":1784466060,"status":"ok","returncode":0,"replaces":"maintenance-nightly"}

        """.utf8).write(to: file)
        let observations = Self.supervised(history: BrainBarCalendarHistory.read(directory: directory)).sampleActivity()
        let status = BrainLayerLaunchdJobGroup.maintenance.status(settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: observations, formatDate: { _ in "today" })
        XCTAssertEqual(status.health, .awaitingRun)
        XCTAssertNil(status.attentionReason)
        XCTAssertTrue(status.note?.contains("Nightly skipped: weekly ran instead") == true)
        // A subsequent nightly run must clear the earlier substitution note.
        let nightly = "{\"event\":\"finished\",\"step\":\"maintenance-nightly\",\"ts\":1784552460,\"status\":\"ok\",\"returncode\":0}\n"
        let handle = try FileHandle(forWritingTo: file)
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(nightly.utf8))
        try handle.close()
        let recovered = Self.supervised(history: BrainBarCalendarHistory.read(directory: directory)).sampleActivity()
        let recoveredStatus = BrainLayerLaunchdJobGroup.maintenance.status(settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: recovered, formatDate: { _ in "today" })
        XCTAssertFalse(recoveredStatus.note?.contains("Nightly skipped") == true)
    }

    func testWindowAttributionRequiresRecordedReasonAndNoSpawn() throws {
        let directory = try journal("migrated").deletingLastPathComponent()
        let logs = directory.appendingPathComponent("logs", isDirectory: true)
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        for (reason, spawn) in [("", false), ("invented", false), ("index-overran-window", true)] {
            let prefix = spawn ? "{\"event\":\"spawn\",\"step\":\"maintenance-weekly\",\"ts\":1784466000,\"pid\":4242}\n" : ""
            try Data((prefix + """
            {"event":"finished","step":"maintenance-weekly","ts":1784466060,"status":"deferred","returncode":75,"reason":"\(reason)"}

            """).utf8).write(to: directory.appendingPathComponent("runs.jsonl"))
            let observations = Self.supervised(history: BrainBarCalendarHistory.read(directory: directory), logs: logs).sampleActivity()
            let status = BrainLayerLaunchdJobGroup.maintenance.status(settings: BrainLayerConfig.defaultConfig.launchdJobs,
                observations: observations, formatDate: { _ in "today" })
            XCTAssertTrue(status.attentionReason?.contains("reason unavailable") == true)
            XCTAssertFalse(status.attentionReason?.contains("Indexing ran past") == true)
        }
    }

    func testDisabledNoSpawnCalendar75IsNeutralSkipped() {
        let history = BrainBarCalendarHistory(runs: ["jsonl-backup": .init(at: Date(), code: 75, status: "deferred")])
        let observations = Self.supervised(history: history, disabled: true).sampleActivity()
        XCTAssertEqual(observations[.jsonlBackup]?.backupScheduleLine?.text, "Skipped — disabled")
        XCTAssertEqual(observations[.jsonlBackup]?.backupScheduleLine?.tone, .neutral)
        XCTAssertEqual(BrainLayerLaunchdJobGroup.backups.status(settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: observations, formatDate: { $0.description }).health, .skipped)
    }

    func testCalendarDeliberateDeferralRequiresThisSpawnedRunsNewestAbortRecord() throws {
        let directory = try journal("migrated").deletingLastPathComponent()
        let logs = directory.appendingPathComponent("logs")
        try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
        let start = Date(timeIntervalSince1970: 1784466000)
        let end = start.addingTimeInterval(60)
        try Data("""
        {"event":"spawn","step":"maintenance-nightly","ts":1784466000,"pid":4242}
        {"event":"finished","step":"maintenance-nightly","ts":1784466060,"status":"deferred","returncode":75}
        {"event":"spawn","step":"jsonl-backup","ts":1784466000,"pid":4243}
        {"event":"finished","step":"jsonl-backup","ts":1784466060,"status":"deferred","returncode":75}

        """.utf8).write(to: directory.appendingPathComponent("runs.jsonl"))
        let history = BrainBarCalendarHistory.read(directory: directory)
        XCTAssertEqual(history.runs["maintenance-nightly"]?.startedAt, start)
        let reasons = ["outside quiet window: now=08:00 start_hour=4 duration_minutes=120",
            "recent queue write activity: 2 file(s) modified recently", "queue depth growing: before=1 after=2",
            "unexpected writer holds brainlayer db: pid=42 command=fixture fd=3u", "failed to resume services"]
        for (index, reason) in reasons.enumerated() {
            for name in ["maintenance-nightly", "jsonl-backup"] {
                let log = logs.appendingPathComponent("\(name).out.log")
                try JSONSerialization.data(withJSONObject: ["status": "aborted", "reason": reason]).write(to: log)
                try FileManager.default.setAttributes([.modificationDate: end], ofItemAtPath: log.path)
            }
            let observations = Self.supervised(history: history, logs: logs).sampleActivity()
            let deliberate = index < 4
            XCTAssertEqual(observations[.jsonlBackup]?.backupScheduleLine?.tone, .neutral)
            XCTAssertTrue(observations[.jsonlBackup]?.backupScheduleLine?.text.hasPrefix(deliberate ? "Skipped" : "Deferred") == true)
            for group in [BrainLayerLaunchdJobGroup.maintenance, .backups] {
                XCTAssertEqual(group.status(settings: BrainLayerConfig.defaultConfig.launchdJobs,
                    observations: observations, formatDate: { $0.description }).health, deliberate ? .skipped : .deferred)
            }
        }
        let log = logs.appendingPathComponent("jsonl-backup.out.log")
        try JSONSerialization.data(withJSONObject: ["status": "aborted", "reason": reasons[0]]).write(to: log)
        // An older run's allowlisted reason, or an unfinished later run, is not verification.
        for stamp in [start.addingTimeInterval(-30), end.addingTimeInterval(30)] {
            try FileManager.default.setAttributes([.modificationDate: stamp], ofItemAtPath: log.path)
            XCTAssertEqual(Self.supervised(history: history, logs: logs).sampleActivity()[.jsonlBackup]?.backupScheduleLine?.tone, .neutral)
        }
        try FileManager.default.setAttributes([.modificationDate: end], ofItemAtPath: log.path)
        var noSpawn = history
        noSpawn.runs["jsonl-backup"]?.startedAt = nil
        XCTAssertEqual(Self.supervised(history: noSpawn, logs: logs).sampleActivity()[.jsonlBackup]?.backupScheduleLine?.tone, .neutral)
    }

    func testMalformedLinkedOrPublicJournalCannotClaimAnOwner() throws {
        let malformed = try journal("migrated")
        try Data("not JSON".utf8).write(to: malformed)
        let linked = malformed.deletingLastPathComponent().appendingPathComponent("linked")
        try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: malformed)
        let publicFile = try journal("migrated")
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: publicFile.path)
        for url in [malformed, linked, publicFile, try journal("unknown-phase")] {
            guard case let .unknown(reason) = BrainBarJobOwnership.read(journalURL: url) else {
                XCTFail("Untrusted journal must not become legacy or supervisor: \(url)")
                continue
            }
            XCTAssertTrue(reason.localizedCaseInsensitiveContains("can't determine job owner"))
        }
    }

    func testSupervisedComponentUsesPinnedJSONStatusAndRetainsPID() {
        let provider = BrainLayerLaunchdStatusProvider(
            commandRunner: { command in
                if command.first == "/opt/homebrew/opt/brainlayer/bin/brainlayer" {
                    XCTAssertEqual(Array(command.dropFirst().prefix(2)), ["jobs", "component"])
                    XCTAssertEqual(command.last, "status")
                    return .init(terminationStatus: 0, output: """
                    {"source":"supervisor","runtime":{"configured":true,"paused":false,"pid":4242,
                    "starts":3,"failures":0,"last_exit_code":0,"last_run_at":1784466000,
                    "next_run_at":null,"error":null}}
                    """)
                }
                return .init(terminationStatus: 113, output: "could not find service")
            },
            runRecordDate: { _ in nil }, schedule: { _ in nil },
            ownership: { .supervisor }, calendarHistory: { .empty }
        )
        let watch = provider.sampleActivity()[.watch]
        XCTAssertEqual(watch?.loadState, .running)
        XCTAssertEqual(watch?.pid, 4242)
        XCTAssertEqual(watch?.runs, 3)
        XCTAssertEqual(watch?.ownership, .supervisor)
    }

    func testUnknownOwnerFailsClosedInsteadOfUnloadedOrGreen() {
        let provider = BrainLayerLaunchdStatusProvider(
            commandRunner: { _ in XCTFail("Unknown ownership must not query a legacy label"); return .init(terminationStatus: 113, output: "") },
            ownership: { .unknown("Can't determine job owner: malformed journal") }
        )
        for observation in provider.sampleActivity().values {
            XCTAssertTrue(observation.loadState.title.contains("Can't determine job owner"))
            XCTAssertNotEqual(observation.loadState, .running)
            XCTAssertNotEqual(observation.loadState, .unloaded)
        }
    }

    func testLegacyOwnerKeepsLaunchctlBehavior() {
        let provider = BrainLayerLaunchdStatusProvider(
            commandRunner: { command in
                XCTAssertEqual(command.first, "/bin/launchctl")
                return .init(terminationStatus: 0, output: "pid = 4242\nruns = 2\nlast exit code = 0")
            },
            runRecordDate: { _ in nil }, schedule: { _ in nil }, ownership: { .legacy }
        )
        XCTAssertEqual(provider.sample()[.watch], .running)
    }

    static func supervised(history: BrainBarCalendarHistory = .empty, logs: URL? = nil, disabled: Bool = false) -> BrainLayerLaunchdStatusProvider {
        BrainLayerLaunchdStatusProvider(commandRunner: { command in
            if command.first == "/bin/launchctl" {
                XCTAssertTrue(command.last!.hasSuffix("/com.brainlayer.maintenance"))
                return .init(terminationStatus: 0, output: "runs = 0")
            }
            return .init(terminationStatus: 0, output: """
            {"source":"supervisor","runtime":{"configured":true,"paused":false,"pid":4242,
            "starts":3,"failures":0,"last_exit_code":0,"last_run_at":1784466000}}
            """)
        }, now: { Date(timeIntervalSince1970: 1784466000) }, ownership: { .supervisor },
        calendarHistory: { history }, calendarSchedule: { DateComponents(hour: 3, minute: 0) },
        calendarLogsDirectory: { logs ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/brainlayer") },
        calendarIsDisabled: { _ in disabled })
    }

    func testCalendarBeforeFirstRunUsesParentScheduleAndDoesNotBorrowParentExit() {
        let observations = Self.supervised().sampleActivity()
        for job in [BrainLayerLaunchdJob.backupDaily, .jsonlBackup, .maintenanceNightly, .maintenanceWeekly] {
            let value = observations[job]
            XCTAssertEqual(value?.loadState, .loaded)
            XCTAssertEqual(value?.runs, 0)
            XCTAssertNil(value?.lastExitCode)
            XCTAssertNotNil(value?.nextRunAt)
            XCTAssertTrue(value?.ownerNote?.contains("Scheduled, not yet run under the supervisor") == true)
        }
    }

    func testCalendarReadsOnlyLatestStepResultsAndFailsClosedOnBrokenHistory() throws {
        let url = try journal("migrated")
        let directory = url.deletingLastPathComponent()
        let historyURL = directory.appendingPathComponent("runs.jsonl")
        try Data("""
        {"event":"finished","step":"backup-daily","ts":1784466000,"status":"ok","returncode":0}
        {"event":"finished","step":"backup-daily","ts":1784466100,"status":"failed","returncode":1}
        {"event":"finished","step":"jsonl-backup","ts":1784466200,"status":"deferred","returncode":75}

        """.utf8).write(to: historyURL)
        let history = BrainBarCalendarHistory.read(directory: directory)
        XCTAssertEqual(history.runs["backup-daily"]?.code, 1)
        let observations = Self.supervised(history: history).sampleActivity()
        XCTAssertEqual(observations[.backupDaily]?.lastExitCode, 1)
        XCTAssertEqual(observations[.jsonlBackup]?.lastExitCode, 75)
        try Data("incomplete newest line".utf8).write(to: historyURL)
        XCTAssertNotNil(BrainBarCalendarHistory.read(directory: directory).error)
    }

    func testWatcherUsesSupervisorPIDAndUnknownOwnerCannotBeRevivedByHeartbeat() {
        let probe = LaunchctlWatcherProcessProbe(commandRunner: { command in
            XCTAssertEqual(command, [BrainBarSupervisorRuntime.cli, "jobs", "component", "watch", "status"])
            return .init(terminationStatus: 0, output: """
            {"source":"supervisor","runtime":{"configured":true,"paused":false,"pid":4242,"starts":1,"failures":0}}
            """)
        }, ownership: { .supervisor })
        XCTAssertEqual(probe.sample(), .running(pid: 4242))
        let owner = WatcherLaunchdEvidence(process: .ownerUnknown("Can't determine job owner"))
        let now = Date()
        let health = WatcherHealthStatus.derive(launchd: owner,
            file: .readable(.init(updatedAt: now, pollCount: 50)), now: now)
        guard case .unknown = health else { return XCTFail("Heartbeat must not erase an untrusted owner") }
    }

    func testPausedFailedAndMalformedSupervisorResultsNeverBecomeUnloaded() {
        for (runtime, expected) in [
            ("{\"configured\":true,\"paused\":true,\"pid\":null,\"starts\":0,\"failures\":0}", BrainLayerLaunchdLoadState.paused),
            ("{\"configured\":true,\"paused\":false,\"pid\":null,\"starts\":1,\"failures\":1,\"last_exit_code\":9}", .failed("last exit 9")),
        ] {
            let provider = BrainLayerLaunchdStatusProvider(commandRunner: { _ in
                .init(terminationStatus: 0, output: "{\"source\":\"supervisor\",\"runtime\":\(runtime)}")
            }, ownership: { .supervisor }, calendarHistory: { .empty })
            XCTAssertEqual(provider.sample()[.watch], expected)
        }
        let broken = BrainLayerLaunchdStatusProvider(commandRunner: { _ in .init(terminationStatus: 0, output: "broken") },
            ownership: { .supervisor }, calendarHistory: { .empty })
        guard case .probeError = broken.sample()[.watch] else { return XCTFail("Broken status cannot become green") }
    }

    func testHotlaneProbeUsesItsOwnSupervisorComponentInsteadOfWatcherPID() {
        let probe = LaunchctlWatcherProcessProbe(label: "com.brainlayer.hotlane-brainbar", commandRunner: { command in
            XCTAssertEqual(command, [BrainBarSupervisorRuntime.cli, "jobs", "component", "hotlane", "status"])
            return .init(terminationStatus: 0, output: """
            {"source":"supervisor","runtime":{"configured":true,"paused":false,"pid":5555,"starts":1,"failures":0}}
            """)
        }, ownership: { .supervisor })
        XCTAssertEqual(probe.sample(), .running(pid: 5555))
    }

    func testSameProviderReReadsOwnershipAfterRollback() throws {
        let url = try journal("migrated")
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { command in
            if command.first == BrainBarSupervisorRuntime.cli {
                return .init(terminationStatus: 0, output: """
                {"source":"supervisor","runtime":{"configured":true,"paused":false,"pid":4242,"starts":1,"failures":0}}
                """)
            }
            return .init(terminationStatus: 113, output: "Could not find service")
        }, runRecordDate: { _ in nil }, schedule: { _ in nil }, ownership: { .read(journalURL: url) },
        calendarHistory: { .empty })
        XCTAssertEqual(provider.sample()[.watch], .running)
        try Data("{\"phase\":\"rolled_back\"}".utf8).write(to: url)
        XCTAssertEqual(provider.sample()[.watch], .unloaded)
    }

    func testUnknownOwnerDoesNotProduceAnAllChecksPassBackupBadge() throws {
        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "observability-main-58849a70",
            withExtension: "json", subdirectory: "Fixtures"))
        guard case let .readable(document) = ObservabilityReader.read(url: fixture) else {
            return XCTFail("Fixture must be readable")
        }
        let checks = BrainBarBackupChecks.derive(document.backups, now: Date(),
            jobStatus: .init(text: "Can't determine job owner", tone: .neutral))
        XCTAssertTrue(checks.scheduleUnknown)
        XCTAssertEqual(checks.summary, "Job schedule unknown")
        XCTAssertEqual(checks.checks.first { $0.id == "job" }?.tone, .neutral)
    }

    @MainActor
    func testRealJobsBackupsAndAdvancedRendersShowSupervisorTruth() throws {
        let url = try journal("migrated")
        let root = url.deletingLastPathComponent()
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let observations = Self.supervised().sampleActivity()
        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "observability-main-58849a70",
            withExtension: "json", subdirectory: "Fixtures"))
        let model = BrainBarSettingsViewModel(workspace: BrainBarRecordingWorkspace(), store: store,
            initialLaunchdObservations: observations, refreshStatusOnLoad: false,
            initialObservabilityResult: ObservabilityReader.read(url: fixture),
            initialWatcherHealth: .readable(.init(updatedAt: Date(), pollCount: 50)))
        XCTAssertTrue(model.launchdStatusTitle(.watch).contains("PID 4242"))
        XCTAssertFalse(model.backupChecks!.details.contains { $0.value.localizedCaseInsensitiveContains("not loaded") })
        XCTAssertTrue(model.presentedBackupSchedules.allSatisfy { $0.cadence.contains("com.brainlayer.maintenance") })
        for section in [BrainBarSettingsSection.jobs, .backups, .advanced] {
            let output = ProcessInfo.processInfo.environment["BRAINBAR_SUPERVISOR_RENDER_DIR"].map { URL(fileURLWithPath: $0) } ?? root
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            let text = try BrainBarRenderHarness.renderSettingsText(model, section: section, directory: output)
            try text.write(to: output.appendingPathComponent("supervised-\(section.rawValue).txt"), atomically: true, encoding: .utf8)
            XCTAssertFalse(text.localizedCaseInsensitiveContains("Unloaded"), text)
            XCTAssertFalse(text.localizedCaseInsensitiveContains("not loaded"), text)
            XCTAssertFalse(text.localizedCaseInsensitiveContains("no LaunchAgents"), text)
            if section == .backups { XCTAssertTrue(text.localizedCaseInsensitiveContains("maintenance"), text) }
            if section == .advanced { XCTAssertTrue(text.contains("4242"), text) }
        }
    }
}
