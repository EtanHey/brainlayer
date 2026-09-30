import XCTest
@testable import BrainBar

/// #968: the Backups section's rows (schedule, last run, next run, latest local copy) from injected
/// readers, and Reveal in Finder / Copy path through an injected workspace.
final class BrainBarBackupSourcesTests: XCTestCase {
    private func plist(_ body: String) -> Data {
        Data("""
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0"><dict><key>Label</key><string>x</string>\(body)</dict></plist>
        """.utf8)
    }

    private func calendar(_ zone: String) -> Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: zone)!
        return calendar
    }

    private func date(_ iso: String) -> Date { ISO8601DateFormatter().date(from: iso)! }

    // MARK: the rows the Backups section shows, with injected readers

    func testRowsStateScheduleLastNextAndLocalCopy() {
        let files: [String: Data] = [
            "/LA/com.brainlayer.backup-daily.plist": plist("<key>StartCalendarInterval</key><dict><key>Hour</key><integer>3</integer><key>Minute</key><integer>17</integer></dict>"),
            "/LA/com.brainlayer.maintenance-weekly.plist": plist("<key>StartCalendarInterval</key><dict><key>Weekday</key><integer>0</integer><key>Hour</key><integer>4</integer><key>Minute</key><integer>0</integer></dict>"),
            "/data/logs/backup-daily.log": Data(#"{"attempted_at": "2026-09-29T03:17:06+00:00", "verified": true, "backup_log_provenance": "real"}"#.utf8),
        ]
        let sources = BrainBarBackupSources(
            paths: .init(
                launchAgents: URL(fileURLWithPath: "/LA"),
                databaseLog: URL(fileURLWithPath: "/data/logs/backup-daily.log"),
                archiveLog: URL(fileURLWithPath: "/data/logs/jsonl-backup.log"),
                maintenanceLog: URL(fileURLWithPath: "/data/logs/maintenance.log"),
                snapshotDirectory: URL(fileURLWithPath: "/data/backups"),
                archiveDirectory: URL(fileURLWithPath: "/data/jsonl-backups")
            ),
            readFile: { files[$0.path] },
            listDirectory: { $0.path == "/data/backups" ? ["2026-09-29.db.gz"] : [] },
            isRegularFile: { _ in true }
        )
        let rows = sources.rows(now: date("2026-09-29T10:00:00Z"), calendar: calendar("UTC"), formatDate: { ISO8601DateFormatter().string(from: $0) })

        XCTAssertEqual(rows.map(\.title), ["Database", "Transcripts", "Weekly maintenance"])
        XCTAssertEqual(rows[0].cadence, "daily at 03:17")
        XCTAssertEqual(rows[0].lastRun, "Last run 2026-09-29T03:17:06Z · verified")
        XCTAssertEqual(rows[0].nextRun, "Next run 2026-09-30T03:17:00Z")
        XCTAssertEqual(rows[0].localCopy?.path, "/data/backups/2026-09-29.db.gz")
        XCTAssertEqual(rows[1].cadence, "Schedule unknown — no LaunchAgent installed at /LA/com.brainlayer.jsonl-backup.plist")
        XCTAssertEqual(rows[1].lastRun, "No run recorded in jsonl-backup.log")
        XCTAssertEqual(rows[1].nextRun, "Next run unknown")
        XCTAssertNil(rows[1].localCopy, "no local archive, so Reveal and Copy are hidden")
        XCTAssertEqual(rows[2].cadence, "weekly on Sunday at 04:00")
        XCTAssertEqual(rows[2].nextRun, "Next run 2026-10-04T04:00:00Z")
        XCTAssertNil(rows[2].localCopy, "maintenance has no local copy of its own")
    }

    // MARK: reveal / copy call the injected workspace with the exact path

    @MainActor
    func testRevealAndCopyUseTheExactLocalPath() {
        final class Recorder: BrainBarWorkspaceActing, @unchecked Sendable {
            var revealed: [URL] = []
            var copied: [String] = []
            func reveal(_ url: URL) { revealed.append(url) }
            func copy(_ text: String) { copied.append(text) }
        }
        let recorder = Recorder()
        let url = URL(fileURLWithPath: "/Users/me/.local/share/brainlayer/backups/2026-09-29.db.gz")
        let row = BrainBarBackupScheduleRow(title: "Database", cadence: "", lastRun: "", nextRun: "", localCopy: url)
        row.reveal(using: recorder)
        row.copyPath(using: recorder)
        XCTAssertEqual(recorder.revealed, [url])
        XCTAssertEqual(recorder.copied, ["/Users/me/.local/share/brainlayer/backups/2026-09-29.db.gz"])

        // A row without a local copy never reaches the workspace.
        BrainBarBackupScheduleRow(title: "Weekly maintenance", cadence: "", lastRun: "", nextRun: "", localCopy: nil)
            .reveal(using: recorder)
        XCTAssertEqual(recorder.revealed, [url])
    }

    @MainActor
    func testSettingsViewModelSamplesTheInjectedSourcesAndRoutesActions() async throws {
        final class Recorder: BrainBarWorkspaceActing, @unchecked Sendable {
            var revealed: [URL] = []
            func reveal(_ url: URL) { revealed.append(url) }
            func copy(_ text: String) {}
        }
        let recorder = Recorder()
        let sources = BrainBarBackupSources(
            paths: .init(
                launchAgents: URL(fileURLWithPath: "/LA"),
                databaseLog: URL(fileURLWithPath: "/d/backup-daily.log"),
                archiveLog: URL(fileURLWithPath: "/d/jsonl-backup.log"),
                maintenanceLog: URL(fileURLWithPath: "/d/maintenance.log"),
                snapshotDirectory: URL(fileURLWithPath: "/d/backups"),
                archiveDirectory: URL(fileURLWithPath: "/d/jsonl-backups")
            ),
            readFile: { _ in nil },
            listDirectory: { $0.path == "/d/jsonl-backups" ? ["claude-jsonl-2026-09-29.tar.gz"] : [] },
            isRegularFile: { _ in true }
        )
        let viewModel = BrainBarSettingsViewModel(
            store: BrainLayerConfigStore(configURL: FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).env")),
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            refreshStatusOnLoad: false,
            backupSources: sources,
            workspace: recorder
        )
        viewModel.refreshBackupSchedules()
        for _ in 0..<200 where viewModel.backupSchedules.isEmpty { try await Task.sleep(nanoseconds: 10_000_000) }
        let transcripts = try XCTUnwrap(viewModel.backupSchedules.first { $0.title == "Transcripts" })
        viewModel.revealBackup(transcripts)
        XCTAssertEqual(recorder.revealed.map(\.path), ["/d/jsonl-backups/claude-jsonl-2026-09-29.tar.gz"])
    }

    /// #1015 B1 surfaced: an aborted weekly pass newer than the last completed one is shown as an
    /// attempt beside the real last run, never as the last run itself.
    func testWeeklyRowShowsANewerIncompleteAttemptSeparately() {
        let logText = """
        {"ts": "2026-08-30T02:23:03+00:00", "mode": "full", "dry_run": false, "vacuum_after_bytes": 15926001664}
        {"ts": "2026-09-27T01:40:00+00:00", "mode": "full", "dry_run": false, "backup_status": "unavailable", "vacuum_after_bytes": null}
        """
        let files: [String: Data] = ["/d/maintenance.log": Data(logText.utf8)]
        let sources = BrainBarBackupSources(
            paths: .init(
                launchAgents: URL(fileURLWithPath: "/LA"),
                databaseLog: URL(fileURLWithPath: "/d/backup-daily.log"),
                archiveLog: URL(fileURLWithPath: "/d/jsonl-backup.log"),
                maintenanceLog: URL(fileURLWithPath: "/d/maintenance.log"),
                snapshotDirectory: URL(fileURLWithPath: "/d/backups"),
                archiveDirectory: URL(fileURLWithPath: "/d/jsonl-backups")
            ),
            readFile: { files[$0.path] },
            listDirectory: { _ in [] },
            isRegularFile: { _ in true }
        )
        let weekly = sources.rows(now: date("2026-09-29T10:00:00Z"), calendar: calendar("UTC"), formatDate: { ISO8601DateFormatter().string(from: $0) })[2]
        XCTAssertEqual(
            weekly.lastRun,
            "Last run 2026-08-30T02:23:03Z · last attempt 2026-09-27T01:40:00Z not completed: backup unavailable, VACUUM skipped"
        )
    }

    // MARK: #1016 review round 1

    private func tempDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// B1: each job's paths come from the environment that job runs with: its installed LaunchAgent's
    /// EnvironmentVariables, overlaid by the env file it names (brainlayer-env-run.sh exports the file
    /// last), then BrainBar's own environment, then the job's default.
    func testJobPathsFollowTheJobsOwnEnvironment() throws {
        let home = URL(fileURLWithPath: "/Users/me")
        func agent(_ label: String, _ env: [String: String]) -> Data {
            try! PropertyListSerialization.data(
                fromPropertyList: ["Label": label, "EnvironmentVariables": env], format: .xml, options: 0
            )
        }
        func paths(_ files: [String: Data], _ brainBar: [String: String] = [:]) -> BrainBarBackupSources.Paths {
            .live(databasePath: "/Users/me/.local/share/brainlayer/brainlayer.db", environment: brainBar, home: home,
                  readFile: { files[$0.path] })
        }
        let jsonlAgent = "/Users/me/Library/LaunchAgents/com.brainlayer.jsonl-backup.plist"

        // Neither set: the job's defaults.
        let defaults = paths([:])
        XCTAssertEqual(defaults.archiveLog.path, "/Users/me/.local/share/brainlayer/logs/jsonl-backup.log")
        XCTAssertEqual(defaults.archiveDirectory.path, "/Users/me/.local/share/brainlayer/jsonl-backups")

        // The reviewer's case: both overrides in the environment BrainBar sees.
        let brainBarOnly = paths([:], [
            "BRAINLAYER_JSONL_BACKUP_LOG_PATH": "/bb/jsonl.log", "BRAINLAYER_JSONL_BACKUP_STAGING_DIR": "/bb/archives",
        ])
        XCTAssertEqual(brainBarOnly.archiveLog.path, "/bb/jsonl.log")
        XCTAssertEqual(brainBarOnly.archiveDirectory.path, "/bb/archives")

        // Set only in the job's LaunchAgent: honoured, and ahead of BrainBar's own value.
        let agentOnly = paths([jsonlAgent: agent("com.brainlayer.jsonl-backup", [
            "BRAINLAYER_JSONL_BACKUP_LOG_PATH": "/agent/jsonl.log", "BRAINLAYER_JSONL_BACKUP_STAGING_DIR": "~/agent-archives",
        ])], ["BRAINLAYER_JSONL_BACKUP_LOG_PATH": "/bb/jsonl.log"])
        XCTAssertEqual(agentOnly.archiveLog.path, "/agent/jsonl.log")
        XCTAssertEqual(agentOnly.archiveDirectory.path, "/Users/me/agent-archives")

        // The env file the agent names is exported last, so it wins over the agent's own value.
        let withFile = paths([
            jsonlAgent: agent("com.brainlayer.jsonl-backup", [
                "BRAINLAYER_ENV_FILE": "/cfg/brainlayer.env", "BRAINLAYER_JSONL_BACKUP_STAGING_DIR": "/agent/archives",
            ]),
            "/cfg/brainlayer.env": Data("export BRAINLAYER_JSONL_BACKUP_STAGING_DIR=\"/file/archives\"\n".utf8),
        ])
        XCTAssertEqual(withFile.archiveDirectory.path, "/file/archives")
        XCTAssertEqual(withFile.archiveLog.path, "/Users/me/.local/share/brainlayer/logs/jsonl-backup.log")

        // The other jobs read their own agents the same way.
        let others = paths([
            "/Users/me/Library/LaunchAgents/com.brainlayer.backup-daily.plist": agent(
                "com.brainlayer.backup-daily", ["BRAINLAYER_BACKUP_STAGING_DIR": "/agent/snapshots"]
            ),
            "/Users/me/Library/LaunchAgents/com.brainlayer.maintenance-weekly.plist": agent(
                "com.brainlayer.maintenance-weekly", ["BRAINLAYER_MAINTENANCE_LOG_PATH": "/agent/maintenance.log"]
            ),
        ])
        XCTAssertEqual(others.snapshotDirectory.path, "/agent/snapshots")
        XCTAssertEqual(others.maintenanceLog.path, "/agent/maintenance.log")
    }

    /// B2: Reveal and Copy appear only for the newest candidate that is a REGULAR file. A directory,
    /// a symlink (which could point outside the backups directory) or a vanished name is skipped.
    func testRevealAndCopyOnlyForARegularLatestFile() throws {
        let root = try tempDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshots = root.appendingPathComponent("backups")
        let archives = root.appendingPathComponent("jsonl-backups")
        try FileManager.default.createDirectory(at: snapshots, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: archives, withIntermediateDirectories: true)
        let outside = root.appendingPathComponent("outside.db")
        try Data("x".utf8).write(to: outside)
        try FileManager.default.createDirectory(at: snapshots.appendingPathComponent("2026-09-30.db"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: snapshots.appendingPathComponent("2026-09-29.db"), withDestinationURL: outside)
        try Data("snapshot".utf8).write(to: snapshots.appendingPathComponent("2026-09-28.db.gz"))
        try FileManager.default.createDirectory(at: archives.appendingPathComponent("claude-jsonl-2026-09-30.tar.gz"), withIntermediateDirectories: true)

        let sources = BrainBarBackupSources(
            paths: .init(
                launchAgents: root.appendingPathComponent("LaunchAgents"),
                databaseLog: root.appendingPathComponent("backup-daily.log"),
                archiveLog: root.appendingPathComponent("jsonl-backup.log"),
                maintenanceLog: root.appendingPathComponent("maintenance.log"),
                snapshotDirectory: snapshots,
                archiveDirectory: archives
            ),
            readFile: { FileManager.default.contents(atPath: $0.path) },
            listDirectory: { (try? FileManager.default.contentsOfDirectory(atPath: $0.path)) ?? [] },
            isRegularFile: { BrainBarBackupSources.isRegularFile($0) }
        )
        let rows = sources.rows(now: date("2026-09-30T12:00:00Z"), calendar: calendar("UTC"), formatDate: { _ in "t" })
        XCTAssertEqual(rows[0].localCopy?.lastPathComponent, "2026-09-28.db.gz", "skip the directory and the symlink")
        XCTAssertNil(rows[1].localCopy, "a directory named like an archive is not a local copy")

        XCTAssertFalse(BrainBarBackupSources.isRegularFile(snapshots.appendingPathComponent("2026-09-30.db")))
        XCTAssertFalse(BrainBarBackupSources.isRegularFile(snapshots.appendingPathComponent("2026-09-29.db")))
        XCTAssertFalse(BrainBarBackupSources.isRegularFile(snapshots.appendingPathComponent("2026-09-27.db")), "missing")
        XCTAssertTrue(BrainBarBackupSources.isRegularFile(snapshots.appendingPathComponent("2026-09-28.db.gz")))
    }
}
