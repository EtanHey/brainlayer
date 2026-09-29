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
            listDirectory: { $0.path == "/data/backups" ? ["2026-09-29.db.gz"] : [] }
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
            listDirectory: { $0.path == "/d/jsonl-backups" ? ["claude-jsonl-2026-09-29.tar.gz"] : [] }
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
}
