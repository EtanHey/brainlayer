import XCTest
@testable import BrainBar

/// Etan row E1 (2026-10-01): the Backups page's bottom status list read "very ad-hoc-ish". The
/// page now shows human checks with plain states and relative times; Drive IDs, file names and
/// launchd labels move behind a Technical details disclosure, each with Copy.
final class BrainBarBackupChecksTests: XCTestCase {
    private let now = ISO8601DateFormatter().date(from: "2026-09-30T12:00:00Z")!
    private let locale = Locale(identifier: "en_US")
    private let archiveID = "1-QM42xZpLr8vDriveFileId"
    private let launchdLabel = "com.brainlayer.jsonl-backup"

    private func backups(
        freshness: String? = "fresh",
        retention: String? = "PASS",
        archives: Int? = 3,
        errorType: String? = nil,
        uploadVerified: Bool? = true,
        snapshotVerified: Bool? = true,
        bootstrapped: Bool = true,
        parked: Bool = false
    ) -> ObservabilityDocument.Backups {
        ObservabilityDocument.Backups(
            state: "measured", reason: "", inputs: [],
            freshness: freshness, thresholdHours: 36, retentionInvariant: retention,
            survivingArchives30D: archives, errorType: errorType,
            lastVerifiedUpload: uploadVerified.map {
                .init(at: now.addingTimeInterval(-3_600), ageHours: 1, archiveId: archiveID, verified: $0)
            },
            dbSnapshot: snapshotVerified.map {
                .init(lastAt: now.addingTimeInterval(-2 * 3_600), destination: "2026-09-30.db.gz", verified: $0)
            },
            launchd: .init(label: launchdLabel, bootstrapped: bootstrapped, disabledDirPresent: parked)
        )
    }

    private func checks(_ backups: ObservabilityDocument.Backups) -> BrainBarBackupChecks {
        BrainBarBackupChecks.derive(backups, now: now, locale: locale)
    }

    /// Everything the default view shows, as one string.
    private func defaultViewText(_ checks: BrainBarBackupChecks) -> String {
        ([checks.summary, checks.alert ?? ""] + checks.checks.flatMap { [$0.title, $0.value] })
            .joined(separator: "\n")
    }

    func test_healthy_backups_read_as_human_checks_with_relative_times() {
        let checks = checks(backups())
        XCTAssertEqual(checks.summary, "All 6 checks pass")
        XCTAssertNil(checks.alert)
        XCTAssertNil(checks.attentionSentence)
        XCTAssertEqual(checks.checks.map(\.title), [
            "Transcripts in Google Drive", "Database copy", "Daily transcript backup",
            "Up to date", "Safe cleanup", "Verified archives, last 30 days",
        ])
        XCTAssertEqual(checks.checks.map(\.value), [
            "Verified 1 hour ago", "Verified 2 hours ago", "Scheduled",
            "Both copies are under 36 hours old", "On", "3",
        ])
        XCTAssertEqual(Set(checks.checks.map(\.tone)), [.green])
    }

    func test_the_default_view_never_shows_a_drive_id_file_name_or_launchd_label() {
        let states = [
            backups(), backups(uploadVerified: false, snapshotVerified: false),
            backups(bootstrapped: false), backups(bootstrapped: false, parked: true),
            backups(freshness: "stale", retention: "FAIL", archives: 0, errorType: "jsonl_backup_attempt_failed"),
        ]
        for state in states {
            let text = defaultViewText(checks(state))
            for raw in [archiveID, "2026-09-30.db.gz", launchdLabel, ".disabled-retention-P0", "invariant"] {
                XCTAssertFalse(text.contains(raw), "default view leaked \(raw):\n\(text)")
            }
        }
    }

    func test_the_raw_identifiers_live_in_technical_details_each_copyable() {
        let details = checks(backups()).details
        XCTAssertEqual(details.first(where: { $0.label == "Transcript archive ID" })?.value, archiveID)
        XCTAssertEqual(details.first(where: { $0.label == "Database copy file" })?.value, "2026-09-30.db.gz")
        XCTAssertEqual(details.first(where: { $0.label == "launchd job" })?.value, "\(launchdLabel) · loaded")
        XCTAssertEqual(details.first(where: { $0.label == "Retention invariant" })?.value, "PASS")
        XCTAssertEqual(Set(details.map(\.id)).count, details.count, "detail ids must be unique")
    }

    func test_a_parked_job_says_so_in_plain_words_and_leads_the_attention_sentence() {
        let checks = checks(backups(bootstrapped: false, parked: true))
        let job = checks.checks.first { $0.id == "job" }
        XCTAssertEqual(job?.value, "Paused by a safety stop")
        XCTAssertEqual(job?.tone, .red)
        XCTAssertEqual(checks.summary, "1 of 6 checks needs attention")
        XCTAssertEqual(checks.attentionSentence, "Daily transcript backup: Paused by a safety stop")
        XCTAssertEqual(
            checks.details.first(where: { $0.label == "launchd job" })?.value,
            "\(launchdLabel) · not loaded · parked in .disabled-retention-P0"
        )
    }

    func test_unverified_and_missing_copies_read_plainly() {
        let unverified = checks(backups(uploadVerified: false, snapshotVerified: false))
        XCTAssertEqual(unverified.checks[0].value, "Uploaded 1 hour ago, not verified")
        XCTAssertEqual(unverified.checks[1].value, "Saved 2 hours ago, not verified")
        let missing = checks(backups(uploadVerified: nil, snapshotVerified: nil, bootstrapped: false))
        XCTAssertEqual(missing.checks[0].value, "No verified upload yet")
        XCTAssertEqual(missing.checks[1].value, "No verified copy yet")
        XCTAssertEqual(missing.checks[2].value, "Not scheduled")
        XCTAssertEqual(missing.summary, "3 of 6 checks need attention")
    }

    func test_stale_failed_and_unknown_states_read_plainly() {
        let bad = checks(backups(freshness: "stale", retention: "FAIL", archives: 0))
        XCTAssertEqual(bad.checks[3].value, "A copy is older than 36 hours")
        XCTAssertEqual(bad.checks[4].value, "Failing")
        XCTAssertEqual(bad.checks[5].value, "None")
        let unknown = checks(backups(freshness: nil, retention: nil, archives: nil))
        XCTAssertEqual(unknown.checks[3].value, "Unknown")
        XCTAssertEqual(unknown.checks[4].value, "Unknown")
        XCTAssertEqual(unknown.checks[5].value, "Unknown")
    }

    /// #1029 B1 still holds: each check's tone is the tone of the status line it replaces, so
    /// the badge can never be green beside a red check, or red beside an all-green list.
    func test_every_check_tone_matches_the_status_line_it_replaces() {
        let states = [
            backups(), backups(uploadVerified: false), backups(uploadVerified: nil, snapshotVerified: false),
            backups(bootstrapped: false, parked: true), backups(freshness: "stale", retention: "FAIL", archives: 0),
            backups(freshness: nil, retention: nil, archives: nil),
        ]
        for state in states {
            let lines = ObservabilityPresentation.backupStatus(for: state, locale: locale)
            let checks = checks(state)
            XCTAssertEqual(
                checks.checks.map(\.tone),
                [lines.upload, lines.snapshot, lines.job, lines.freshness, lines.retention, lines.archives].map(\.tone)
            )
        }
    }

    func test_a_job_alert_is_the_alert_and_leads_the_attention_sentence() {
        let alert = "Transcript backup failed; check the backup log"
        let checks = checks(backups(errorType: "job_alert:\(alert)", uploadVerified: nil))
        XCTAssertEqual(checks.alert, alert)
        XCTAssertEqual(checks.alertTone, .red)
        XCTAssertEqual(checks.attentionSentence, alert)
    }

    func test_a_restored_drive_credential_is_a_neutral_note_not_an_attention() {
        let checks = checks(backups(errorType: "drive_credentials_restored_backup_pending"))
        XCTAssertEqual(checks.alertTone, .neutral)
        XCTAssertNil(checks.attentionSentence)
        XCTAssertEqual(checks.summary, "All 6 checks pass")
    }

    func test_relative_time_is_plain_and_never_in_the_future() {
        XCTAssertEqual(BrainBarBackupChecks.relative(now.addingTimeInterval(-20), now: now, locale: locale), "just now")
        XCTAssertEqual(BrainBarBackupChecks.relative(now.addingTimeInterval(600), now: now, locale: locale), "just now")
        XCTAssertEqual(BrainBarBackupChecks.relative(now.addingTimeInterval(-3 * 86_400), now: now, locale: locale), "3 days ago")
        XCTAssertEqual(BrainBarBackupChecks.relative(now.addingTimeInterval(-25 * 60), now: now, locale: locale), "25 minutes ago")
    }
}

/// E1 + E5 through the settings view model: the badge reason speaks the checks' words, and the one
/// Config file row copies and reveals the file the model actually reads.
@MainActor
final class BrainBarSettingsPlainPathsTests: XCTestCase {
    private final class Recorder: BrainBarWorkspaceActing, @unchecked Sendable {
        var revealed: [URL] = []
        var copied: [String] = []
        func reveal(_ url: URL) { revealed.append(url) }
        func copy(_ text: String) { copied.append(text) }
    }

    private func makeViewModel(
        observability: ObservabilityReadResult, workspace: Recorder = Recorder()
    ) throws -> (BrainBarSettingsViewModel, URL) {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-plain-paths-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let configURL = root.appendingPathComponent("brainlayer.env")
        let store = BrainLayerConfigStore(configURL: configURL)
        try store.save(.defaultConfig)
        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            initialLaunchdObservations: [
                .backupDaily: .init(loadState: .loaded, runs: 1, lastExitCode: 0, lastRunAt: Date(), nextRunAt: Date(), isContinuous: false),
                .jsonlBackup: .init(loadState: .loaded, runs: 1, lastExitCode: 0, lastRunAt: Date(), nextRunAt: Date(), isContinuous: false),
            ],
            refreshStatusOnLoad: false,
            initialObservabilityResult: observability,
            workspace: workspace
        )
        return (viewModel, configURL)
    }

    func test_the_backups_badge_reason_never_names_the_launchd_label() throws {
        guard case let .readable(healthy) = BrainBarDashboardFixture.healthyObservabilityResult else {
            return XCTFail("fixture must be readable")
        }
        let parked = ObservabilityDocument.Backups(
            state: "measured", reason: "", inputs: [], freshness: "fresh", thresholdHours: 36,
            retentionInvariant: "PASS", survivingArchives30D: 3, errorType: nil,
            lastVerifiedUpload: healthy.backups.lastVerifiedUpload, dbSnapshot: healthy.backups.dbSnapshot,
            launchd: .init(label: "com.brainlayer.jsonl-backup", bootstrapped: false, disabledDirPresent: true)
        )
        let document = ObservabilityDocument(
            schemaVersion: healthy.schemaVersion, generatedAt: healthy.generatedAt, dbPath: healthy.dbPath,
            windowHours: healthy.windowHours, stores: healthy.stores, emitters: healthy.emitters,
            authorUnknown: healthy.authorUnknown, backups: parked
        )
        let (viewModel, _) = try makeViewModel(observability: .readable(document))
        let health = viewModel.backupsHealth(drive: nil)
        XCTAssertEqual(health.badge, .attention)
        XCTAssertEqual(health.reason, "Daily transcript backup: Paused by a safety stop")
        XCTAssertEqual(viewModel.backupChecks?.attentionSentence, health.reason)
    }

    func test_the_config_file_row_copies_and_reveals_the_file_the_model_reads() throws {
        let workspace = Recorder()
        let (viewModel, configURL) = try makeViewModel(observability: .unreadable("x"), workspace: workspace)
        XCTAssertEqual(viewModel.configFileURL, configURL)
        viewModel.copyConfigFilePath()
        viewModel.revealConfigFile()
        XCTAssertEqual(workspace.copied, [configURL.path])
        XCTAssertEqual(workspace.revealed, [configURL])
        viewModel.copyText("1-QM42")
        XCTAssertEqual(workspace.copied.last, "1-QM42")
    }
}
