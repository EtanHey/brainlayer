import XCTest
@testable import BrainBar

/// Etan row E1 (2026-10-01): the Backups page's bottom status list read "very ad-hoc-ish". The
/// page now shows human checks with plain states and relative times; Drive IDs, file names and
/// launchd labels move behind a Technical details disclosure, each with Copy.
final class BrainBarBackupChecksTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_790_769_600) // 2026-09-30T12:00:00Z
    private let locale = Locale(identifier: "en_US")
    private let archiveID = "1-QM42xZpLr8vDriveFileId"
    private let launchdLabel = "com.brainlayer.jsonl-backup"

    private func backups(
        freshness: String? = "fresh",
        retention: String? = "PASS",
        archives: Int? = 3,
        errorType: String? = nil,
        uploadVerified: Bool? = true,
        uploadAgeHours: Double = 1,
        snapshotVerified: Bool? = true,
        bootstrapped: Bool = true,
        parked: Bool = false
    ) -> ObservabilityDocument.Backups {
        ObservabilityDocument.Backups(
            state: "measured", reason: "", inputs: [],
            freshness: freshness, thresholdHours: 36, retentionInvariant: retention,
            survivingArchives30D: archives, errorType: errorType,
            lastVerifiedUpload: uploadVerified.map { verified in
                .init(at: now.addingTimeInterval(-uploadAgeHours * 3_600), ageHours: uploadAgeHours, archiveId: archiveID, verified: verified)
            },
            dbSnapshot: snapshotVerified.map { verified in
                .init(lastAt: now.addingTimeInterval(-2 * 3_600), destination: "2026-09-30.db.gz", verified: verified)
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
        XCTAssertEqual(details.first(where: { $0.label == "Drive file ID — latest transcript archive" })?.value, archiveID)
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

    func test_stale_verified_transcript_upload_keeps_receipt_status_without_claiming_a_failed_attempt() {
        for (code, age) in [("jsonl_backup_attempt_stale", 40.0), ("jsonl_backup_attempt_absent_multiple_nights", 72.0)] {
            let state = backups(freshness: "stale", errorType: code, uploadAgeHours: age)
            let lines = ObservabilityPresentation.backupStatus(for: state, locale: locale, now: now)
            let result = checks(state)
            XCTAssertEqual(state.lastVerifiedUpload?.verified, true)
            XCTAssertEqual(result.alert, lines.error?.text, code)
            XCTAssertFalse(result.alert?.contains("reported an error") == true, code)
            XCTAssertEqual(result.alertTone, .red, code)
            XCTAssertEqual(result.checks.first { $0.id == "freshness" }?.tone, .red, code)
            XCTAssertEqual(result.checks.first { $0.id == "freshness" }?.value, "A copy is older than 36 hours", code)
            XCTAssertNotNil(result.attentionSentence, code)
            XCTAssertFalse(result.summary.contains("checks pass"), code)
            XCTAssertEqual(result.details.first { $0.label == "Error code" }?.value, code)
        }
    }

    func test_missing_and_invalid_transcript_receipts_keep_their_status_without_claiming_a_failed_attempt() {
        for code in ["jsonl_backup_attempt_missing", "jsonl_backup_attempt_invalid"] {
            let state = backups(freshness: nil, errorType: code, uploadVerified: nil)
            let lines = ObservabilityPresentation.backupStatus(for: state, locale: locale, now: now)
            let result = checks(state)
            XCTAssertEqual(result.alert, lines.error?.text, code)
            XCTAssertFalse(result.alert?.contains("reported an error") == true, code)
            XCTAssertEqual(result.alertTone, .red, code)
            XCTAssertEqual(result.checks.first { $0.id == "upload" }?.value, "No verified upload yet", code)
            XCTAssertEqual(result.checks.first { $0.id == "freshness" }?.tone, .red, code)
            XCTAssertNotNil(result.attentionSentence, code)
            XCTAssertFalse(result.summary.contains("checks pass"), code)
        }
    }

    func test_failed_transcript_attempt_still_reports_an_error() {
        let result = checks(backups(freshness: "stale", errorType: "jsonl_backup_attempt_failed"))
        XCTAssertEqual(result.alert, "The last transcript backup reported an error.")
        XCTAssertEqual(result.alertTone, .red)
        XCTAssertNotNil(result.attentionSentence)
    }

    func test_fresh_verified_upload_and_no_op_keep_their_existing_verdicts() {
        let uploaded = checks(backups())
        XCTAssertEqual(uploaded.summary, "All 6 checks pass")
        XCTAssertEqual(uploaded.checks.first { $0.id == "upload" }?.value, "Verified 1 hour ago")
        XCTAssertNil(uploaded.alert)
        XCTAssertNil(uploaded.attentionSentence)

        var idle = backups(archives: 0, uploadVerified: nil)
        idle.transcriptAttempt = .init(at: now.addingTimeInterval(-3_600), status: "no-op", verified: true, uploaded: false)
        let result = checks(idle)
        XCTAssertEqual(result.summary, "Idle — no new transcripts")
        XCTAssertEqual(result.checks.first { $0.id == "upload" }?.tone, .neutral)
        XCTAssertTrue(result.checks.first { $0.id == "upload" }?.value.hasPrefix("No new transcripts") == true)
        XCTAssertEqual(result.checks.first { $0.id == "archives" }?.tone, .neutral)
        XCTAssertNil(result.alert)
        XCTAssertNil(result.attentionSentence)
    }

    func test_future_transcript_receipt_check_says_invalid_timestamp_and_stays_red() throws {
        let iso = ISO8601DateFormatter()
        let referenceNow = iso.date(from: "2026-09-14T12:00:00Z")!
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let dates = [referenceNow.addingTimeInterval(1), referenceNow.addingTimeInterval(86_400),
                     iso.date(from: "2027-01-01T12:00:00Z")!]
        for date in dates {
            for verified in [true, false] {
                for noOp in [false, true] {
                    var state = ObservabilityDocument.Backups(
                        state: "measured", reason: "", inputs: [], freshness: "fresh", thresholdHours: 36,
                        retentionInvariant: "PASS", survivingArchives30D: 6, errorType: nil,
                        lastVerifiedUpload: .init(at: date, ageHours: 0, archiveId: "fixture-archive", verified: verified),
                        dbSnapshot: .init(lastAt: referenceNow.addingTimeInterval(-7200), destination: "fixture.db.zst", verified: true),
                        launchd: .init(label: launchdLabel, bootstrapped: true, disabledDirPresent: false))
                    if noOp {
                        state.transcriptAttempt = .init(at: referenceNow.addingTimeInterval(-60), status: "no-op", verified: true, uploaded: false)
                    }
                    let result = BrainBarBackupChecks.derive(state, now: referenceNow, locale: locale, calendar: calendar)
                    let row = try XCTUnwrap(result.checks.first { $0.id == "upload" })
                    XCTAssertEqual(row.value, "Invalid timestamp")
                    XCTAssertEqual(row.tone, .red)
                    XCTAssertEqual(result.verdictTone, .red)
                    XCTAssertEqual(result.attentionSentence, "Transcripts in Google Drive: Invalid timestamp")
                }
            }
        }
    }

    func test_receipt_at_now_remains_verified_with_generic_relative_clamping_unchanged() {
        let result = checks(backups(uploadAgeHours: 0))
        XCTAssertEqual(result.checks.first { $0.id == "upload" }?.value, "Verified just now")
        XCTAssertEqual(result.checks.first { $0.id == "upload" }?.tone, .green)
        XCTAssertEqual(result.summary, "All 6 checks pass")
        XCTAssertEqual(BrainBarBackupChecks.relative(now.addingTimeInterval(1), now: now, locale: locale), "just now")
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
            let lines = ObservabilityPresentation.backupStatus(for: state, locale: locale, now: now)
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
        observability: ObservabilityReadResult, workspace: Recorder = Recorder(),
        jsonl: BrainLayerLaunchdJobObservation? = nil
    ) throws -> (BrainBarSettingsViewModel, URL) {
        let now: Date
        switch observability {
        case .readable(let document): now = document.generatedAt
        case .unreadable: now = BrainBarDashboardFixture.fetchedAt
        }
        let root = URL(fileURLWithPath: brainBarTestTemporaryPath(), isDirectory: true)
            .appendingPathComponent("brainbar-plain-paths-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let configURL = root.appendingPathComponent("brainlayer.env")
        let store = BrainLayerConfigStore(configURL: configURL)
        try store.save(.defaultConfig)
        let viewModel = BrainBarSettingsViewModel(workspace: workspace, store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            initialLaunchdObservations: [
                .backupDaily: .init(loadState: .loaded, runs: 1, lastExitCode: 0, lastRunAt: now, nextRunAt: now, isContinuous: false),
                .jsonlBackup: jsonl ?? .init(loadState: .loaded, runs: 1, lastExitCode: 0, lastRunAt: now, nextRunAt: now, isContinuous: false),
            ],
            refreshStatusOnLoad: false,
            now: { now },
            initialObservabilityResult: observability)
        return (viewModel, configURL)
    }

    func test_the_backups_badge_reason_never_names_the_launchd_label() throws {
        guard case let .readable(healthy) = BrainBarDashboardFixture.healthyObservabilityResult else {
            XCTFail("fixture must be readable")
            return
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
        // S5 dot policy: a deliberate safety stop is visible text, not a dot; the copies' own
        // 36 h threshold is what turns a long pause into one.
        XCTAssertEqual(health.badge, .unknown)
        XCTAssertEqual(health.reason, "Daily transcript backup: Paused by a safety stop")
        XCTAssertEqual(viewModel.backupChecks?.attentionSentence, health.reason)
    }

    /// Lead UX r1 on #1064: each failure is shown once. The Backups header keeps its badge but drops
    /// any reason the page already shows (a recovery check, the Drive card, the job-alert card); a
    /// reason shown nowhere else (the launchd job's exit) still appears under the badge.
    func test_the_backups_header_never_repeats_a_reason_the_page_already_shows() throws {
        guard case let .readable(healthy) = BrainBarDashboardFixture.healthyObservabilityResult else {
            XCTFail("fixture must be readable")
            return
        }
        let unverified = healthy.replacingBackupsLastVerifiedUpload(verified: false)
        let (model, _) = try makeViewModel(observability: .readable(unverified))
        XCTAssertEqual(model.backupsHealth(drive: nil).badge, .attention)
        XCTAssertNil(model.backupsBadgeReason(drive: nil), "the Recovery checks card already says it")

        let drive = DriveAuthPresentation.derive(
            status: .init(state: .missing, reason: nil, expiresAt: nil),
            isReconnecting: false, lastOutcome: nil, now: Date(), formatDate: { _ in "" }
        )
        let (healthyModel, _) = try makeViewModel(observability: .readable(healthy))
        XCTAssertEqual(healthyModel.backupsHealth(drive: drive).reason, drive.line)
        XCTAssertNil(healthyModel.backupsBadgeReason(drive: drive), "the Google Drive card already says it")

        let failedJob = BrainLayerLaunchdJobObservation(loadState: .loaded, runs: 2, lastExitCode: 1, lastRunAt: Date(),
                                                        nextRunAt: Date(), isContinuous: false)
        let (jobModel, _) = try makeViewModel(observability: .readable(healthy), jsonl: failedJob)
        let reason = try XCTUnwrap(jobModel.backupsHealth(drive: nil).reason)
        XCTAssertEqual(jobModel.backupsBadgeReason(drive: nil), reason, "shown nowhere else, so it stays")
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

private extension ObservabilityDocument {
    func replacingBackupsLastVerifiedUpload(verified: Bool) -> Self {
        let b = backups
        return Self(
            schemaVersion: schemaVersion, generatedAt: generatedAt, dbPath: dbPath, windowHours: windowHours,
            stores: stores, emitters: emitters, authorUnknown: authorUnknown,
            backups: .init(
                state: b.state, reason: b.reason, inputs: b.inputs, freshness: b.freshness,
                thresholdHours: b.thresholdHours, retentionInvariant: b.retentionInvariant,
                survivingArchives30D: b.survivingArchives30D, errorType: b.errorType,
                lastVerifiedUpload: b.lastVerifiedUpload.map { .init(at: $0.at, ageHours: $0.ageHours, archiveId: $0.archiveId, verified: verified) },
                dbSnapshot: b.dbSnapshot, launchd: b.launchd
            )
        )
    }
}
