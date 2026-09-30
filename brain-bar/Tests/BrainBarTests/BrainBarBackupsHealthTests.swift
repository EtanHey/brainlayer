import XCTest
@testable import BrainBar

/// #1029 review round 1. B1: the Backups page never says "Backups can't upload" (red) beside
/// "Healthy" (green): one derivation feeds the group badge, the Google Drive card and the status
/// lines. B2: a job alert, the failed job's own sentence, leads the attention summary.
final class BrainBarBackupsHealthTests: XCTestCase {
    private let now = ISO8601DateFormatter().date(from: "2026-09-30T12:00:00Z")!
    private let jobAlert = "Transcript backup failed; check the backup log"

    private func drive(_ state: DriveAuthStatus.State) -> DriveAuthPresentation {
        let expiresAt: Date? = switch state {
        case .valid: now.addingTimeInterval(5 * 86_400)
        case .expiring: now.addingTimeInterval(5 * 3_600)
        default: nil
        }
        return DriveAuthPresentation.derive(
            status: .init(state: state, reason: state == .unknown ? "brainlayer CLI not found" : nil, expiresAt: expiresAt),
            isReconnecting: false, lastOutcome: nil, now: now, formatDate: { ISO8601DateFormatter().string(from: $0) }
        )
    }

    private func job(_ health: BrainLayerLaunchdGroupHealth) -> BrainLayerLaunchdGroupStatus {
        let reason: String? = switch health {
        case .unhealthy: "Transcript backup last run exited 1."
        case .unknown: "Transcript backup status is unavailable."
        default: nil
        }
        return .init(health: health, attentionReason: reason, lastRunText: "", nextRunText: "")
    }

    private var greenStatus: ObservabilityBackupStatus {
        let green = ObservabilityStatusLine(text: "ok", tone: .green)
        return .init(upload: green, snapshot: green, job: green, freshness: green, retention: green, archives: green, error: nil)
    }

    private var redStatus: ObservabilityBackupStatus {
        let green = ObservabilityStatusLine(text: "ok", tone: .green)
        return .init(
            upload: green, snapshot: .init(text: "DB snapshot verification failed", tone: .red),
            job: green, freshness: green, retention: green, archives: green, error: nil
        )
    }

    /// A job alert beside another red line (no verified upload on record).
    private var jobAlertStatus: ObservabilityBackupStatus {
        ObservabilityPresentation.backupStatus(for: ObservabilityDocument.Backups(
            state: "measured", reason: "", inputs: [], freshness: "stale",
            thresholdHours: nil, retentionInvariant: nil, survivingArchives30D: nil,
            errorType: "job_alert:\(jobAlert)", lastVerifiedUpload: nil, dbSnapshot: nil, launchd: nil
        ), locale: Locale(identifier: "en_US"))
    }

    func test_the_badge_agrees_with_every_line_for_every_drive_and_backup_state() {
        let drives: [DriveAuthPresentation?] = [nil] + [DriveAuthStatus.State.valid, .expiring, .missing, .invalid, .unknown].map(drive)
        let jobs: [BrainLayerLaunchdGroupHealth] = [.healthy, .awaitingRun, .unhealthy, .unknown]
        let statuses: [ObservabilityBackupStatus?] = [nil, greenStatus, redStatus, jobAlertStatus]
        for drive in drives {
            for health in jobs {
                for status in statuses {
                    let health = self.job(health)
                    let verdict = BrainBarBackupsHealth.derive(
                        job: health, drive: drive, status: status, statusUnavailableReason: "Backup status is unmeasurable."
                    )
                    let context = "drive=\(drive?.tone as Any) job=\(health.health) status=\(status == nil ? "unmeasurable" : status!.attentionLine?.text ?? "green")"
                    // Every red line the page shows.
                    let redLines = [drive?.tone == .attention ? drive?.line : nil,
                                    health.health == .unhealthy ? health.attentionReason : nil]
                        .compactMap { $0 }
                        + (status.map { $0.lines.filter { $0.tone == .red }.map(\.text) } ?? ["Backup status is unmeasurable."])
                    XCTAssertEqual(verdict.badge == .attention, !redLines.isEmpty, context)
                    if verdict.badge == .attention {
                        XCTAssertTrue(redLines.contains(verdict.reason ?? ""), "the badge names a red line: \(context)")
                    }
                    XCTAssertEqual(verdict.badge == .expiring, redLines.isEmpty && drive?.tone == .expiring, context)
                    if verdict.badge == .expiring { XCTAssertEqual(verdict.reason, drive?.line, context) }
                    let allClear = redLines.isEmpty && (drive == nil || drive?.tone == .connected)
                    XCTAssertEqual(verdict.badge == .healthy, allClear && health.health == .healthy, context)
                    XCTAssertEqual(verdict.badge == .awaitingRun, allClear && health.health == .awaitingRun, context)
                    if verdict.badge == .healthy { XCTAssertNil(verdict.reason, context) }
                }
            }
        }
    }

    func test_missing_drive_beside_a_healthy_job_is_attention_with_the_drive_line() {
        for state in [DriveAuthStatus.State.missing, .invalid] {
            let verdict = BrainBarBackupsHealth.derive(
                job: job(.healthy), drive: drive(state), status: greenStatus, statusUnavailableReason: "x"
            )
            XCTAssertEqual(verdict, .init(badge: .attention, reason: drive(state).line))
        }
    }

    func test_a_job_alert_leads_the_backups_page_and_the_hero() {
        XCTAssertEqual(jobAlertStatus.lines.first(where: { $0.tone == .red })?.text, "No verified transcript upload on record",
                       "the fixture has another red line ahead of the alert")
        XCTAssertEqual(jobAlertStatus.attentionLine?.text, jobAlert)
        let page = BrainBarBackupsHealth.derive(job: job(.healthy), drive: drive(.missing), status: jobAlertStatus, statusUnavailableReason: "x")
        XCTAssertEqual(page, .init(badge: .attention, reason: jobAlert))
    }

    @MainActor
    func test_the_hero_and_the_one_page_summary_lead_with_the_job_alert() throws {
        let stats = BrainBarDashboardFixture.stats
        let collector = BrainBarDashboardFixture.makeCollector(stats: stats)
        let flow = DashboardFlowSummary.derive(daemon: collector.daemon, stats: stats, now: BrainBarOnePageTestFixture.now)
        let hero = BrainBarHeroPresentation.derive(
            flow: flow, stats: stats, backupTruth: .measured(jobAlertStatus), locale: Locale(identifier: "en_US")
        )
        XCTAssertEqual(hero.backupFailureReason, jobAlert)
        let page = BrainBarOnePagePresentation.derive(
            snapshotFreshness: .live(ageSeconds: 0), hero: hero, observability: try BrainBarOnePageTestFixture.healthyResult(),
            stats: stats, agentActivity: BrainBarDashboardFixture.agentActivity, now: BrainBarOnePageTestFixture.now,
            calendar: Calendar(identifier: .gregorian), locale: Locale(identifier: "en_US")
        )
        XCTAssertEqual(page.status.reason, jobAlert)
    }
}
