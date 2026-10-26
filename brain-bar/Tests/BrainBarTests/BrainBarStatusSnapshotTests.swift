import XCTest
@testable import BrainBar

/// S5 (Etan, 2026-10-26: "I want to never see a .dot there unless something really horribly goes
/// wrong"): ONE status snapshot, projected to the menu-bar icon, the menu text, the Dashboard, Jobs
/// and Backups. Every case below feeds one fixture to every surface and asserts they agree.
@MainActor
final class BrainBarStatusSnapshotTests: XCTestCase {
    nonisolated static let now = ISO8601DateFormatter().date(from: "2026-10-26T14:10:00Z")!
    /// 03:15 local (UTC+2) tomorrow morning: the real calendar trigger.
    nonisolated static let nextRun = ISO8601DateFormatter().date(from: "2026-10-27T01:15:00Z")!
    nonisolated static var calendar: Calendar {
        var value = Calendar(identifier: .gregorian)
        value.timeZone = TimeZone(secondsFromGMT: 7200)!
        return value
    }

    nonisolated static func hoursAgo(_ hours: Double) -> Date { now.addingTimeInterval(-hours * 3600) }

    nonisolated static func backups(db: Date? = hoursAgo(6), transcripts: Date? = hoursAgo(12), error: String? = nil,
                        threshold: Double = 36) -> ObservabilityDocument.Backups {
        .init(state: "measured", reason: "", inputs: [], freshness: "fresh", thresholdHours: threshold,
              retentionInvariant: "PASS", survivingArchives30D: 27, errorType: error,
              lastVerifiedUpload: transcripts.map { .init(at: $0, ageHours: now.timeIntervalSince($0) / 3600, archiveId: "fixture", verified: true) },
              dbSnapshot: db.map { .init(lastAt: $0, destination: "fixture.db.zst", verified: true) },
              launchd: .init(label: "com.brainlayer.jsonl-backup", bootstrapped: true, disabledDirPresent: false))
    }

    nonisolated static func step(_ state: BrainLayerLaunchdLoadState = .loaded, code: Int32? = 0, at: Date? = hoursAgo(12),
                     next: Date? = nextRun, consecutiveFailures: Int = 0) -> BrainLayerLaunchdJobObservation {
        var value = BrainLayerLaunchdJobObservation(loadState: state, runs: nil, lastExitCode: code,
                                                    lastRunAt: at, nextRunAt: next, isContinuous: false)
        value.ownership = .supervisor
        value.calendarConsecutiveFailures = consecutiveFailures
        value.ownerNote = "com.brainlayer.maintenance · last result: " + (code == 0 ? "ok" : "failed")
        return value
    }

    nonisolated static func failed(_ job: BrainLayerLaunchdJob, code: Int32, at: Date = hoursAgo(12), consecutive: Int = 1,
                       next: Date? = nextRun) -> BrainLayerLaunchdJobObservation {
        var value = step(.failed("\(job.title): calendar result failed, exit \(code)"), code: code, at: at,
                         next: next, consecutiveFailures: consecutive)
        value.ownerNoteTone = .warning
        return value
    }

    nonisolated static let connected = DriveAuthPresentation(line: "Connected", detail: nil, tone: .connected,
                                                 showsReconnect: false, reconnectEnabled: false, needsAttention: false)

    nonisolated static func healthy() -> BrainBarStatusInputs {
        var observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] = [:]
        for job in [BrainLayerLaunchdJob.index, .backupDaily, .jsonlBackup, .maintenanceNightly, .maintenanceWeekly] {
            observations[job] = step()
        }
        var watch = BrainLayerLaunchdJobObservation(loadState: .running, runs: 1, lastExitCode: nil,
                                                    lastRunAt: hoursAgo(1), nextRunAt: nil, isContinuous: true)
        watch.ownership = .supervisor
        observations[.watch] = watch
        let settings = Dictionary(uniqueKeysWithValues: BrainLayerLaunchdJob.allCases.map {
            ($0, BrainLayerLaunchdJobSetting(enabled: true, loadState: observations[$0]?.loadState ?? .loaded))
        })
        let heartbeat = now.addingTimeInterval(-30)
        return BrainBarStatusInputs(
            now: now, calendar: calendar,
            badge: .init(badgeOn: false, reason: "", activeCodes: []),
            jobAlerts: BrainBarJobAlerts(active: [:]),
            watcher: .running(heartbeatAt: heartbeat), watcherHeartbeatAt: heartbeat,
            settings: settings, observations: observations,
            backups: .measured(backups()), drive: connected,
            maintenance: .init(weeklyCompletion: .completed(now.addingTimeInterval(-2 * 86_400)), runRecords: [:])
        )
    }

    struct Surfaces: Sendable {
        let icon: Bool
        let menu: String
        let dashboard: BrainBarOnePageStatus
        let dashboardItems: [String]
        let jobs: [BrainLayerLaunchdJobGroup: BrainLayerLaunchdGroupStatus]
        let backups: BrainBarBackupsHealth
    }

    nonisolated static func surfaces(_ inputs: BrainBarStatusInputs) -> (BrainBarStatusSnapshot, Surfaces) {
        let snapshot = BrainBarStatusSnapshot.derive(inputs)
        var jobs: [BrainLayerLaunchdJobGroup: BrainLayerLaunchdGroupStatus] = [:]
        for group in BrainLayerLaunchdJobGroup.allCases {
            let raw = group.status(settings: inputs.settings, observations: inputs.observations ?? [:],
                                   formatDate: DashboardMetricFormatter.jobDateTimeString,
                                   watcher: inputs.watcher, maintenance: inputs.maintenance, now: inputs.now)
            jobs[group] = snapshot.project(raw, group: group)
        }
        let backupStatus: ObservabilityBackupStatus? = if case let .measured(value) = inputs.backups {
            ObservabilityPresentation.backupStatus(for: value, now: inputs.now)
        } else { nil }
        let backups = snapshot.project(BrainBarBackupsHealth.derive(job: jobs[.backups]!, drive: inputs.drive,
            status: backupStatus, statusUnavailableReason: "Backup status is unmeasurable."))
        return (snapshot, Surfaces(
            icon: snapshot.needsAttention,
            menu: BrainBarStatusPopoverController.statusLineTitle(for: snapshot.verdict),
            dashboard: snapshot.dashboardStatus,
            dashboardItems: snapshot.attention.map(\.text),
            jobs: jobs, backups: backups
        ))
    }

    /// Asserts the four surfaces agree on the dot; returns them for case-specific checks.
    @discardableResult
    nonisolated static func assertAgree(_ inputs: BrainBarStatusInputs, dot: Bool, _ name: String,
                            file: StaticString = #filePath, line: UInt = #line) -> (BrainBarStatusSnapshot, Surfaces) {
        let (snapshot, value) = surfaces(inputs)
        XCTAssertEqual(value.icon, dot, "\(name): icon dot", file: file, line: line)
        XCTAssertEqual(value.menu.hasPrefix("Needs attention:"), dot, "\(name): menu says \(value.menu)", file: file, line: line)
        XCTAssertEqual(value.dashboard.tone == .amber, dot, "\(name): dashboard \(value.dashboard)", file: file, line: line)
        XCTAssertEqual(!value.dashboardItems.isEmpty, dot, "\(name): dashboard items \(value.dashboardItems)", file: file, line: line)
        let unhealthyJobs = value.jobs.values.contains { $0.health == .unhealthy }
        let jobDot = snapshot.attention.contains { !$0.jobs.isEmpty || $0.component == .watcher }
        XCTAssertEqual(unhealthyJobs, jobDot, "\(name): jobs \(value.jobs.mapValues(\.health))", file: file, line: line)
        if !dot {
            XCTAssertFalse(unhealthyJobs, "\(name): no dot, but Jobs says Needs attention", file: file, line: line)
            XCTAssertNotEqual(value.backups.badge, .attention, "\(name): no dot, but Backups says \(value.backups)", file: file, line: line)
        }
        return (snapshot, value)
    }

    // MARK: no dot

    func testHealthyFixtureIsAllGoodEverywhere() {
        let (_, value) = Self.assertAgree(Self.healthy(), dot: false, "healthy")
        XCTAssertEqual(value.menu, "Nothing needs attention")
        XCTAssertEqual(value.dashboard.headline, "All good")
        XCTAssertEqual(value.dashboard.tone, .green)
        XCTAssertEqual(value.backups.badge, .healthy)
    }

    func testNothingReadYetIsCheckingAndNeverAllGoodNorADot() {
        let inputs = BrainBarStatusInputs(now: Self.now, calendar: Self.calendar)
        let (snapshot, value) = Self.assertAgree(inputs, dot: false, "unread")
        XCTAssertTrue(snapshot.isChecking)
        XCTAssertNotEqual(value.dashboard.headline, "All good")
        XCTAssertEqual(value.dashboard.tone, .neutral)
        XCTAssertEqual(value.menu, "Checking status…")
    }

    func testDeferredWeeklyMaintenanceWithNewerVerifiedCopyIsANeutralNoticeEverywhere() {
        var inputs = Self.healthy()
        // 02:00Z exit 76; the verified DB copy at 08:30Z is newer; next run 03:15 local is pending.
        inputs.observations?[.maintenanceWeekly] = Self.failed(.maintenanceWeekly, code: 76,
            at: ISO8601DateFormatter().date(from: "2026-10-26T02:00:16Z")!)
        inputs.backups = .measured(Self.backups(db: ISO8601DateFormatter().date(from: "2026-10-26T08:30:02Z")!))
        inputs.jobAlerts = BrainBarJobAlerts(active: ["maintenance-full":
            "BrainLayer full maintenance failed: tonight's local verified snapshot unavailable; VACUUM skipped."])
        let (snapshot, value) = Self.assertAgree(inputs, dot: false, "deferred weekly")
        let notice = snapshot.notices.first { $0.jobs.contains(.maintenanceWeekly) }
        XCTAssertEqual(notice?.text, "Weekly maintenance deferred — retries tonight 03:15")
        XCTAssertEqual(snapshot.notices.filter { $0.jobs.contains(.maintenanceWeekly) }.count, 1,
                       "the job alert and the calendar failure are ONE issue: \(snapshot.notices)")
        XCTAssertEqual(value.menu, "Info: Weekly maintenance deferred: retries tonight 03:15")
        XCTAssertNotEqual(value.dashboard.headline, "All good")
        XCTAssertEqual(value.dashboard.reason, "Weekly maintenance deferred — retries tonight 03:15")
        XCTAssertEqual(value.jobs[.maintenance]?.health, .deferred)
        XCTAssertEqual(value.jobs[.maintenance]?.attentionReason, "Weekly maintenance deferred — retries tonight 03:15")
        XCTAssertEqual(value.backups.badge, .healthy, "a maintenance deferral is not a backup failure")
    }

    func testScheduleUnreadableWithHistoryIsNeutralNotAllGood() {
        var inputs = Self.healthy()
        for job in [BrainLayerLaunchdJob.index, .backupDaily, .jsonlBackup, .maintenanceNightly, .maintenanceWeekly] {
            var value = Self.step(next: nil)
            value.calendarScheduleUncertainty = "Maintenance calendar schedule unavailable"
            inputs.observations?[job] = value
        }
        let (_, value) = Self.assertAgree(inputs, dot: false, "schedule unreadable")
        XCTAssertNotEqual(value.dashboard.headline, "All good")
        XCTAssertTrue(value.menu.hasPrefix("Info:"), value.menu)
    }

    func testProbeErrorAndUnknownBadgeAreNeutralNeverADot() {
        var inputs = Self.healthy()
        inputs.observations?[.backupDaily] = .stateOnly(.probeError("Maintenance calendar status unavailable"))
        inputs.observations?[.maintenanceNightly] = .stateOnly(.probeError("Maintenance calendar status unavailable"))
        inputs.badge = .failVisible("Badge state is stale.")
        inputs.watcher = .unknown(reason: "Watcher is running, but its health file has not been read yet.")
        let (_, value) = Self.assertAgree(inputs, dot: false, "unknown")
        XCTAssertNotEqual(value.dashboard.headline, "All good")
        XCTAssertNotEqual(value.jobs[.backups]?.health, .unhealthy)
        XCTAssertNotEqual(value.jobs[.maintenance]?.health, .unhealthy)
    }

    func testSingleRequiredJobFailureIsANoticeAndTheSecondConsecutiveIsADot() {
        var inputs = Self.healthy()
        inputs.observations?[.index] = Self.failed(.index, code: 1, consecutive: 1)
        Self.assertAgree(inputs, dot: false, "index failed once")
        inputs.observations?[.index] = Self.failed(.index, code: 1, consecutive: 2)
        let (snapshot, value) = Self.assertAgree(inputs, dot: true, "index failed twice")
        XCTAssertEqual(value.jobs[.ingest]?.health, .unhealthy)
        XCTAssertTrue(snapshot.attention.first?.text.contains("2 consecutive") == true, "\(snapshot.attention)")
    }

    func testWarningOnlyBadgeIsInfoNotADot() {
        var inputs = Self.healthy()
        inputs.badge = .init(badgeOn: false, reason: "queue backed up", activeCodes: ["queue_backed_up"],
                             activeMessages: ["queue backed up"], activeSeverities: ["warning"])
        let (_, value) = Self.assertAgree(inputs, dot: false, "warning badge")
        XCTAssertEqual(value.menu, "Info: queue backed up")
    }

    func testDriveExpiringIsANoticeAndDriveLostIsADot() {
        var inputs = Self.healthy()
        inputs.drive = .init(line: "1 thing needs you: reconnect Drive by Oct 28", detail: nil, tone: .expiring,
                             showsReconnect: true, reconnectEnabled: true, needsAttention: true)
        Self.assertAgree(inputs, dot: false, "drive expiring")
        inputs.drive = .init(line: "Drive access expired or was revoked. Backups can't upload.", detail: nil,
                             tone: .attention, showsReconnect: true, reconnectEnabled: true, needsAttention: true)
        let (_, value) = Self.assertAgree(inputs, dot: true, "drive lost")
        XCTAssertEqual(value.backups.badge, .attention)
    }

    // MARK: dot

    func testStaleDatabaseCopyWithFreshTranscriptsIsADot() {
        var inputs = Self.healthy()
        inputs.backups = .measured(Self.backups(db: Self.hoursAgo(37)))
        let (_, value) = Self.assertAgree(inputs, dot: true, "stale DB copy")
        XCTAssertEqual(value.backups.badge, .attention)
    }

    func testStaleTranscriptCopyWithFreshDatabaseIsADot() {
        var inputs = Self.healthy()
        inputs.backups = .measured(Self.backups(transcripts: Self.hoursAgo(37)))
        Self.assertAgree(inputs, dot: true, "stale transcripts")
        inputs.backups = .measured(Self.backups(transcripts: Self.hoursAgo(35)))
        Self.assertAgree(inputs, dot: false, "35 h is inside the 36 h threshold")
    }

    func testBackupFailureWithoutNewerVerifiedCopyIsADotAndWithOneIsNot() {
        var inputs = Self.healthy()
        inputs.observations?[.backupDaily] = Self.failed(.backupDaily, code: 1, at: Self.hoursAgo(4))
        inputs.backups = .measured(Self.backups(db: Self.hoursAgo(10)))
        Self.assertAgree(inputs, dot: true, "backup failed after the newest copy")
        inputs.backups = .measured(Self.backups(db: Self.hoursAgo(2)))
        Self.assertAgree(inputs, dot: false, "a newer verified copy exists")
    }

    func testDeferredMaintenanceReturnsToADotOnEachLeadCondition() {
        let failureAt = ISO8601DateFormatter().date(from: "2026-10-26T02:00:16Z")!
        let newerCopy = Self.backups(db: ISO8601DateFormatter().date(from: "2026-10-26T08:30:02Z")!)
        var base = Self.healthy()
        base.backups = .measured(newerCopy)
        base.observations?[.maintenanceWeekly] = Self.failed(.maintenanceWeekly, code: 76, at: failureAt)
        Self.assertAgree(base, dot: false, "single exit 76")

        // Lead amendment L1 (supersedes the 2-failure / 8-day / newer-copy / pending-run rule).
        var second = base
        second.observations?[.maintenanceWeekly] = Self.failed(.maintenanceWeekly, code: 76, at: failureAt, consecutive: 2)
        Self.assertAgree(second, dot: false, "second consecutive exit 76")

        var third = base
        third.observations?[.maintenanceWeekly] = Self.failed(.maintenanceWeekly, code: 76, at: failureAt, consecutive: 3)
        Self.assertAgree(third, dot: true, "third consecutive exit 76")

        var nineDays = base
        nineDays.maintenance = .init(weeklyCompletion: .completed(Self.now.addingTimeInterval(-9 * 86_400)), runRecords: [:])
        Self.assertAgree(nineDays, dot: false, "9 days is inside the 14-day housekeeping window")

        var stale = base
        stale.maintenance = .init(weeklyCompletion: .completed(Self.now.addingTimeInterval(-15 * 86_400)), runRecords: [:])
        Self.assertAgree(stale, dot: true, "last successful weekly > 14 days")

        var noCopy = base
        noCopy.backups = .measured(Self.backups(db: Self.hoursAgo(14)))
        Self.assertAgree(noCopy, dot: false, "maintenance severity does not depend on a newer DB copy")

        var noNextRun = base
        noNextRun.observations?[.maintenanceWeekly] = Self.failed(.maintenanceWeekly, code: 76, at: failureAt, next: nil)
        Self.assertAgree(noNextRun, dot: false, "a missing next run is uncertainty, not an emergency")
    }

    func testMalformedMaintenancePlistIsADotAndDedupesWithTheHealthCheck() {
        var inputs = Self.healthy()
        let reason = "com.brainlayer.maintenance.plist is malformed: not a property-list dictionary"
        for job in [BrainLayerLaunchdJob.index, .backupDaily, .jsonlBackup, .maintenanceNightly, .maintenanceWeekly] {
            inputs.observations?[job]?.calendarScheduleProblem = reason
        }
        inputs.badge = .init(badgeOn: true,
            reason: "cannot inspect BrainLayer plist /Users/x/Library/LaunchAgents/com.brainlayer.maintenance.plist: Invalid file",
            activeCodes: ["job_scan_failed"],
            activeMessages: ["cannot inspect BrainLayer plist /Users/x/Library/LaunchAgents/com.brainlayer.maintenance.plist: Invalid file"],
            activeSeverities: ["critical"])
        let (snapshot, _) = Self.assertAgree(inputs, dot: true, "malformed plist")
        XCTAssertEqual(snapshot.attention.filter { $0.component == .configuration }.count, 1, "\(snapshot.attention)")
    }

    func testIssuesDedupeByIdentityNotSentence() {
        var inputs = Self.healthy()
        inputs.backups = .measured(Self.backups(db: Self.hoursAgo(40), transcripts: Self.hoursAgo(40)))
        let snapshot = BrainBarStatusSnapshot.derive(inputs)
        XCTAssertEqual(Set(snapshot.issues.map(\.id)).count, snapshot.issues.count)
        XCTAssertEqual(snapshot.attention.count, 2, "DB and transcripts are two issues: \(snapshot.attention)")
    }

    func testSameInputsGiveTheSameVerdictEveryTime() {
        var inputs = Self.healthy()
        inputs.backups = .measured(Self.backups(db: Self.hoursAgo(37)))
        XCTAssertEqual(BrainBarStatusSnapshot.derive(inputs), BrainBarStatusSnapshot.derive(inputs))
    }

    // MARK: real app surfaces

    func testRealOnePagePresentationUsesTheSnapshotHeadlineAndItems() {
        var inputs = Self.healthy()
        inputs.backups = .measured(Self.backups(db: Self.hoursAgo(37)))
        let snapshot = BrainBarStatusSnapshot.derive(inputs)
        let collector = BrainBarDashboardFixture.makeCollector(watcherHeartbeatAt: Self.now)
        let flow = DashboardFlowSummary.derive(daemon: collector.daemon, stats: collector.stats, now: Self.now)
        let hero = BrainBarHeroPresentation.derive(flow: flow, stats: collector.stats,
            backupTruth: .unavailable("fixture"), statusVerdict: snapshot.verdict)
        XCTAssertEqual(hero.healthVerdict, "Needs attention")
        XCTAssertEqual(hero.healthReason, snapshot.attention.first?.text)
        let page = BrainBarOnePagePresentation.derive(snapshotFreshness: .live(ageSeconds: 0), hero: hero,
            observability: .unreadable("fixture"), stats: collector.stats, agentActivity: collector.agentActivity,
            now: Self.now, statusVerdict: snapshot.verdict)
        XCTAssertEqual(page.attentionItems, snapshot.attention.map(\.text))
        XCTAssertEqual(page.status, snapshot.dashboardStatus)

        let calm = BrainBarStatusSnapshot.derive(Self.healthy())
        let calmHero = BrainBarHeroPresentation.derive(flow: flow, stats: collector.stats,
            backupTruth: .unavailable("fixture"), statusVerdict: calm.verdict)
        let calmPage = BrainBarOnePagePresentation.derive(snapshotFreshness: .live(ageSeconds: 0), hero: calmHero,
            observability: .unreadable("fixture"), stats: collector.stats, agentActivity: collector.agentActivity,
            now: Self.now, statusVerdict: calm.verdict)
        XCTAssertEqual(calmPage.attentionItems, [], "an unreadable observability file is not an action for Etan")
        XCTAssertNotEqual(calmPage.status.tone, .amber)
    }
}

extension ObservabilityDocument {
    /// The fixture document with these backups, generated at `generatedAt`.
    func replacingBackupsForStatusTests(_ backups: Backups, generatedAt: Date) -> Self {
        Self(schemaVersion: schemaVersion, generatedAt: generatedAt, dbPath: dbPath, windowHours: windowHours,
             stores: stores, emitters: emitters, authorUnknown: authorUnknown, backups: backups)
    }
}
