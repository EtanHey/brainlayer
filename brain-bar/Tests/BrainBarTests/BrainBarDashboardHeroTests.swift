import XCTest
@testable import BrainBar

final class BrainBarDashboardHeroTests: XCTestCase {
    @MainActor
    private func fixtureFlow(
        watcher state: WatcherFlowState = .flowing,
        ingressStatus: DashboardFlowLaneStatus? = nil
    ) -> DashboardFlowSummary {
        let baseline = DashboardFlowSummary.derive(
            daemon: BrainBarDashboardFixture.daemon,
            stats: BrainBarDashboardFixture.stats,
            now: BrainBarDashboardFixture.fetchedAt
        )
        let ingress = ingressStatus.map { status in
            DashboardFlowLane(
                name: baseline.ingress.name,
                status: status,
                statusText: baseline.ingress.statusText,
                windowLabel: baseline.ingress.windowLabel,
                activityWindowMinutes: baseline.ingress.activityWindowMinutes,
                rateText: baseline.ingress.rateText,
                volumeText: baseline.ingress.volumeText,
                lastEventText: baseline.ingress.lastEventText,
                values: baseline.ingress.values,
                sparklineLabel: baseline.ingress.sparklineLabel,
                latestBucketName: baseline.ingress.latestBucketName,
                accentColor: baseline.ingress.accentColor,
                primarySeriesLabel: baseline.ingress.primarySeriesLabel,
                secondaryValues: baseline.ingress.secondaryValues,
                secondarySeriesLabel: baseline.ingress.secondarySeriesLabel,
                secondaryAccentColor: baseline.ingress.secondaryAccentColor,
                tertiaryValues: baseline.ingress.tertiaryValues,
                tertiarySeriesLabel: baseline.ingress.tertiarySeriesLabel,
                tertiaryAccentColor: baseline.ingress.tertiaryAccentColor
            )
        } ?? baseline.ingress
        return DashboardFlowSummary(
            headline: baseline.headline,
            detail: baseline.detail,
            windowLabel: baseline.windowLabel,
            allCommits: baseline.allCommits,
            ingress: ingress,
            queue: baseline.queue,
            enrichment: baseline.enrichment,
            watcherFlowState: state,
            watcherHealth: baseline.watcherHealth,
            watcherHealthIsFresh: baseline.watcherHealthIsFresh
        )
    }

    private func healthyBackups() -> ObservabilityBackupStatus {
        let green = ObservabilityStatusLine(text: "verified", tone: .green)
        return .init(
            upload: green,
            snapshot: green,
            job: green,
            freshness: green,
            retention: green,
            archives: green,
            error: nil
        )
    }

    private func readableDocument() throws -> ObservabilityDocument {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "observability-main-58849a70",
            withExtension: "json",
            subdirectory: "Fixtures"
        ))
        guard case let .readable(document) = ObservabilityReader.read(url: url) else {
            throw XCTSkip("observability fixture was unreadable")
        }
        return document
    }

    private func document(
        generatedAt: Date,
        backups: ObservabilityDocument.Backups
    ) throws -> ObservabilityDocument {
        let base = try readableDocument()
        return ObservabilityDocument(
            schemaVersion: base.schemaVersion,
            generatedAt: generatedAt,
            dbPath: base.dbPath,
            windowHours: base.windowHours,
            stores: base.stores,
            emitters: base.emitters,
            authorUnknown: base.authorUnknown,
            backups: backups
        )
    }

    private func measuredBackups(at date: Date) -> ObservabilityDocument.Backups {
        .init(
            state: "measured",
            reason: "",
            inputs: [],
            freshness: "fresh",
            thresholdHours: 36,
            retentionInvariant: "PASS",
            survivingArchives30D: 3,
            errorType: nil,
            lastVerifiedUpload: .init(
                at: date,
                ageHours: 2,
                archiveId: "transcripts-2026-09-14.tar.zst",
                verified: true
            ),
            dbSnapshot: .init(
                lastAt: date,
                destination: "brainlayer-2026-09-14.db.gz",
                verified: true
            ),
            launchd: .init(
                label: "com.brainlayer.jsonl-backup",
                bootstrapped: true,
                disabledDirPresent: false
            )
        )
    }

    @MainActor
    func testHeroShowsHealthBackupsAndIndexedCountsWithoutEnrichment() {
        let stats = BrainBarDashboardFixture.stats
        let flow = DashboardFlowSummary.derive(
            daemon: BrainBarDashboardFixture.daemon,
            stats: stats,
            now: BrainBarDashboardFixture.fetchedAt
        )
        let backups = ObservabilityDocument.Backups(
            state: "measured",
            reason: "",
            inputs: [],
            freshness: "fresh",
            thresholdHours: 36,
            retentionInvariant: "PASS",
            survivingArchives30D: 3,
            errorType: nil,
            lastVerifiedUpload: .init(
                at: BrainBarDashboardFixture.fetchedAt,
                ageHours: 2,
                archiveId: "transcripts-2026-09-14.tar.zst",
                verified: true
            ),
            dbSnapshot: .init(
                lastAt: BrainBarDashboardFixture.fetchedAt,
                destination: "brainlayer-2026-09-14.db.gz",
                verified: true
            ),
            launchd: .init(
                label: "com.brainlayer.jsonl-backup",
                bootstrapped: true,
                disabledDirPresent: false
            )
        )
        let sharedBackupStatus = ObservabilityPresentation.backupStatus(
            for: backups,
            locale: Locale(identifier: "en_US")
        )

        let hero = BrainBarHeroPresentation.derive(
            flow: flow,
            stats: stats,
            backupTruth: .measured(sharedBackupStatus),
            locale: Locale(identifier: "en_US")
        )

        XCTAssertEqual(hero.healthTitle, "Health")
        XCTAssertEqual(hero.backupsTitle, "Backups")
        XCTAssertEqual(hero.indexedTitle, "Indexed")
        XCTAssertEqual(hero.healthVerdict, "Healthy")
        XCTAssertEqual(hero.healthTone, .green)
        XCTAssertEqual(hero.dbBackup, sharedBackupStatus.snapshot)
        XCTAssertEqual(hero.transcriptBackup, sharedBackupStatus.upload)
        XCTAssertEqual(hero.indexedInWindow, "67 chunk rows indexed in last 1h")
        XCTAssertEqual(hero.totalIndexed, "297,412 chunk rows total")

        let renderedHeroText = [
            hero.healthTitle,
            hero.healthVerdict,
            hero.healthReason,
            hero.backupsTitle,
            hero.dbBackup.text,
            hero.transcriptBackup.text,
            hero.indexedTitle,
            hero.indexedInWindow,
            hero.totalIndexed,
        ].joined(separator: "\n")
        XCTAssertFalse(renderedHeroText.localizedCaseInsensitiveContains("enrich"))
        XCTAssertFalse(renderedHeroText.contains("188,204"))
        XCTAssertFalse(renderedHeroText.contains("12,840"))
    }

    @MainActor
    func testBackupFailureMakesHealthRed() {
        let healthy = healthyBackups()
        let failure = ObservabilityStatusLine(text: "DB snapshot verification failed", tone: .red)
        let failed = ObservabilityBackupStatus(
            upload: healthy.upload,
            snapshot: failure,
            job: healthy.job,
            freshness: healthy.freshness,
            retention: healthy.retention,
            archives: healthy.archives,
            error: nil
        )

        let hero = BrainBarHeroPresentation.derive(
            flow: fixtureFlow(),
            stats: BrainBarDashboardFixture.stats,
            backupTruth: .measured(failed),
            locale: Locale(identifier: "en_US")
        )

        XCTAssertEqual(hero.healthTone, .red)
        XCTAssertEqual(hero.healthReason, failure.text)
    }

    @MainActor
    func testWatcherOfflineAndStalledMakeHealthRed() {
        for state in [WatcherFlowState.offline, .stalled] {
            let hero = BrainBarHeroPresentation.derive(
                flow: fixtureFlow(watcher: state),
                stats: BrainBarDashboardFixture.stats,
                backupTruth: .measured(healthyBackups())
            )
            XCTAssertEqual(hero.healthTone, .red, "state=\(state)")
        }
    }

    @MainActor
    func testUnavailableIngressMakesHealthRed() {
        let hero = BrainBarHeroPresentation.derive(
            flow: fixtureFlow(ingressStatus: .unavailable),
            stats: BrainBarDashboardFixture.stats,
            backupTruth: .measured(healthyBackups())
        )

        XCTAssertEqual(hero.healthTone, .red)
        XCTAssertEqual(hero.healthReason, "Ingest health is unavailable.")
    }

    @MainActor
    func testUnknownAndUnverifiedWatcherFlowMakeHealthAmber() {
        for state in [WatcherFlowState.unknown, .runningFlowUnverified] {
            let hero = BrainBarHeroPresentation.derive(
                flow: fixtureFlow(watcher: state),
                stats: BrainBarDashboardFixture.stats,
                backupTruth: .measured(healthyBackups())
            )
            XCTAssertEqual(hero.healthTone, .amber, "state=\(state)")
        }
    }

    @MainActor
    func testUnmeasurableBackupsUseCardWordingInsteadOfInventingMissingRecords() throws {
        let reason = "unmeasurable — backup log unreadable"
        let backups = ObservabilityDocument.Backups(
            state: "unmeasurable",
            reason: "backup log unreadable",
            inputs: [],
            freshness: nil,
            thresholdHours: nil,
            retentionInvariant: nil,
            survivingArchives30D: nil,
            errorType: nil,
            lastVerifiedUpload: nil,
            dbSnapshot: nil,
            launchd: nil
        )
        let truth = BrainBarHeroBackupTruth.derive(
            from: .readable(try document(generatedAt: BrainBarDashboardFixture.fetchedAt, backups: backups)),
            now: BrainBarDashboardFixture.fetchedAt,
            cadence: .known(300)
        )
        let hero = BrainBarHeroPresentation.derive(
            flow: fixtureFlow(),
            stats: BrainBarDashboardFixture.stats,
            backupTruth: truth
        )

        XCTAssertEqual(hero.dbBackup, .init(text: reason, tone: .neutral))
        XCTAssertEqual(hero.transcriptBackup, .init(text: reason, tone: .neutral))
        XCTAssertEqual(hero.healthReason, reason)
        XCTAssertFalse(hero.dbBackup.text.contains("No verified"))
        XCTAssertFalse(hero.transcriptBackup.text.contains("No verified"))
    }

    @MainActor
    func testStaleBackupDocumentCapsOtherwiseHealthyHeroAtAmber() throws {
        let generatedAt = BrainBarDashboardFixture.fetchedAt
        let document = try document(
            generatedAt: generatedAt,
            backups: measuredBackups(at: generatedAt)
        )
        let now = document.generatedAt.addingTimeInterval(3 * 24 * 60 * 60)
        let truth = BrainBarHeroBackupTruth.derive(
            from: .readable(document),
            now: now,
            cadence: .known(300)
        )
        let hero = BrainBarHeroPresentation.derive(
            flow: fixtureFlow(),
            stats: BrainBarDashboardFixture.stats,
            backupTruth: truth
        )

        XCTAssertEqual(hero.healthTone, .amber)
        XCTAssertTrue(hero.healthReason.contains("4320m old"))
    }

    @MainActor
    func testLoadingObservabilityIsCheckingNotRed() {
        let truth = BrainBarHeroBackupTruth.derive(
            from: .unreadable("Loading observability data."),
            now: BrainBarDashboardFixture.fetchedAt,
            cadence: .known(300)
        )
        let hero = BrainBarHeroPresentation.derive(
            flow: fixtureFlow(),
            stats: BrainBarDashboardFixture.stats,
            backupTruth: truth
        )

        XCTAssertEqual(hero.healthTone, .amber)
        XCTAssertEqual(hero.healthVerdict, "Checking health")
        XCTAssertEqual(hero.healthReason, "Loading observability data.")
    }
}
