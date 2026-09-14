import XCTest
@testable import BrainBar

final class BrainBarDashboardHeroTests: XCTestCase {
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
            locale: Locale(identifier: "en_US_POSIX")
        )

        let hero = BrainBarHeroPresentation.derive(
            flow: flow,
            stats: stats,
            backupTruth: .measured(sharedBackupStatus),
            locale: Locale(identifier: "en_US_POSIX")
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
}
