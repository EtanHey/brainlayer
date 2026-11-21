import XCTest
@testable import BrainBar

@MainActor final class BrainBarBackupProgressTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)

    private func backups(uploaded: Bool = false, pending: Bool = false, live: Bool = true,
                         total: Int64? = 100, goodAt: Date? = nil, state: String = "in_progress",
                         checkpointAge: TimeInterval? = 30, localVerified: Bool = true,
                         localAge: TimeInterval = 0) throws -> ObservabilityDocument.Backups {
        let data = Data("""
        {"state":"measured","reason":"","inputs":[],"threshold_hours":36,
         "db_snapshot":{"last_at":\((goodAt ?? now.addingTimeInterval(-3600)).timeIntervalSinceReferenceDate),"destination":"old.db.gz","verified":true},
         "local_snapshot":{"last_at":\(now.addingTimeInterval(-localAge).timeIntervalSinceReferenceDate),"destination":"new.db.gz",
           "local_verified":\(localVerified),"uploaded":\(uploaded),"upload_state":"\(state)",
           "upload_process_live":\(live),"upload_process_started_at":\(now.addingTimeInterval(-60).timeIntervalSinceReferenceDate),
           "bytes_confirmed":25,"bytes_total":\(total.map(String.init) ?? "null"),"verification_pending":\(pending),
           "bytes_confirmed_at":\(checkpointAge.map { String(now.addingTimeInterval(-$0).timeIntervalSinceReferenceDate) } ?? "null")}}
        """.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(ObservabilityDocument.Backups.self, from: data)
    }

    func testMeasuredBytesReachPercentage() throws {
        let p = try XCTUnwrap(BrainBarBackupProgress.read(backups(), now: now))
        XCTAssertEqual(p.percentage, 25)
        XCTAssertEqual(p.fraction, 0.25)
        XCTAssertEqual(p.phase, .upload)
        XCTAssertEqual(p.lastGoodAt, now.addingTimeInterval(-3600))
    }

    func testMissingTotalIsIndeterminateWithNoNumber() throws {
        let p = try XCTUnwrap(BrainBarBackupProgress.read(backups(total: nil), now: now))
        XCTAssertNil(p.fraction)
        XCTAssertNil(p.percentage)
    }

    func testOnlyAdvancingBytesAnimateAndReduceMotionDisablesIt() throws {
        let old = try XCTUnwrap(BrainBarBackupProgress.read(backups(), now: now))
        var next = old
        next.confirmed = 50
        XCTAssertTrue(next.animates(after: old, reduceMotion: false))
        XCTAssertFalse(next.animates(after: old, reduceMotion: true))
        XCTAssertFalse(old.animates(after: old, reduceMotion: false))
        XCTAssertFalse(old.animates(after: next, reduceMotion: false))
        next.identity = "different run"
        XCTAssertFalse(next.animates(after: old, reduceMotion: false))
    }

    func testVerificationNeverReusesUploadPercentage() throws {
        let p = try XCTUnwrap(BrainBarBackupProgress.read(backups(uploaded: true, pending: true), now: now))
        XCTAssertEqual(p.phase, .verification)
        XCTAssertEqual(p.title, BrainBarMenuPresenter.SurfaceSay.backupVerifying)
        XCTAssertNil(p.percentage)
        XCTAssertFalse(p.animates(after: p, reduceMotion: false))
    }

    func testInterruptedOrFinishedCheckpointDoesNotClaimRunning() throws {
        XCTAssertNil(try BrainBarBackupProgress.read(backups(live: false), now: now))
        XCTAssertNil(try BrainBarBackupProgress.read(backups(uploaded: true), now: now))
        let checking = try backups(uploaded: true, pending: true)
        XCTAssertNotNil(BrainBarBackupProgress.read(checking, now: now))
        let done = try backups(uploaded: true, pending: false, state: "verified")
        XCTAssertNil(BrainBarBackupProgress.read(done, now: now))
    }

    func testInvalidCountsNeverBecomePercentages() throws {
        var p = try XCTUnwrap(BrainBarBackupProgress.read(backups(total: 0), now: now))
        XCTAssertNil(p.percentage)
        p.total = 10
        XCTAssertNil(p.percentage)
        p.confirmed = -1
        XCTAssertNil(p.percentage)
    }

    func testClassicRedOutputAndHeaderCountRemainWithProgress() throws {
        let b = try backups(goodAt: now.addingTimeInterval(-3 * 86400))
        let p = try XCTUnwrap(BrainBarBackupProgress.read(b, now: now))
        let checks = BrainBarBackupChecks.derive(b, now: now)
        let output = checks.checks.compactMap { BrainBarBackupChecksCard.displayedCheck($0, progress: p) }
        let row = try XCTUnwrap(output.first { $0.id == "snapshot-upload" })
        XCTAssertEqual(row.tone, .red)
        XCTAssertTrue(row.value.contains("upload still pending"))
        XCTAssertEqual(output.filter { $0.tone == .red }.count, checks.checks.filter { $0.tone == .red }.count)
    }

    func testClassicAmberAndCleanOutputStayBesideProgress() throws {
        let p = try XCTUnwrap(BrainBarBackupProgress.read(backups(), now: now))
        for tone in [ObservabilityStatusTone.amber, .green] {
            let row = BrainBarBackupChecks.Check(id: "snapshot-upload", title: "Local database snapshot", value: "existing words", tone: tone)
            XCTAssertEqual(BrainBarBackupChecksCard.displayedCheck(row, progress: p), row)
        }
    }

    func testChartRedAndAmberOutputKeepsStatusWords() throws {
        let p = try XCTUnwrap(BrainBarBackupProgress.read(backups(), now: now))
        for severity in [NDSeverity.red, .orange] {
            var state = NDPipelineState.calm
            state.status[.drive] = .init(text: severity == .red ? "Signed out" : "Last upload 40 h ago", severity: severity)
            XCTAssertEqual(NDPipelineChart.drawingState(state, progress: p).status[.drive], state.status[.drive])
            // Gap 22/23, round 13: a compact bar accompanies the fault instead of suppressing progress.
            XCTAssertTrue(NDPipelineChart.showsProgress(state, progress: p))
            let frame = NDPipelineChart.progressFrame(state)
            XCTAssertEqual(frame.height, 3)
            XCTAssertGreaterThan(frame.minY, NDPipelineLayout.point(.drive).y + 46)
            XCTAssertLessThan(frame.maxY, NDPipelineLayout.point(.queue).y)
        }
        XCTAssertTrue(NDPipelineChart.showsProgress(.calm, progress: p))
    }

    func testBackupLaneOutputRetainsRedTimelineAndColumns() throws {
        let p = try XCTUnwrap(BrainBarBackupProgress.read(backups(), now: now))
        let lane = NDBackupsPage.Lane(id: "backup_offsite", name: "Google Drive", events: [.init(hours: -72, kind: .fail), .init(hours: 2, kind: .next)],
            unknown: false, stalledFrom: -72, number: .init(value: "3", unit: "d", tone: .fault), strong: true, accessibility: "late")
        let output = try XCTUnwrap(NDBackupsPage.displayedLane(lane, progress: p))
        XCTAssertTrue(output.strong)
        XCTAssertEqual(output.name, lane.name)
        XCTAssertEqual(output.stalledFrom, -72)
        XCTAssertEqual(output.events.count, 2)
        XCTAssertEqual(output.number, lane.number)
    }

    func testOldOrUndatedCheckpointsLoseNumberAndHaveAccessibleValue() throws {
        let fresh = try XCTUnwrap(BrainBarBackupProgress.read(backups(), now: now))
        XCTAssertEqual(fresh.checkpointAt, now.addingTimeInterval(-30))
        XCTAssertEqual(fresh.accessibilityValue, "25 percent")
        for age in [TimeInterval?.none, .some(1201)] {
            let p = try XCTUnwrap(BrainBarBackupProgress.read(backups(checkpointAge: age), now: now))
            XCTAssertNil(p.percentage)
            XCTAssertEqual(p.isStalled, age == .some(1201))
            XCTAssertEqual(p.accessibilityValue, BrainBarMenuPresenter.SurfaceSay.backupInProgress)
        }
    }

    func testResumedVerificationIsCheckingRatherThanQueued() throws {
        let b = try backups(uploaded: true, pending: true, state: "pending")
        let p = try XCTUnwrap(BrainBarBackupProgress.read(b, now: now))
        XCTAssertEqual(p.phase, .verification)
        XCTAssertNil(p.percentage)
        XCTAssertEqual(ObservabilityPresentation.backupStatus(for: b, now: now).snapshotUpload?.text,
                       BrainBarMenuPresenter.SurfaceSay.backupVerifying)
        XCTAssertFalse(ObservabilityPresentation.backupStatus(for: b, now: now).uploadQueued)
    }

    func testNeutralCheckingRequiresVerifiedRecentLocalCopy() throws {
        for b in [try backups(uploaded: true, pending: true, state: "pending", localVerified: false),
                  try backups(uploaded: true, pending: true, state: "pending", localAge: 37 * 3600)] {
            let row = try XCTUnwrap(ObservabilityPresentation.backupStatus(for: b, now: now).snapshotUpload)
            XCTAssertEqual(row.tone, .red)
            XCTAssertNotEqual(row.text, ObservabilityPresentation.backupVerificationText)
        }
    }
}
