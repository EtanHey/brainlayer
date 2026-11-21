import XCTest
@testable import BrainBar

final class BrainBarBackupProgressTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)

    private func backups(uploaded: Bool = false, pending: Bool = false, live: Bool = true,
                         total: Int64? = 100, goodAt: Date? = nil) throws -> ObservabilityDocument.Backups {
        let data = Data("""
        {"state":"measured","reason":"","inputs":[],
         "db_snapshot":{"last_at":\((goodAt ?? now.addingTimeInterval(-3600)).timeIntervalSinceReferenceDate),"destination":"old.db.gz","verified":true},
         "local_snapshot":{"last_at":\(now.timeIntervalSinceReferenceDate),"destination":"new.db.gz",
           "local_verified":true,"uploaded":\(uploaded),"upload_state":"in_progress",
           "upload_process_live":\(live),"upload_process_started_at":\(now.addingTimeInterval(-60).timeIntervalSinceReferenceDate),
           "bytes_confirmed":25,"bytes_total":\(total.map(String.init) ?? "null"),"verification_pending":\(pending)}}
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
        var done = try backups(uploaded: true, goodAt: now)
        done.localSnapshot = nil
        XCTAssertNil(BrainBarBackupProgress.read(done, now: now))
        let before = try backups()
        let oldReceipt = ObservabilityPresentation.backupReceiptLines(for: before, now: now).first?.text
        let newReceipt = ObservabilityPresentation.backupReceiptLines(for: done, now: now).first?.text
        XCTAssertNotEqual(oldReceipt, newReceipt, "Completion replaces the last good time on the existing receipt row")
    }

    func testInvalidCountsNeverBecomePercentages() throws {
        var p = try XCTUnwrap(BrainBarBackupProgress.read(backups(total: 0), now: now))
        XCTAssertNil(p.percentage)
        p.total = 10
        XCTAssertNil(p.percentage)
        p.confirmed = -1
        XCTAssertNil(p.percentage)
    }
}
