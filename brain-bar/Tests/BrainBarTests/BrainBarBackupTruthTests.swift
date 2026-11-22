import XCTest
@testable import BrainBar

@MainActor final class BrainBarBackupTruthTests: XCTestCase {
    private var now: Date { BrainBarDesignFixture.now }

    private func upload(_ hours: Double, ok: Bool = true) -> String {
        let at = ISO8601DateFormatter().string(from: now.addingTimeInterval(-hours * 3600))
        return "{\"attempted_at\":\"\(at)\",\"uploaded\":\(ok),\"verified\":\(ok)}"
    }

    private func parse(_ lines: [String]) throws -> (attempts: [NDAttempt]?, skipped: Int) {
        try XCTUnwrap(NDBackupHistoryReader.parseConversations(.init(lines: lines.map { Substring($0) }, unreadable: false), now: now))
    }

    private let retention = #"{"event":"local_retention_error","error":"invented inventory failure"}"#

    private func lanes(_ attempts: [NDAttempt]?) -> [NDBackupsPage.Lane] {
        var inputs = BrainBarDesignFixture.inputs(.calm)
        inputs.backupHistory = .read(.init(local: [], drive: [], conversations: attempts, integrity: []))
        return NDBackupsPage(store: NDStore(fixture: inputs)).lanes(NDModel.make(inputs))
    }

    func testRetentionErrorUsesNearestDatedNeighbourWithoutDashingLane() throws {
        let result = try parse([upload(6), retention, upload(3)])
        XCTAssertEqual(result.skipped, 0)
        XCTAssertEqual(result.attempts?.map(\.ok), [true, false, true])
        XCTAssertEqual(result.attempts?[1].at, now.addingTimeInterval(-6 * 3600))
        XCTAssertFalse(lanes(result.attempts).contains { $0.unknown })
        XCTAssertTrue(lanes(result.attempts).contains { $0.id == "transcripts" && $0.events.contains { $0.kind == .fail } })
    }

    func testOldRetentionErrorFallsOutsideWindowAndUndatedOnlyLogIsNotDamage() throws {
        let old = try parse([upload(300), retention, upload(250), upload(3)])
        XCTAssertEqual(old.attempts?.map(\.ok), [true])
        XCTAssertEqual(old.skipped, 0)
        let undated = try parse([retention])
        XCTAssertEqual(undated.attempts, [])
        XCTAssertEqual(undated.skipped, 0)
        let following = try parse([retention, upload(3)])
        XCTAssertEqual(following.attempts?.map(\.ok), [false, true])
        XCTAssertEqual(following.attempts?.first?.at, now.addingTimeInterval(-3 * 3600))
    }

    func testGarbageStillDashesLaneAndFailedUploadStillDrawsFailure() throws {
        let damaged = try parse([upload(6), "{garbage", upload(3)])
        XCTAssertNil(damaged.attempts)
        XCTAssertEqual(damaged.skipped, 1)
        XCTAssertTrue(NDBackupsPage.legend(lanes(damaged.attempts)).contains(BrainBarMenuPresenter.SurfaceSay.unreadRecord))
        let failed = try parse([upload(6), upload(3, ok: false)])
        XCTAssertEqual(failed.attempts?.map(\.ok), [true, false])
        XCTAssertTrue(lanes(failed.attempts).contains { $0.id == "transcripts" && $0.events.contains { $0.kind == .fail } })
    }

    func testStalledProgressNamesTheStallAndMovingProgressDoesNot() throws {
        var progress = try XCTUnwrap(BrainBarBackupProgress.read(BrainBarDesignFixture.backupFacts(.uploadProgress), now: BrainBarDesignFixture.now))
        XCTAssertFalse(progress.title.contains(BrainBarMenuPresenter.SurfaceSay.notUploadingSince))
        progress.isStalled = true
        XCTAssertTrue(progress.title.contains(BrainBarMenuPresenter.SurfaceSay.notUploadingSince))
    }

    func testStalledFixtureNamesTheStall() throws {
        let progress = try XCTUnwrap(BrainBarDesignFixture.inputs(.uploadStalled).backupProgress)
        XCTAssertTrue(progress.isStalled)
        XCTAssertTrue(progress.title.contains(BrainBarMenuPresenter.SurfaceSay.notUploadingSince))
    }
}
