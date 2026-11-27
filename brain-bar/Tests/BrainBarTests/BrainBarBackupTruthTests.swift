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

    func testRailShowsRetentionFailureInsteadOfSameTimeSuccessInEitherInputOrder() throws {
        let attempts = try XCTUnwrap(parse([upload(6), retention, upload(3)]).attempts)
        for ordered in [attempts, Array(attempts.reversed())] {
            let lane = try XCTUnwrap(lanes(ordered).first { $0.id == "transcripts" })
            let shown = NDRail.shown(lane.events, width: 480, past: 144, ahead: 24)
            XCTAssertEqual(shown.filter { $0.hours == -3 }.map(\.kind), [.fail])
            XCTAssertEqual(shown.filter { $0.hours == -6 }.map(\.kind), [.ok])
        }
    }

    func testNextRingNearNowCannotHideFailureAtSameTimeAsSuccess() {
        let events: [NDRail.Event] = [.init(hours: 0, kind: .fail), .init(hours: 0, kind: .ok), .init(hours: 0.1, kind: .next)]
        XCTAssertEqual(NDRail.shown(events, width: 480, past: 144, ahead: 24).map(\.kind), [.next, .fail])
    }

    func testRetentionWarningBeforePendingRunDoesNotMoveUploadStallClock() throws {
        let at = ISO8601DateFormatter().string(from: now.addingTimeInterval(-3 * 3600))
        let pending = "{\"attempted_at\":\"\(at)\",\"status\":\"pending\",\"uploaded\":false,\"verified\":false}"
        let result = try parse([upload(6), retention, pending])
        var inputs = BrainBarDesignFixture.inputs(.driveSignedOut)
        inputs.backupHistory = .read(.init(local: [], drive: [], conversations: result.attempts, integrity: []))
        let lane = try XCTUnwrap(NDBackupsPage(store: NDStore(fixture: inputs)).lanes(NDModel.make(inputs)).first { $0.id == "transcripts" })
        XCTAssertEqual(lane.stalledFrom, -6)
        XCTAssertTrue(lane.events.contains { $0.hours == -3 && $0.kind == .fail })
        XCTAssertTrue(lane.events.contains { $0.hours == -6 && $0.kind == .ok })
    }

    func testSuccessfulRetryAdvancesStallClockWhileKeepingCombinedRetryMark() throws {
        let result = try parse([upload(6, ok: false), upload(3)])
        var inputs = BrainBarDesignFixture.inputs(.driveSignedOut)
        inputs.backupHistory = .read(.init(local: [], drive: [], conversations: result.attempts, integrity: []))
        let lane = try XCTUnwrap(NDBackupsPage(store: NDStore(fixture: inputs)).lanes(NDModel.make(inputs)).first { $0.id == "transcripts" })
        XCTAssertEqual(lane.stalledFrom, -3)
        XCTAssertTrue(lane.events.contains { $0.hours == -6 && $0.kind == .retried })
        XCTAssertFalse(lane.events.contains { $0.hours == -3 && $0.kind == .ok })
    }

    func testRetentionErrorUsesFollowingRunDateWithoutDashingLane() throws {
        let result = try parse([upload(6), retention, upload(3)])
        XCTAssertEqual(result.skipped, 0)
        XCTAssertEqual(result.attempts?.map(\.ok), [true, false, true])
        XCTAssertEqual(result.attempts?[1].at, now.addingTimeInterval(-3 * 3600))
        XCTAssertFalse(lanes(result.attempts).contains { $0.unknown })
        XCTAssertTrue(lanes(result.attempts).contains { $0.id == "transcripts" && $0.events.contains { $0.kind == .fail } })
    }

    func testRetentionErrorAtEndOfLogFallsBackToPrecedingRun() throws {
        let result = try parse([upload(3), retention])
        XCTAssertEqual(result.attempts?.map(\.ok), [true, false])
        XCTAssertEqual(result.attempts?.last?.at, now.addingTimeInterval(-3 * 3600))
        XCTAssertEqual(result.skipped, 0)
    }

    func testFutureDatedNeighbourCannotDateRetentionErrorIntoFuture() throws {
        let result = try parse([retention, upload(-3)])
        XCTAssertEqual(result.attempts?.filter { !$0.ok }, [])
        XCTAssertEqual(result.skipped, 0)
    }

    func testRetentionErrorsAreNotSpokenAsUploadAttemptsOrFailures() throws {
        let result = try parse([upload(6), retention, upload(3)])
        let lane = try XCTUnwrap(lanes(result.attempts).first { $0.id == "transcripts" })
        XCTAssertTrue(lane.accessibility.contains("2 attempts this week, 0 failed"), lane.accessibility)
        XCTAssertTrue(lane.accessibility.contains(", 1 retention error"), lane.accessibility)
        XCTAssertTrue(lane.events.contains { $0.kind == .fail })
        let uploads = try parse([upload(6), upload(3)])
        let normal = try XCTUnwrap(lanes(uploads.attempts).first { $0.id == "transcripts" })
        XCTAssertTrue(normal.accessibility.contains("2 attempts this week, 0 failed"))
        XCTAssertFalse(normal.accessibility.contains("retention"))
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
        let at = try XCTUnwrap(progress.checkpointAt)
        let stalledTitle = BrainBarMenuPresenter.SurfaceSay.driveStalledAt(BrainBarBackupProgress.clock(at))
        XCTAssertNotEqual(progress.title, stalledTitle)
        progress.isStalled = true
        XCTAssertEqual(progress.title, stalledTitle, "the Backups page and Drive's line name a stall in the same words")
        XCTAssertEqual(progress.accessibilityValue, try XCTUnwrap(progress.checkpointAt).formatted(date: .abbreviated, time: .shortened))
    }

    /// The Pipeline names the stall on Drive's own line, as a clock time; "Last good" leaves the compact node (its date is
    /// when the copy was made, not when it was verified) and the Backups page says so in full.
    func testStalledFixtureNamesTheStallOnDrivesLineAndKeepsTheFullCaptionOnBackups() throws {
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        let inputs = BrainBarDesignFixture.inputs(.uploadStalled)
        let progress = try XCTUnwrap(inputs.backupProgress)
        XCTAssertTrue(progress.isStalled)
        XCTAssertEqual(progress.title, progress.stationLine(), "one name for the stall on both surfaces")
        XCTAssertNil(BrainBarBackupProgressView(progress: progress).checkpointText, "no time pushed to the right")
        XCTAssertNil(BrainBarBackupProgressView(progress: progress, compact: true).lastGoodText)
        let caption = try XCTUnwrap(BrainBarBackupProgressView(progress: progress).lastGoodText)
        XCTAssertTrue(caption.hasPrefix(Say.backupLastGood + " "), caption)
        XCTAssertTrue(caption.contains(try XCTUnwrap(progress.lastGoodAt).formatted(date: .abbreviated, time: .shortened)))
        let zone = try XCTUnwrap(TimeZone(identifier: "Asia/Jerusalem"))
        let line = progress.stationLine(in: zone)
        XCTAssertEqual(line, Say.driveStalledAt(BrainBarBackupProgress.clock(try XCTUnwrap(progress.checkpointAt), in: zone)))
        let state = NDR1.make(inputs, NDModel.make(inputs)).pipeline
        XCTAssertEqual(NDPipelineChart.drawingState(state, progress: progress).status[.drive]?.text, progress.stationLine())
        // Drive's widest line keeps the Kept-safe rule: 12 pt to the frame on its right.
        let L = NDPipelineLayout.self
        XCTAssertLessThanOrEqual(L.textWidth(line, size: 11, weight: .semibold) / 2, L.keptSafe.maxX - 12 - L.point(.drive).x)
    }
}
