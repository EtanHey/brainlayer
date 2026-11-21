#if DEBUG
import XCTest
@testable import BrainBar

@MainActor final class BrainBarBackupCaptureFixtureTests: XCTestCase {
    typealias Fixture = BrainBarDesignFixture

    private func inputs(_ name: String) throws -> NDInputs {
        let fixture = try XCTUnwrap(Fixture.Name(rawValue: name))
        return Fixture.inputs(fixture)
    }

    func testUploadProgress() throws {
        let i = try inputs("upload-progress")
        let p = try XCTUnwrap(i.backupProgress)
        XCTAssertEqual(p.percentage, 25)
        XCTAssertEqual(p.phase, .upload)
        XCTAssertTrue(NDPipelineChart.showsProgress(NDR1.make(i, NDModel.make(i)).pipeline, progress: p))
    }

    func testUploadNoTotal() throws {
        let p = try XCTUnwrap(inputs("upload-no-total").backupProgress)
        XCTAssertNil(p.total)
        XCTAssertNil(p.percentage)
        XCTAssertNil(p.fraction)
    }

    func testUploadStalled() throws {
        let i = try inputs("upload-stalled")
        let p = try XCTUnwrap(i.backupProgress)
        XCTAssertEqual(p.percentage, 25)
        XCTAssertEqual(i.capture.now.timeIntervalSince(try XCTUnwrap(p.checkpointAt)), 15 * 60)
        XCTAssertFalse(p.animates(after: p, reduceMotion: false))
    }

    func testVerifying() throws {
        let i = try inputs("verifying")
        let p = try XCTUnwrap(i.backupProgress)
        XCTAssertEqual(p.phase, .verification)
        XCTAssertNil(p.percentage)
        XCTAssertEqual(p.title, "Checking the Google Drive copy")
        XCTAssertFalse(ObservabilityPresentation.backupStatus(for: Fixture.backupFacts(.verifying), now: i.capture.now).uploadQueued)
    }

    func testUploadSnapshotRed() throws {
        let i = try inputs("upload-snapshot-red")
        let checks = try XCTUnwrap(i.captureBackupChecks)
        let row = try XCTUnwrap(checks.checks.first { $0.id == "snapshot-upload" })
        XCTAssertEqual(row.tone, .red)
        XCTAssertTrue(row.value.contains("upload still pending"))
        XCTAssertEqual(i.backupProgress?.percentage, 25)
        XCTAssertEqual(BrainBarBackupChecksCard.displayedCheck(row, progress: i.backupProgress), row)
    }

    func testUploadDriveSignedOut() throws {
        let i = try inputs("upload-drive-signed-out")
        let state = NDR1.make(i, NDModel.make(i)).pipeline
        XCTAssertEqual(state.status[.drive]?.severity, .red)
        XCTAssertNotNil(i.backupProgress)
        XCTAssertFalse(NDPipelineChart.showsProgress(state, progress: i.backupProgress))
        XCTAssertEqual(NDPipelineChart.drawingState(state, progress: i.backupProgress).status[.drive], state.status[.drive])
    }

    func testUploadDriveStalled() throws {
        let i = try inputs("upload-drive-stalled")
        let page = NDBackupsPage(store: NDStore(fixture: i))
        let lane = try XCTUnwrap(page.lanes(NDModel.make(i)).first { $0.id == "backup_offsite" })
        XCTAssertTrue(lane.strong)
        XCTAssertNotNil(lane.stalledFrom)
        XCTAssertFalse(lane.events.isEmpty)
        XCTAssertEqual(i.backupProgress?.percentage, 25)
        XCTAssertNotNil(NDBackupsPage.displayedLane(lane, progress: i.backupProgress))
    }
}
#endif
