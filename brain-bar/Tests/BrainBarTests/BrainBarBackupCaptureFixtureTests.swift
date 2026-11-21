#if DEBUG
import XCTest
import SwiftUI
@testable import BrainBar

@MainActor final class BrainBarBackupCaptureFixtureTests: XCTestCase {
    typealias Fixture = BrainBarDesignFixture

    private func inputs(_ name: String) throws -> NDInputs {
        let fixture = try XCTUnwrap(Fixture.Name(rawValue: name))
        return Fixture.inputs(fixture)
    }

    func testUploadingFaultHeadlinesFitOneOverviewRow() throws {
        for name in ["upload-snapshot-red", "upload-drive-stalled"] {
            let i = try inputs(name)
            let r1 = NDR1.make(i, NDModel.make(i))
            XCTAssertEqual(r1.problemRows.count, r1.items.filter { $0.severity == .red }.count)
            for row in r1.problemRows {
                let width = 14 + 14 + 10 + NDPipelineLayout.textWidth(row.words, size: 13, weight: .semibold) + 12 + 110 + 14
                XCTAssertLessThanOrEqual(width, 852, "\(name): \(row.words)")
                print("backup row \(name): \(width) pt")
            }
            let drive = try XCTUnwrap(r1.stationItems[.drive])
            XCTAssertTrue(drive.title.contains("an upload is running"))
            XCTAssertTrue(drive.title.contains("Nov 12 at 07:59"))
            let store = NDStore(fixture: i, page: .overview)
            let model = NDStationPopover.model(.drive, item: drive, i, store.model, r1.pipeline)
            let chart = NDPipelineChart(state: r1.pipeline, memories: store.model.memories, items: r1.stationItems)
            let panel = chart.panel(.drive, content: AnyView(NDStationPopover(store: store, station: .drive)))
            print("backup panel \(name): top=\(panel.outer.minY) title=\(model.title) lines=\(model.lines)")
            XCTAssertGreaterThanOrEqual(panel.outer.minY, NDPipelineLayout.activityLabelsBottom)
        }
    }

    func testUploadProgress() throws {
        let i = try inputs("upload-progress")
        let p = try XCTUnwrap(i.backupProgress)
        XCTAssertEqual(p.percentage, 25)
        XCTAssertEqual(p.phase, .upload)
        XCTAssertTrue(NDPipelineChart.showsProgress(NDR1.make(i, NDModel.make(i)).pipeline, progress: p))
    }

    func testUploadingTitleDatesTheDatabaseCopyWhenSessionsAreNewer() throws {
        var i = try inputs("upload-drive-stalled")
        let verdict = try XCTUnwrap(i.capture.verdict)
        let checks = verdict.checks.map { check in
            check.name == "transcripts" ? BrainBarHealthVerdict.Check(name: check.name, state: .green, ageS: 3600, note: "") : check
        }
        i.capture.verdict = BrainBarHealthVerdict(dot: verdict.dot, checks: checks, notes: verdict.notes, evaluatedAt: verdict.evaluatedAt)
        let title = try XCTUnwrap(NDR1.make(i, NDModel.make(i)).stationItems[.drive]?.title)
        XCTAssertTrue(title.contains("database backup"), title)
        XCTAssertTrue(title.contains("Nov 12 at 07:59"), title)
        XCTAssertTrue(title.contains("an upload is running"), title)
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
        XCTAssertTrue(NDPipelineChart.showsProgress(state, progress: i.backupProgress))
        XCTAssertEqual(NDPipelineChart.drawingState(state, progress: i.backupProgress).status[.drive], state.status[.drive])
    }

    func testUploadDriveStalled() throws {
        let i = try inputs("upload-drive-stalled")
        let page = NDBackupsPage(store: NDStore(fixture: i))
        let lane = try XCTUnwrap(page.lanes(NDModel.make(i)).first { $0.id == "backup_offsite" })
        XCTAssertTrue(lane.strong)
        XCTAssertNotNil(lane.stalledFrom)
        XCTAssertFalse(lane.events.isEmpty)
        XCTAssertNil(i.backupProgress?.percentage, "Gap 22: a stalled checkpoint older than the display limit carries no number")
        XCTAssertEqual(i.capture.driveAuthModel, .valid)
        XCTAssertEqual(i.capture.verdict?.checks.first { $0.name == "drive_auth" }?.state, .green)
        XCTAssertEqual(i.backupProgress?.isStalled, true)
        XCTAssertNotNil(NDBackupsPage.displayedLane(lane, progress: i.backupProgress))
    }

    func testOverviewFaultFixturesKeepWordsAndAddBarInMeasuredSpace() throws {
        for name in ["upload-snapshot-red", "upload-drive-signed-out", "upload-drive-stalled"] {
            let i = try inputs(name)
            let state = NDR1.make(i, NDModel.make(i)).pipeline
            XCTAssertNotNil(state.status[.drive]?.severity, name)
            XCTAssertTrue(NDPipelineChart.showsProgress(state, progress: i.backupProgress), name)
            XCTAssertEqual(NDPipelineChart.drawingState(state, progress: i.backupProgress).status[.drive], state.status[.drive])
            XCTAssertEqual(NDPipelineChart.progressFrame(state), CGRect(x: 688, y: 115, width: 96, height: 3))
        }
    }

    func testOldByteCheckpointDoesNotCreateANewRedWhenBackupChecksAreGreen() throws {
        var i = try inputs("upload-progress")
        i.backupProgress?.isStalled = true
        let page = NDBackupsPage(store: NDStore(fixture: i))
        let lane = try XCTUnwrap(page.lanes(NDModel.make(i)).first { $0.id == "backup_offsite" })
        XCTAssertFalse(lane.strong)
        XCTAssertNil(lane.stalledFrom)
    }
}
#endif
