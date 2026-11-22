#if DEBUG
import AppKit
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
        XCTAssertNil(p.percentage)
        XCTAssertTrue(p.isStalled)
        XCTAssertEqual(i.capture.now.timeIntervalSince(try XCTUnwrap(p.checkpointAt)), 21 * 60)
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
            typealias L = NDPipelineLayout
            let frame = NDPipelineChart.progressFrame(state)
            let drive = L.point(.drive)
            let wordsBottom = try XCTUnwrap(L.labelRects(.drive, state).map(\.maxY).max())
            XCTAssertEqual(frame.midX, drive.x)
            XCTAssertEqual(frame.size, CGSize(width: 96, height: 3))
            XCTAssertEqual(frame.minY - wordsBottom, 5, accuracy: 0.01, name)
            XCTAssertTrue(L.keptSafe.contains(frame), name)
            for station in NDStation.allCases {
                for rect in L.labelRects(station, state) { XCTAssertFalse(frame.intersects(rect), "\(name): \(station) text") }
                XCTAssertFalse(frame.intersects(L.circle(station)), "\(name): \(station) glyph")
            }
            for (caption, rect) in L.laneLabelRects() {
                XCTAssertFalse(frame.intersects(rect.insetBy(dx: -4, dy: -4)), "\(name): \(caption)")
            }
            // Use the native font sizes and the same baseline-to-centre placement as Canvas.put, independently
            // of labelRects' allocated boxes. This catches text extending into the bar despite a stale layout box.
            let nameSize = NSAttributedString(string: L.name(.drive), attributes: [.font: NSFont.systemFont(ofSize: 13)]).size()
            let status = try XCTUnwrap(state.status[.drive])
            let statusSize = NSAttributedString(string: status.text,
                attributes: [.font: NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .semibold)]).size()
            let nativeBottom = max(drive.y + 29 - 13 * 0.36 + nameSize.height / 2,
                                   L.statusTop(.drive) + 10 - 11 * 0.36 + statusSize.height / 2)
            XCTAssertGreaterThanOrEqual(frame.minY - nativeBottom, 4, name)
            XCTAssertFalse(frame.intersects(L.hub), "\(name): Memory text")
            XCTAssertFalse(frame.intersects(L.agentsFrame), "\(name): agent labels")
            XCTAssertFalse(frame.intersects(L.pipelineHeading), "\(name): Pipeline heading")
            for rect in L.footerTexts { XCTAssertFalse(frame.intersects(rect), "\(name): footer text") }
            for connection in L.connections {
                for (a, b) in zip(connection.line.points, connection.line.points.dropFirst()) {
                    let stroke = CGRect(x: min(a.x, b.x) - 1.25, y: min(a.y, b.y) - 1.25,
                        width: abs(a.x - b.x) + 2.5, height: abs(a.y - b.y) + 2.5)
                    XCTAssertFalse(frame.intersects(stroke), "\(name): \(connection.pipe) stroke")
                }
            }
            print("backup free space \(name): bar=\(frame), text bottom=\(wordsBottom), native bottom=\(nativeBottom)")
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
