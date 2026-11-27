import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// The Pipeline's Google Drive node during a backup upload: in every upload state, with Local copy at its longest
/// realistic status, the progress bar and Drive's line clear every other station's words, the Kept-safe border, every
/// connector and arrowhead, and every ring and frame. A UI change is checked in every state it touches, not only at rest.
@MainActor final class BrainBarDriveUploadStateTests: XCTestCase {
    typealias L = NDPipelineLayout
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    static let now = BrainBarDesignFixture.now

    struct UploadCase {
        var name: String
        var progress: BrainBarBackupProgress?
        /// Google Drive carries its own fault words (red or amber) while the upload runs.
        var driveFault: NDPipelineState.Status? = nil
    }

    /// A receipt as the producer writes it, read through the same presenter the app uses.
    static func progress(confirmed: Int64 = 45, total: Int64? = 100, checkpointAge: TimeInterval? = 30,
                         startedAgo: TimeInterval = 3600, verifying: Bool = false) throws -> BrainBarBackupProgress {
        let at = { (age: TimeInterval) in String(now.addingTimeInterval(-age).timeIntervalSinceReferenceDate) }
        let data = Data("""
        {"state":"measured","reason":"","inputs":[],"threshold_hours":36,
         "db_snapshot":{"last_at":\(at(26 * 3600)),"destination":"old.db.gz","verified":true},
         "local_snapshot":{"last_at":\(at(startedAgo + 60)),"destination":"new.db.gz",
           "local_verified":true,"uploaded":\(verifying),"upload_state":"\(verifying ? "pending" : "in_progress")",
           "upload_process_live":true,"upload_process_started_at":\(at(startedAgo)),
           "bytes_confirmed":\(confirmed),"bytes_total":\(total.map(String.init) ?? "null"),"verification_pending":\(verifying),
           "bytes_confirmed_at":\(checkpointAge.map(at) ?? "null")}}
        """.utf8)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let backups = try decoder.decode(ObservabilityDocument.Backups.self, from: data)
        return try XCTUnwrap(BrainBarBackupProgress.read(backups, now: now))
    }

    /// The states the brief names: none, 0 / 45 / 99 %, no total, stalled, verifying, Drive in error (red and amber,
    /// at its widest designed words), and an upload that started yesterday with a fresh checkpoint.
    static func cases() throws -> [UploadCase] {
        [
            UploadCase(name: "none", progress: nil),
            UploadCase(name: "uploading-0", progress: try progress(confirmed: 0)),
            UploadCase(name: "uploading-45", progress: try progress(confirmed: 45)),
            UploadCase(name: "uploading-99", progress: try progress(confirmed: 99)),
            UploadCase(name: "uploading-no-total", progress: try progress(total: nil)),
            UploadCase(name: "stalled", progress: try progress(checkpointAge: 21 * 60)),
            UploadCase(name: "verifying", progress: try progress(verifying: true)),
            UploadCase(name: "drive-error-red", progress: try progress(),
                       driveFault: .init(text: Say.signedOutValue, severity: .red)),
            UploadCase(name: "drive-error-amber", progress: try progress(),
                       driveFault: .init(text: Say.lastUpload(NDFormat.ageText(27 * 3600)), severity: .orange)),
            UploadCase(name: "many-hours", progress: try progress(confirmed: 61, checkpointAge: 40, startedAgo: 20 * 3600)),
        ]
    }

    /// The calm fixture's statuses everywhere, Local copy at its longest realistic status in Semibold (a late copy).
    static func state(_ c: UploadCase) -> NDPipelineState {
        let inputs = BrainBarDesignFixture.inputs(.calm)
        var s = NDR1.make(inputs, NDModel.make(inputs)).pipeline
        s.notices.insert(.mac)
        s.status[.mac] = .init(text: Say.keptNewest(3, NDFormat.ageText(30 * 3600)), severity: .orange)
        if let fault = c.driveFault {
            if fault.severity == .red { s.faults.insert(.drive) } else { s.notices.insert(.drive) }
            s.status[.drive] = fault
        }
        return s
    }

    /// Everything the progress row may not touch, each with the clearance it is owed: other stations' words 16 pt
    /// across (the Kept-safe rule between text blocks) and 3 pt up and down; lines, borders, rings and frames 3 pt
    /// (the chart's `edgeGap`).
    static func obstacles(_ drawn: NDPipelineState) -> [(String, CGRect)] {
        let gap = L.edgeGap
        var out: [(String, CGRect)] = []
        for s in NDStation.allCases where s != .drive {
            for r in L.labelRects(s, drawn) { out.append(("\(s.rawValue) words", r.insetBy(dx: -16, dy: -gap))) }
        }
        out.append(("drive name", L.labelRects(.drive, drawn)[0].insetBy(dx: -gap, dy: -gap)))
        for (words, r) in L.laneLabelRects() { out.append(("caption \(words)", r.insetBy(dx: -gap, dy: -gap))) }
        for s in NDStation.allCases where L.stations[s] != nil {
            let p = L.point(s), r = L.ringRadius + gap
            out.append(("\(s.rawValue) ring", CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)))
        }
        out.append(("memory box", L.hub.insetBy(dx: -gap, dy: -gap)))
        out.append(("agents frame", L.agentsFrame.insetBy(dx: -gap, dy: -gap)))
        let k = L.keptSafe
        for (edge, r) in [("top", CGRect(x: k.minX, y: k.minY - 0.5, width: k.width, height: 1)),
                          ("bottom", CGRect(x: k.minX, y: k.maxY - 0.5, width: k.width, height: 1)),
                          ("left", CGRect(x: k.minX - 0.5, y: k.minY, width: 1, height: k.height)),
                          ("right", CGRect(x: k.maxX - 0.5, y: k.minY, width: 1, height: k.height))] {
            out.append(("kept-safe \(edge) border", r.insetBy(dx: -gap, dy: -gap)))
        }
        for c in L.connections {
            let pts = c.line.points
            guard pts.count > 1 else { continue }
            for i in 1..<pts.count {
                let a = pts[i - 1], b = pts[i]
                let seg = CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(b.x - a.x), height: abs(b.y - a.y))
                out.append(("\(c.pipe) line", seg.insetBy(dx: -(0.75 + gap), dy: -(0.75 + gap))))
            }
            // An arrowhead reaches 4 pt either side of its line and 6.5 pt back from its tip.
            var heads = [pts[pts.count - 1]]
            if c.bothWays { heads.append(pts[0]) }
            for tip in heads {
                let r = 6.5 + gap
                out.append(("\(c.pipe) arrowhead", CGRect(x: tip.x - r, y: tip.y - r, width: 2 * r, height: 2 * r)))
            }
        }
        return out
    }

    /// The progress row as the chart draws it: the frame it places the progress view in, and Drive's own line.
    static func progressElements(_ state: NDPipelineState, _ progress: BrainBarBackupProgress) -> [(String, CGRect)] {
        progressElements(state, drawn: NDPipelineChart.drawingState(state, progress: progress))
    }

    static func progressElements(_ state: NDPipelineState, drawn: NDPipelineState) -> [(String, CGRect)] {
        var out = [("progress view", NDPipelineChart.progressFrame(state))]
        let drive = L.labelRects(.drive, drawn)
        if drive.count > 1 { out.append(("drive line \"\(drawn.status[.drive]?.text ?? "")\"", drive[drive.count - 1])) }
        return out
    }

    func testEveryUploadStateClearsTheChart() throws {
        var checks: [(String, NDPipelineState, NDPipelineState)] = []
        for c in try Self.cases() {
            let state = Self.state(c)
            guard let progress = c.progress else {
                XCTAssertFalse(NDPipelineChart.showsProgress(state, progress: nil), c.name)
                continue
            }
            XCTAssertTrue(NDPipelineChart.showsProgress(state, progress: progress), c.name)
            checks.append((c.name, state, NDPipelineChart.drawingState(state, progress: progress)))
        }
        // The stall's line at its widest clock, in the person's own 12- or 24-hour setting ("12:58 PM", "오후 12:58").
        let stalled = try XCTUnwrap(Self.cases().first { $0.name == "stalled" })
        let utc = try XCTUnwrap(TimeZone(identifier: "UTC"))
        let widest = try XCTUnwrap(Calendar(identifier: .gregorian).date(from: DateComponents(timeZone: utc, year: 2026, month: 11,
                                                                                              day: 26, hour: 12, minute: 58)))
        for id in ["en_US", "ko_KR", "en_GB"] {
            var drawn = Self.state(stalled)
            drawn.status[.drive] = .init(text: Say.driveStalledAt(BrainBarBackupProgress.clock(widest, in: utc, locale: Locale(identifier: id))))
            checks.append(("stalled-\(id)", Self.state(stalled), drawn))
        }
        for (name, state, drawn) in checks {
            for (what, rect) in Self.progressElements(state, drawn: drawn) {
                // Drive's line sits directly under Drive's name, as every station's status does.
                let own = what.hasPrefix("drive line") ? "drive name" : ""
                for (obstacle, box) in Self.obstacles(drawn) where obstacle != own && rect.intersects(box) {
                    XCTFail("\(name): \(what) \(rect) overlaps \(obstacle) \(box)")
                }
                XCTAssertTrue(L.keptSafe.contains(rect), "\(name): \(what) \(rect) stays inside Kept safe \(L.keptSafe)")
            }
        }
    }

    /// The view the chart places is no taller than the frame it is placed in: nothing hangs below it.
    /// (Checked with Drive calm: the compact view does not vary with Drive's severity.)
    func testCompactProgressViewFitsItsFrame() throws {
        for c in try Self.cases() where c.driveFault == nil {
            guard let progress = c.progress else { continue }
            let frame = NDPipelineChart.progressFrame(Self.state(c))
            let view = BrainBarBackupProgressView(progress: progress, compact: true).frame(width: frame.width)
            let fitted = NSHostingController(rootView: view)
                .sizeThatFits(in: CGSize(width: frame.width, height: .greatestFiniteMagnitude))
            XCTAssertLessThanOrEqual(fitted.height, frame.height + 0.5, "\(c.name): the compact view measures \(fitted)")
        }
    }

    /// Drive's line says what the upload is doing in every state; a fault keeps its own words; on the Backups page the
    /// checkpoint reads "updated 06:26", never a bare "06:26", and the upload's start is said in full.
    func testDrivesLineAndPanelSayWhatTheyMean() throws {
        let zone = try XCTUnwrap(TimeZone(identifier: "Asia/Jerusalem"))
        let expected = ["uploading-0": Say.driveUploadingPercent(0), "uploading-45": Say.driveUploadingPercent(45),
                        "uploading-99": Say.driveUploadingPercent(99), "uploading-no-total": Say.backupUploadingShort,
                        "verifying": Say.backupVerifyingShort, "many-hours": Say.driveUploadingPercent(61)]
        for c in try Self.cases() {
            guard let progress = c.progress else { continue }
            let drawn = NDPipelineChart.drawingState(Self.state(c), progress: progress)
            if let fault = c.driveFault {
                XCTAssertEqual(drawn.status[.drive], fault, c.name)
            } else if c.name == "stalled" {
                let at = try XCTUnwrap(progress.checkpointAt)
                XCTAssertEqual(progress.stationLine(in: zone), Say.driveStalledAt(BrainBarBackupProgress.clock(at, in: zone)))
            } else {
                XCTAssertEqual(progress.stationLine(in: zone), expected[c.name], c.name)
                XCTAssertEqual(drawn.status[.drive]?.text, expected[c.name], c.name)
            }
        }
        // An upload that started yesterday: the Backups page says when it started and when bytes last advanced.
        let long = try XCTUnwrap(Self.cases().first { $0.name == "many-hours" }?.progress)
        let started = try XCTUnwrap(long.startedAt)
        XCTAssertEqual(Self.now.timeIntervalSince(started), 20 * 3600)
        XCTAssertEqual(BrainBarBackupProgressView(progress: long).startedText,
                       Say.backupUploadStarted + " " + started.formatted(date: .abbreviated, time: .shortened))
        XCTAssertNil(BrainBarBackupProgressView(progress: long, compact: true).startedText)
        XCTAssertEqual(BrainBarBackupProgressView(progress: long).checkpointText,
                       "61% · " + Say.backupUpdatedAt(BrainBarBackupProgress.clock(try XCTUnwrap(long.checkpointAt))))
        XCTAssertNil(BrainBarBackupProgressView(progress: long, compact: true).checkpointText)
    }

    /// Review 1886 B2: no dashed bar. Stalled keeps its last amount in grey, a check fills the track in a lighter grey, an
    /// unknown amount is the track alone, a Drive in error greys the amount; none of them moves.
    func testBarStatesAreStillAndDistinct() throws {
        func fill(_ name: String, muted: Bool = false) throws -> BrainBarBackupProgress.BarFill {
            try XCTUnwrap(Self.cases().first { $0.name == name }?.progress).barFill(muted: muted)
        }
        XCTAssertEqual(try fill("uploading-45"), .accent(0.45))
        XCTAssertEqual(try fill("stalled"), .grey(0.45), "the stall keeps the amount it reached")
        XCTAssertEqual(try fill("verifying"), .checking)
        XCTAssertEqual(try fill("uploading-no-total"), .unknown)
        XCTAssertEqual(try fill("uploading-45", muted: true), .grey(0.45), "Drive in error: grey, never the accent")
        let still = [try fill("stalled"), try fill("verifying"), try fill("uploading-no-total")]
        XCTAssertEqual(Set(still.map { "\($0)" }).count, 3, "three different bars")
        for name in ["stalled", "verifying", "uploading-no-total"] {
            let p = try XCTUnwrap(Self.cases().first { $0.name == name }?.progress)
            var next = p
            next.confirmed = (p.confirmed ?? 0) + 10
            XCTAssertFalse(next.animates(after: p, reduceMotion: false), "\(name) is a still bar")
        }
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/BrainBarBackupProgress.swift"), encoding: .utf8)
        XCTAssertFalse(source.contains("dash:"), "no dashed bar in any state")
    }

    /// The window has one width: its minimum is its default, so one render per state covers both.
    func testWindowMinimumWidthIsItsDefaultWidth() {
        XCTAssertEqual(NDGrid.windowMinimum.width, NDGrid.windowWidth)
    }

    /// `DRIVE_UPLOAD_RENDER_DIR=<absolute dir>`: every upload state in dark and light, offscreen, at the window's width;
    /// Drive's panel for a running upload; the Backups page's progress.
    func testRenderRequestedUploadStates() throws {
        guard let path = ProcessInfo.processInfo.environment["DRIVE_UPLOAD_RENDER_DIR"] else { return }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        typealias F = BrainBarDesignFixture
        let cases = try Self.cases()
        func progress(_ name: String) -> BrainBarBackupProgress? { cases.first { $0.name == name }?.progress }
        // Each state on the inputs whose stations it belongs with: calm Drive on the upload fixture, Drive in error on
        // the signed-out and the 72 h late fixtures, and Local copy at its longest status on the late-copy fixture.
        var states: [(String, F.Name, BrainBarBackupProgress?)] = cases.filter { $0.driveFault == nil }
            .map { ($0.name, $0.name == "none" ? .calm : .uploadProgress, $0.progress) }
        states += [("drive-error-signed-out", .uploadDriveSignedOut, progress("uploading-45")),
                   ("drive-error-late", .uploadSnapshotRed, progress("uploading-45")),
                   ("local-copy-longest", .backupLate, progress("uploading-45"))]
        for (suffix, appearance) in NDRenderHarness.appearances {
            for (name, fixture, value) in states {
                var inputs = F.inputs(fixture)
                inputs.backupProgress = value
                let store = NDStore(fixture: inputs, page: .overview)
                _ = try NDRenderHarness.window(store, name: "overview-\(name)-\(suffix)", appearance: appearance, in: directory)
            }
            for name in ["uploading-45", "many-hours"] {
                var inputs = F.inputs(.uploadProgress)
                inputs.backupProgress = progress(name)
                let store = NDStore(fixture: inputs, page: .overview)
                _ = try NDRenderHarness.hover(store, station: .drive, name: "hover-drive-\(name)-\(suffix)", appearance: appearance, in: directory)
            }
            for name in ["uploading-45", "stalled"] {
                var inputs = F.inputs(.uploadProgress)
                inputs.backupProgress = progress(name)
                let store = NDStore(fixture: inputs, page: .backups)
                _ = try NDRenderHarness.window(store, name: "backups-\(name)-\(suffix)", appearance: appearance, in: directory)
            }
        }
    }
}
