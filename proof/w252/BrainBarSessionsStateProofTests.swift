import AppKit
import XCTest
@testable import BrainBar

/// Proof-only harness, published beside its PNG baselines rather than changing the application branch.
@MainActor final class BrainBarSessionsStateProofTests: XCTestCase {
    func testSessionsHistoryStateTable() throws {
        guard let path = ProcessInfo.processInfo.environment["W252_STATE_PROOF"] else {
            return XCTFail("W252_STATE_PROOF must name the offscreen output directory")
        }
        NSApplication.shared.setActivationPolicy(.prohibited)
        NDMarkView.captureClock = Date(timeIntervalSinceReferenceDate: 0.3)
        defer { NDMarkView.captureClock = nil }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let now = BrainBarDesignFixture.now
        let legacy = #"{"already_covered_files":4,"discovered_file_count":4,"message":"No new files","skipped_active_count":0,"status":"no-op","uploaded":false,"verified":false}"#
        func upload(_ hours: Double, ok: Bool = true) -> String {
            let at = ISO8601DateFormatter().string(from: now.addingTimeInterval(-hours * 3600))
            return "{\"attempted_at\":\"\(at)\",\"uploaded\":\(ok),\"verified\":\(ok)}"
        }
        let head = [legacy, legacy, legacy, upload(400), upload(250)]
        let states: [(name: String, rows: [String], unknown: Bool, ok: Int, failed: Int)] = [
            ("legacy-recent", head + [upload(6), upload(3)], false, 2, 0),
            ("legacy-empty", head, false, 0, 0),
            ("recent-undated", [upload(6), legacy, upload(3)], true, 0, 0),
            ("recent-failed", head + [upload(6), upload(3, ok: false)], false, 1, 1),
        ]
        var manifest: [[String: Any]] = []
        var images: [String: [Data]] = [:]
        for state in states {
            let result = try XCTUnwrap(NDBackupHistoryReader.parseConversations(
                .init(lines: state.rows.map { Substring($0) }, unreadable: false), now: now))
            var inputs = BrainBarDesignFixture.inputs(.calm)
            guard case var .read(history) = inputs.backupHistory else { return XCTFail("calm history must be read") }
            history.conversations = result.attempts
            history.skippedConversationLines = result.skipped
            inputs.backupHistory = .read(history)
            let store = NDStore(fixture: inputs, page: .backups)
            let lanes = NDBackupsPage(store: store).lanes(store.model)
            let lane = try XCTUnwrap(lanes.first { $0.id == "transcripts" })
            let legend = NDBackupsPage.legend(lanes)
            XCTAssertEqual(lane.unknown, state.unknown, state.name)
            XCTAssertEqual(lane.events.filter { $0.kind == .ok }.count, state.ok, state.name)
            XCTAssertEqual(lane.events.filter { $0.kind == .fail }.count, state.failed, state.name)
            XCTAssertEqual(legend.contains(BrainBarMenuPresenter.SurfaceSay.unreadRecord), state.unknown, state.name)
            XCTAssertEqual(legend.contains(BrainBarMenuPresenter.SurfaceSay.failed), state.failed > 0, state.name)
            for (suffix, appearance) in NDRenderHarness.appearances {
                let name = "sessions-\(state.name)-\(suffix)"
                _ = try NDRenderHarness.window(store, name: name, appearance: appearance, in: directory)
                let png = try Data(contentsOf: directory.appendingPathComponent(name + ".png"))
                XCTAssertGreaterThan(png.count, 2_000)
                let bitmap = try XCTUnwrap(NSBitmapImageRep(data: png))
                XCTAssertEqual(bitmap.pixelsWide, Int(NDGrid.windowWidth * 2))
                images[suffix, default: []].append(png)
                if let goldenPath = ProcessInfo.processInfo.environment["W252_STATE_BASELINE"] {
                    let golden = URL(fileURLWithPath: goldenPath).appendingPathComponent(name + ".png")
                    XCTAssertTrue(png == (try Data(contentsOf: golden)), "\(name): drawn page differs from published baseline")
                }
                manifest.append([
                    "state": state.name, "appearance": suffix, "file": name + ".png",
                    "width_pt": Double(NDGrid.windowWidth), "width_px": bitmap.pixelsWide, "height_px": bitmap.pixelsHigh,
                    "legend": legend, "unknown": lane.unknown, "newest": lane.number.text,
                    "accessibility": lane.accessibility, "skipped": result.skipped,
                    "events": lane.events.map { event -> [String: Any] in
                        ["hours": event.hours, "kind": String(describing: event.kind)]
                    },
                ])
            }
        }
        for (appearance, drawn) in images {
            for a in drawn.indices {
                for b in drawn.indices where b > a {
                    XCTAssertNotEqual(drawn[a], drawn[b], "\(appearance): distinct history states must draw differently")
                }
            }
        }
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("manifest.json"))
    }
}
