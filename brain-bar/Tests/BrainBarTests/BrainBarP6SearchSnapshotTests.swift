import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarP6SearchSnapshotTests: XCTestCase {
    func testSearchAndCommandShellStatesRenderToTheCallerProvidedDirectory() throws {
        guard let renderDirectory = ProcessInfo.processInfo.environment["BRAINBAR_RENDER_DIR"],
              !renderDirectory.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw XCTSkip("Set BRAINBAR_RENDER_DIR to render the P6 Search/shell matrix")
        }

        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-p6-search-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        let db = BrainDatabase(path: tempRoot.appendingPathComponent("fixture.db").path)
        XCTAssertTrue(db.isOpen)
        defer {
            db.close()
            try? FileManager.default.removeItem(at: tempRoot)
        }

        let neutral = makeViewModel(db: db) { _, _ in .init(count: 0, formatted: "", results: []) }
        try render(
            CommandShellSnapshot(viewModel: neutral, compact: false),
            named: "search-shell-neutral.png",
            directory: renderDirectory,
            size: NSSize(width: 820, height: 360)
        )

        let focused = makeViewModel(db: db) { query, limit in
            Self.fixedSearchResult(query: query, limit: limit)
        }
        focused.setMode(QuickCapturePanelState.Mode.search)
        focused.inputText = "watcher truth"
        focused.submit()
        focused.panelDidAppear()
        XCTAssertNotNil(focused.selectedResultID)
        XCTAssertGreaterThan(focused.focusRequestCount, 0)
        try render(
            CommandShellSnapshot(viewModel: focused, compact: false),
            named: "search-shell-keyboard-focus.png",
            directory: renderDirectory,
            size: NSSize(width: 820, height: 360)
        )

        let empty = makeViewModel(db: db) { _, _ in .init(count: 0, formatted: "", results: []) }
        empty.setMode(.search)
        empty.inputText = "no matching memory"
        empty.submit()
        XCTAssertTrue(empty.results.isEmpty)
        XCTAssertTrue(empty.feedback.isIdle)
        try render(
            CommandShellSnapshot(viewModel: empty, compact: false),
            named: "search-shell-empty.png",
            directory: renderDirectory,
            size: NSSize(width: 820, height: 360)
        )

        let error = makeViewModel(db: db) { _, _ in throw SearchFixtureError.unavailable }
        error.setMode(.search)
        error.inputText = "temporarily unavailable"
        error.submit()
        XCTAssertEqual(error.feedback, .error("Fixture search unavailable"))
        try render(
            CommandShellSnapshot(viewModel: error, compact: false),
            named: "search-shell-error.png",
            directory: renderDirectory,
            size: NSSize(width: 820, height: 360)
        )

        let compact = makeViewModel(db: db) { _, _ in .init(count: 0, formatted: "", results: []) }
        try render(
            CommandShellSnapshot(viewModel: compact, compact: true),
            named: "search-shell-compact-command-bar.png",
            directory: renderDirectory,
            size: NSSize(width: 520, height: 72)
        )
    }

    private func makeViewModel(
        db: BrainDatabase,
        search: @escaping QuickCaptureSearch
    ) -> QuickCaptureViewModel {
        QuickCaptureViewModel(
            db: db,
            panelState: QuickCapturePanelState(),
            feedbackAutoClearDelay: .seconds(60),
            searchDebounceDelay: .seconds(60),
            search: search
        )
    }

    nonisolated private static func fixedSearchResult(
        query: String,
        limit: Int
    ) -> QuickCaptureController.SearchResult {
        let rows: [[String: Any]] = [
            [
                "chunk_id": "watcher-truth-contract",
                "content": "Watcher liveness uses process plus recent distinct-ingest evidence.",
                "full_content": "Watcher liveness uses process plus recent distinct-ingest evidence.",
                "created_at": "2026-07-19T10:00:00Z",
                "importance": 9,
            ],
            [
                "chunk_id": "freshness-contract",
                "content": "Dashboard snapshots become stale strictly after sixty seconds.",
                "full_content": "Dashboard snapshots become stale strictly after sixty seconds.",
                "created_at": "2026-07-19T09:55:00Z",
                "importance": 8,
            ],
        ]
        return .init(count: min(rows.count, limit), formatted: query, results: Array(rows.prefix(limit)))
    }

    private func render<V: View>(
        _ view: V,
        named name: String,
        directory: String,
        size: NSSize
    ) throws {
        let host = NSHostingView(rootView: view.environment(\.colorScheme, .dark))
        host.frame = NSRect(origin: .zero, size: size)
        host.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.25))
        host.layoutSubtreeIfNeeded()

        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
            throw SearchSnapshotError.bitmapUnavailable(name)
        }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            throw SearchSnapshotError.encodingFailed(name)
        }

        let output = URL(fileURLWithPath: directory, isDirectory: true).appendingPathComponent(name)
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        try png.write(to: output)
        XCTAssertGreaterThan(png.count, 5_000, "Expected a substantive Search/shell PNG for \(name)")
        XCTAssertGreaterThan(distinctSampledColorCount(in: bitmap), 12, "Expected visible Search/shell content for \(name)")
        print("[brainbar-render] wrote \(output.path) (\(png.count) bytes)")
    }

    private func distinctSampledColorCount(in bitmap: NSBitmapImageRep) -> Int {
        guard let data = bitmap.bitmapData else { return 0 }
        let bytesPerPixel = max(bitmap.bitsPerPixel / 8, 1)
        let baseStride = max(bitmap.bytesPerRow / 32, bytesPerPixel)
        let sampleStride = baseStride - (baseStride % bytesPerPixel)
        var colors = Set<String>()
        for y in stride(from: 0, to: bitmap.pixelsHigh, by: 24) {
            let rowStart = y * bitmap.bytesPerRow
            for x in stride(from: 0, to: bitmap.bytesPerRow, by: sampleStride) {
                let offset = rowStart + x
                guard offset + 2 < bitmap.bytesPerRow * bitmap.pixelsHigh else { continue }
                colors.insert("\(data[offset])-\(data[offset + 1])-\(data[offset + 2])")
            }
        }
        return colors.count
    }
}

private struct CommandShellSnapshot: View {
    @ObservedObject var viewModel: QuickCaptureViewModel
    let compact: Bool

    var body: some View {
        ZStack(alignment: .top) {
            Color.brainBarBackgroundBase
            VStack(spacing: 0) {
                BrainBarCommandBar(viewModel: viewModel)
                if !compact {
                    BrainBarCommandBarResultsOverlay(viewModel: viewModel, isOnActiveTab: true)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding(compact ? 8 : 18)
        }
        .transaction { $0.disablesAnimations = true }
    }
}

private enum SearchFixtureError: LocalizedError {
    case unavailable

    var errorDescription: String? { "Fixture search unavailable" }
}

private enum SearchSnapshotError: Error {
    case bitmapUnavailable(String)
    case encodingFailed(String)
}
