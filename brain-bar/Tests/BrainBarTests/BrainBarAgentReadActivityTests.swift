import XCTest
import SwiftUI
import Vision
@testable import BrainBar

final class BrainBarAgentReadActivityTests: XCTestCase {
    @MainActor
    func testReadColourIsDistinctAndVisibleOnBothCardBackgrounds() {
        func luminance(_ color: NSColor) -> Double {
            let values = [color.redComponent, color.greenComponent, color.blueComponent].map { value in
                let v = Double(value)
                return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
            }
            return values[0] * 0.2126 + values[1] * 0.7152 + values[2] * 0.0722
        }
        for (_, appearance) in NDRenderHarness.appearances {
            NSAppearance(named: appearance)!.performAsCurrentDrawingAppearance {
                let color = NDColor.agentRead.usingColorSpace(.sRGB)!
                for other in [NDColor.accent, NDColor.saved, NDColor.read] {
                    XCTAssertNotEqual(color, other.usingColorSpace(.sRGB)!)
                }
                let a = luminance(color), b = luminance(NDColor.card.usingColorSpace(.sRGB)!)
                XCTAssertGreaterThanOrEqual((max(a, b) + 0.05) / (min(a, b) + 0.05), 3)
            }
        }
    }

    @MainActor
    func testInventedCallTrafficIncludesQuietAndIndependentBusyStretches() {
        for range in NDRange.allCases {
            let wave = BrainBarDesignFixture.inputs(.calm).heroWaves[range]!
            let calls = wave.agentReads!
            XCTAssertTrue(calls.prefix(7).allSatisfy { $0 == 0 })
            XCTAssertGreaterThan(calls.max() ?? 0, 0)
            XCTAssertTrue(zip(calls, wave.saves).contains { pair in pair.0 > pair.1 })
            XCTAssertTrue(zip(calls, wave.saves).contains { pair in pair.0 < pair.1 })
            XCTAssertNotEqual(calls, wave.saves.map { $0 * 2 })
        }
    }

    @MainActor
    func testReadLineZeroIsOnAxisAndPeakUsesTheSavedHalf() {
        let canvas = NDActivitySection.canvas
        let axis = NDWaveform.baseline(canvas, nowTick: true).midY
        XCTAssertEqual(NDActivitySection.agentReadY(count: 0, peak: 12, canvas: canvas), axis)
        XCTAssertEqual(NDActivitySection.agentReadY(count: 12, peak: 12, canvas: canvas),
                       axis - NDWaveform.halves(.hero).up)
    }

    @MainActor
    func testAllSeriesCaptionsUseTheSelectedBucketSpan() {
        for range in NDRange.allCases {
            let wave = BrainBarDesignFixture.inputs(.calm).heroWaves[range]!
            let span = range.minutes / wave.count
            XCTAssertEqual(NDActivitySection.agentReadScale(wave, minutes: range.minutes),
                           BrainBarMenuPresenter.SurfaceSay.upTo(String(wave.agentReads!.max()!), minutes: span))
        }
    }

    func testBurstCallsCoalesceIntoOneAtomicSnapshot() throws {
        final class Writes: @unchecked Sendable {
            let lock = NSLock()
            var values: [Data] = []
            func append(_ data: Data) { lock.lock(); values.append(data); lock.unlock() }
        }
        let writes = Writes()
        let receipts = BrainBarOperationReceipts(url: brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString),
                                                persist: { data, _ in writes.append(data) })
        for _ in 0..<200 { receipts.recordAgentRead(toolName: "brain_recall") }
        receipts.waitForWritesForTesting()
        XCTAssertEqual(writes.values.count, 1)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: XCTUnwrap(writes.values.last)) as? [String: Any])
        XCTAssertEqual((json["agentReads"] as? [String: Int])?.values.reduce(0, +), 200)
    }

    func testPreCountingReceiptsSurviveTheFirstCountedWrite() throws {
        struct Legacy: Encodable {
            let search: BrainBarOperationReceipt
            let ingest: BrainBarOperationReceipt
        }
        let old = Legacy(search: .init(kind: .search, durationMillis: 142, count: 10, recordedAt: Date(timeIntervalSince1970: 1000)),
                         ingest: .init(kind: .ingest, durationMillis: 85, count: 1, recordedAt: Date(timeIntervalSince1970: 2000)))
        let url = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        try JSONEncoder().encode(old).write(to: url)
        let receipts = BrainBarOperationReceipts(url: url)
        XCTAssertEqual(receipts.search, old.search)
        XCTAssertEqual(receipts.ingest, old.ingest)
        XCTAssertNil(receipts.agentReadBuckets(minutes: 60, count: 60, end: Date()))
        receipts.recordAgentRead(toolName: "brain_recall")
        receipts.waitForWritesForTesting()
        let restarted = BrainBarOperationReceipts(url: url)
        XCTAssertEqual(restarted.search, old.search)
        XCTAssertEqual(restarted.ingest, old.ingest)
        XCTAssertEqual(restarted.agentReadBuckets(minutes: 60, count: 60, end: Date())?.reduce(0, +), 1)
    }

    func testRouterCountsSearchAndRecallCallsNotStoresOrResults() throws {
        let url = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        let receipts = BrainBarOperationReceipts(url: url)
        let router = MCPRouter(profile: "full", receiptStore: receipts)
        // Invalid calls still count as calls, without requiring a database or helper.
        for name in ["brain_search", "brain_recall", "brain_store", "brain_search"] {
            _ = router.handle(["jsonrpc": "2.0", "id": 1, "method": "tools/call",
                               "params": ["name": name, "arguments": [:] as [String: Any]]])
        }
        receipts.waitForWritesForTesting()
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let counts = try XCTUnwrap(json["agentReads"] as? [String: Int])
        XCTAssertEqual(counts.values.reduce(0, +), 3)
        XCTAssertFalse(String(decoding: try Data(contentsOf: url), as: UTF8.self).contains("query"))
    }

    func testBucketsSurviveRestartPruneAndConcurrentCalls() throws {
        let url = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        let receipts = BrainBarOperationReceipts(url: url)
        let now = Date(timeIntervalSince1970: 6_000_000)
        receipts.recordAgentRead(toolName: "brain_search", at: now.addingTimeInterval(-49 * 3600))
        receipts.waitForWritesForTesting()
        DispatchQueue.concurrentPerform(iterations: 200) { index in
            receipts.recordAgentRead(toolName: index.isMultiple(of: 2) ? "brain_search" : "brain_recall", at: now.addingTimeInterval(-30))
        }
        receipts.recordAgentRead(toolName: "brain_store", at: now)
        receipts.recordAgentRead(toolName: "brain_search", at: now.addingTimeInterval(-2 * 3600))
        receipts.recordAgentRead(toolName: "brain_recall", at: now.addingTimeInterval(-23 * 3600))
        receipts.waitForWritesForTesting()
        let restarted = BrainBarOperationReceipts(url: url)
        XCTAssertEqual(restarted.agentReadActivity(minutes: 1440, count: 60, end: now)?.peakPerMinute, 200)
        for (minutes, total) in [(60, 200), (180, 201), (1440, 202)] {
            XCTAssertEqual(restarted.agentReadBuckets(minutes: minutes, count: 60, end: now)?.reduce(0, +), total)
        }
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let counts = try XCTUnwrap(json["agentReads"] as? [String: Int])
        XCTAssertEqual(counts.count, 3)
        XCTAssertEqual(counts.values.reduce(0, +), 202)
    }

    func testLegacyAndDamagedSidecarsStayUnavailable() throws {
        let url = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        for content in ["{}", "broken"] {
            try Data(content.utf8).write(to: url)
            XCTAssertNil(BrainBarOperationReceipts(url: url).agentReadBuckets(minutes: 60, count: 60, end: Date()))
        }
    }

    func testOldIngestWordingIsAbsentFromSources() throws {
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar")
        let files = try XCTUnwrap(FileManager.default.enumerator(at: source, includingPropertiesForKeys: nil))
        for case let url as URL in files where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            XCTAssertFalse(text.contains("read from agents"), url.path)
        }
    }

    @MainActor
    func testSelectedRangeLoadsSidecarAndHoldsItWithTheChart() async throws {
        let url = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        let now = Date(timeIntervalSince1970: 6_000_000)
        let receipts = BrainBarOperationReceipts(url: url)
        for age in [30.0, 7200, 82800] {
            receipts.recordAgentRead(toolName: "brain_recall", at: now.addingTimeInterval(-age))
        }
        receipts.waitForWritesForTesting()
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        store.waveNow = { now }
        store.agentReadReceiptsURL = url
        store.fetchWaveBuckets = { minutes, count in
            .init(activityWindowMinutes: minutes, bucketCount: count, allWriteBuckets: [],
                  agentWriteBuckets: Array(repeating: 0, count: count), watcherWriteBuckets: Array(repeating: 0, count: count),
                  enrichmentBuckets: [], watcherFlowReadability: .readable)
        }
        store.setWindowActive(true)
        defer { store.setWindowActive(false) }
        await store.waitForWaveFetchForTesting()
        XCTAssertEqual(store.inputs.heroWaves[.hour]?.agentReads?.reduce(0, +), 1)
        for (range, expected) in [(NDRange.threeHours, 2), (.day, 3)] {
            store.range = range
            await store.waitForWaveFetchForTesting()
            XCTAssertEqual(store.inputs.heroWaves[range]?.agentReads?.reduce(0, +), expected)
        }
        store.holdChart(true)
        receipts.recordAgentRead(toolName: "brain_search", at: now.addingTimeInterval(-20))
        receipts.waitForWritesForTesting()
        store.refreshWaves(force: true)
        await store.waitForWaveFetchForTesting()
        XCTAssertEqual(store.inputs.heroWaves[.day]?.agentReads?.reduce(0, +), 3)
        store.holdChart(false)
        XCTAssertEqual(store.inputs.heroWaves[.day]?.agentReads?.reduce(0, +), 4)
    }

    @MainActor
    func testThirdSeriesIsDrawnIndependentlyOfTotals() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        for (_, appearance) in NDRenderHarness.appearances {
            var pictures: [Data] = []
            for counts in [[10, 0, 0, 0], [0, 0, 10, 0]] {
                var inputs = BrainBarDesignFixture.inputs(.calm)
                inputs.heroWaves[.hour]?.agentReads = counts
                let store = NDStore(fixture: inputs, page: .overview)
                let rep = try NDRenderHarness.host(NDActivitySection(store: store).padding(24)
                    .frame(width: NDGrid.windowWidth, height: 340, alignment: .topLeading).background(ND.bg),
                    size: NSSize(width: NDGrid.windowWidth, height: 340), appearance: appearance)
                pictures.append(try XCTUnwrap(rep.representation(using: .png, properties: [:])))
            }
            XCTAssertNotEqual(pictures[0], pictures[1], "Same totals and peak, different call buckets must draw different charts")
        }
    }

    enum ActivityState: String, CaseIterable {
        case quiet, readsOnly, savesOnly, burst, readsMore, readsLess, sixDigit, unavailable, unread
    }

    @MainActor
    static func activityInputs(_ state: ActivityState, range: NDRange) -> NDInputs {
        var inputs = BrainBarDesignFixture.inputs(.calm)
        let multiplier = state == .sixDigit ? 1 : range.minutes / 60
        var saves = Array(repeating: 0, count: 60)
        var lines = saves
        var calls = saves
        for index in 0..<60 where index != 15 {
            switch state {
            case .quiet: break
            case .readsOnly: calls[index] = index % 20 == 8 ? 12 : 0
            case .savesOnly: saves[index] = index.isMultiple(of: 3) ? 4 : 0
            case .sixDigit: saves[index] = 4000; lines[index] = 6000; calls[index] = 3000
            case .readsMore: saves[index] = 1; lines[index] = 3; calls[index] = 8
            case .readsLess: saves[index] = 12; lines[index] = 3; calls[index] = 1
            case .burst, .unavailable, .unread:
                saves[index] = index.isMultiple(of: 5) ? 3 : 0
                lines[index] = 2
                calls[index] = index % 20 == 8 ? 12 : 0
            }
        }
        let end = inputs.heroWaves[range]?.endsAt
        inputs.heroWaves[range] = NDWave(saves: saves.map { $0 * multiplier }, reads: lines.map { $0 * multiplier },
            agentReads: state == .unavailable ? nil : calls.map { $0 * multiplier },
            agentReadsPeakPerMinute: state == .unavailable ? nil : calls.max(), readable: state != .unread, endsAt: end)
        return inputs
    }

    @MainActor
    func testActivityStateTableAtRealWidthInBothAppearances() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let output = ProcessInfo.processInfo.environment["BRAINBAR_ACTIVITY_PROOF_DIR"]
        for state in ActivityState.allCases {
            for range in NDRange.allCases {
                let inputs = Self.activityInputs(state, range: range)
                let wave = inputs.heroWaves[range]!
                let store = NDStore(fixture: inputs, page: .overview)
                store.range = range
                let naturalWidth = NSHostingView(rootView: NDActivitySection(store: store)
                    .fixedSize(horizontal: true, vertical: false)).fittingSize.width
                if state == .sixDigit { print("Activity six-digit \(range.rawValue) page width: \(naturalWidth) pt (852 pt available)") }
                XCTAssertLessThanOrEqual(naturalWidth, NDGrid.windowWidth - 2 * NDGrid.pageSide + 0.5,
                                         "\(state) \(range): full header must fit the real page width")
                let modes: [(String, Int?)] = state == .burst || state == .sixDigit
                    ? [("rest", nil), ("zero-hover", 15), ("peak-hover", 8)] : [("rest", nil)]
                for (name, appearance) in NDRenderHarness.appearances {
                    for (mode, hover) in modes {
                        let rep = try NDRenderHarness.host(
                            NDActivitySection(store: store, hover: hover).padding(NDGrid.pageSide)
                                .frame(width: NDGrid.windowWidth, height: 340, alignment: .topLeading).background(ND.bg),
                            size: NSSize(width: NDGrid.windowWidth, height: 340), appearance: appearance)
                        if let output {
                            let directory = URL(fileURLWithPath: output)
                            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                            try NDRenderHarness.write(rep, name: "\(state.rawValue)-\(range.rawValue)-\(name)-\(mode)", in: directory)
                        }
                        let request = VNRecognizeTextRequest()
                        request.recognitionLevel = .accurate
                        try VNImageRequestHandler(cgImage: try XCTUnwrap(rep.cgImage)).perform([request])
                        let text = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ")
                        let context = "\(state) \(range) \(name) \(mode): \(text)"
                        XCTAssertTrue(text.contains("session lines ingested"), context)
                        XCTAssertTrue(text.contains("reads by agents"), context)
                        XCTAssertTrue(text.contains("saved by agents"), context)
                        let peaks = wave.readable ? [wave.savesPeak, wave.readsPeak, wave.agentReads?.max() ?? 0].filter { $0 > 0 }.count : 0
                        XCTAssertEqual(text.components(separatedBy: "up to").count - 1, peaks, context)
                        let span = max(1, range.minutes / wave.count)
                        let suffix = span == 1 ? "/min" : "/\(span)min"
                        XCTAssertFalse(text.contains(" per "), context)
                        let compactText = text.filter { !$0.isWhitespace }
                        XCTAssertEqual(compactText.components(separatedBy: suffix).count - 1, peaks, context)
                        if state == .sixDigit {
                            for total in [wave.savesTotal, wave.readsTotal, wave.agentReads!.reduce(0, +)] {
                                XCTAssertTrue(text.contains(NDFormat.integer(total)), context)
                            }
                        }
                    }
                }
            }
        }
    }
}
