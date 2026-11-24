import XCTest
import SwiftUI
import Vision
@testable import BrainBar

final class BrainBarAgentReadActivityTests: XCTestCase {
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
                    .frame(width: 920, height: 340, alignment: .topLeading).background(ND.bg),
                    size: NSSize(width: 920, height: 340), appearance: appearance)
                pictures.append(try XCTUnwrap(rep.representation(using: .png, properties: [:])))
            }
            XCTAssertNotEqual(pictures[0], pictures[1], "Same totals and peak, different call buckets must draw different charts")
        }
    }

    @MainActor
    func testActivityDrawsUnambiguousLabelsInBothAppearances() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let output = ProcessInfo.processInfo.environment["BRAINBAR_ACTIVITY_PROOF_DIR"]
        for range in [NDRange.hour, .day] {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
            store.range = range
            for (name, appearance) in NDRenderHarness.appearances {
                for hovered in [false, true] {
                    let rep = try NDRenderHarness.host(
                        NDActivitySection(store: store, hover: hovered ? 30 : nil)
                            .padding(24).frame(width: 920, height: 340, alignment: .topLeading).background(ND.bg),
                        size: NSSize(width: 920, height: 340), appearance: appearance)
                    if let output {
                        let directory = URL(fileURLWithPath: output)
                        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                        try NDRenderHarness.write(rep, name: "\(range.rawValue)-\(name)-\(hovered ? "hover" : "rest")", in: directory)
                    }
                    let request = VNRecognizeTextRequest()
                    request.recognitionLevel = .accurate
                    try VNImageRequestHandler(cgImage: try XCTUnwrap(rep.cgImage)).perform([request])
                    let text = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ")
                    XCTAssertTrue(text.contains("session lines ingested"), text)
                    XCTAssertTrue(text.contains("reads by agents"), text)
                    XCTAssertTrue(text.contains("saved by agents"), text)
                }
            }
        }
    }
}
