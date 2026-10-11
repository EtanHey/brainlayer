import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

final class BrainBarEmbeddingResidencyTests: XCTestCase {
    // MARK: - Model name

    func testConfiguredModelNameIsThePythonDefaultModel() throws {
        // There is no config or env key for the embedding model: the hotlane calls
        // `get_embedding_model()` with no argument, so `DEFAULT_MODEL` is the only source.
        let embeddings = try String(
            contentsOf: repoRoot().appendingPathComponent("src/brainlayer/embeddings.py"),
            encoding: .utf8
        )
        let regex = try NSRegularExpression(pattern: #"(?m)^DEFAULT_MODEL\s*=\s*"([^"]+)""#)
        let match = try XCTUnwrap(
            regex.firstMatch(in: embeddings, range: NSRange(embeddings.startIndex..., in: embeddings)),
            "embeddings.py no longer declares DEFAULT_MODEL as a string literal"
        )
        let pythonDefault = String(embeddings[try XCTUnwrap(Range(match.range(at: 1), in: embeddings))])

        XCTAssertEqual(BrainBarEmbeddingModel.configuredName, pythonDefault)
        let presentation = BrainBarModelResidencyPresentation(
            modelName: BrainBarEmbeddingModel.configuredName,
            process: .unmeasurable
        )
        XCTAssertEqual(presentation.value(for: "Model"), pythonDefault)
    }

    // MARK: - Probe

    func testRunningHotlaneReportsItsPIDAndThatPIDsResidentMemory() {
        let probe = HotlaneEmbeddingResidencyProbe(
            processProbe: SequencedProcessProbe([.running(pid: 4242)]),
            residentBytes: { pid in pid == 4242 ? 1_288_490_188 : nil }
        )
        XCTAssertEqual(probe.sample(), .running(pid: 4242, residentBytes: 1_288_490_188))
    }

    func testProbeResolvesThePIDOnEverySampleAndNeverReusesAStaleOne() {
        // #972: a restarted hotlane has a new PID; reading RSS for the old one would
        // describe whatever process now owns that number.
        let seen = PIDRecorder()
        let probe = HotlaneEmbeddingResidencyProbe(
            processProbe: SequencedProcessProbe([.running(pid: 4242), .running(pid: 5151)]),
            residentBytes: { pid in
                seen.record(pid)
                return UInt64(pid) * 1_000
            }
        )
        XCTAssertEqual(probe.sample(), .running(pid: 4242, residentBytes: 4_242_000))
        XCTAssertEqual(probe.sample(), .running(pid: 5151, residentBytes: 5_151_000))
        XCTAssertEqual(seen.pids, [4242, 5151])
    }

    func testAbsentHotlaneIsStoppedAndProbeFailureIsUnmeasurable() {
        let rssNeverRead: @Sendable (pid_t) -> UInt64? = { _ in
            XCTFail("RSS must not be read without a live PID")
            return nil
        }
        let stopped = HotlaneEmbeddingResidencyProbe(
            processProbe: SequencedProcessProbe([.absent]),
            residentBytes: rssNeverRead
        )
        XCTAssertEqual(stopped.sample(), .stopped)

        let failed = HotlaneEmbeddingResidencyProbe(
            processProbe: SequencedProcessProbe([.failure("launchctl exited 1")]),
            residentBytes: rssNeverRead
        )
        XCTAssertEqual(failed.sample(), .unmeasurable)
    }

    func testDefaultProbeTargetsTheHotlaneLaunchdLabel() {
        XCTAssertEqual(HotlaneEmbeddingResidencyProbe.launchdLabel, "com.brainlayer.hotlane-brainbar")
        XCTAssertEqual(HotlaneEmbeddingResidencyProbe.launchdLabel, BrainLayerLaunchdJob.hotlane.launchdLabel)
    }

    // MARK: - Presentation

    func testRunningPresentationShowsConfiguredNamePIDAndLabelledProcessMemory() {
        let bytes: UInt64 = 1_288_490_188
        let presentation = BrainBarModelResidencyPresentation(
            modelName: "BAAI/bge-large-en-v1.5",
            process: .running(pid: 4242, residentBytes: bytes)
        )
        XCTAssertEqual(presentation.rows.map(\.label), ["Model", "Status", "Resident memory"])
        XCTAssertEqual(presentation.value(for: "Model"), "BAAI/bge-large-en-v1.5")
        XCTAssertEqual(presentation.value(for: "Status"), "Hotlane running · PID 4242")
        XCTAssertEqual(
            presentation.value(for: "Resident memory"),
            "Hotlane process · " + ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
        )
        XCTAssertTrue(presentation.value(for: "Resident memory")?.contains("GB") == true)
        XCTAssertFalse(
            presentation.rows.contains { $0.value.localizedCaseInsensitiveContains("model memory") },
            "RSS is the whole hotlane process, never the model alone"
        )
    }

    func testStoppedPresentationIsHonestAndHidesMemory() {
        let presentation = BrainBarModelResidencyPresentation(
            modelName: "BAAI/bge-large-en-v1.5",
            process: .stopped
        )
        XCTAssertEqual(presentation.rows.map(\.label), ["Model", "Status"])
        XCTAssertEqual(presentation.value(for: "Status"), "Hotlane stopped")
        XCTAssertNil(presentation.value(for: "Resident memory"))
    }

    func testUnmeasurableValuesHideTheirRows() {
        let unmeasurable = BrainBarModelResidencyPresentation(
            modelName: "BAAI/bge-large-en-v1.5",
            process: .unmeasurable
        )
        XCTAssertEqual(unmeasurable.rows.map(\.label), ["Model"])

        let rssUnreadable = BrainBarModelResidencyPresentation(
            modelName: "BAAI/bge-large-en-v1.5",
            process: .running(pid: 4242, residentBytes: nil)
        )
        XCTAssertEqual(rssUnreadable.rows.map(\.label), ["Model", "Status"])
    }

    func testNoStateEverRendersAPermanentUnavailable() {
        let states: [BrainBarEmbeddingProcessState] = [
            .running(pid: 1, residentBytes: 1_024),
            .running(pid: 1, residentBytes: nil),
            .stopped,
            .unmeasurable,
        ]
        for state in states {
            let presentation = BrainBarModelResidencyPresentation(
                modelName: BrainBarEmbeddingModel.configuredName,
                process: state
            )
            for row in presentation.rows {
                XCTAssertFalse(row.value.localizedCaseInsensitiveContains("unavailable"), "\(state): \(row)")
            }
        }
    }

    // MARK: - View model

    @MainActor
    func testViewModelShowsOnlyTheModelUntilTheFirstSampleThenTheSampledProcess() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-embedding-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(
                state: .running(pid: 4242, residentBytes: 1_288_490_188)
            ),
            refreshStatusOnLoad: false
        )
        XCTAssertEqual(viewModel.modelResidencyPresentation.rows.map(\.label), ["Model"])

        viewModel.refreshLaunchdStatus()
        let deadline = Date().addingTimeInterval(2)
        while viewModel.modelResidencyPresentation.value(for: "Status") == nil {
            if Date() > deadline {
                XCTFail("Timed out waiting for the embedding sample")
                break
            }
            await Task.yield()
        }

        XCTAssertEqual(viewModel.modelResidencyPresentation.value(for: "Status"), "Hotlane running · PID 4242")
        XCTAssertNotNil(viewModel.modelResidencyPresentation.value(for: "Resident memory"))
    }

    @MainActor
    func testEmbeddingRowsResolveWhileTheAllJobsLaunchdSweepIsStalled() async throws {
        // #974 review B1: the production all-jobs sweep has no deadline. A hung launchctl
        // for any job must not keep the hotlane rows pending or pinned to an old sample.
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-embedding-stall-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let stalled = StalledLaunchdStatusProvider()
        defer { stalled.release() }
        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: stalled,
            embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(
                state: .running(pid: 4242, residentBytes: 1_288_490_188)
            ),
            refreshStatusOnLoad: false
        )

        viewModel.refreshLaunchdStatus()
        let deadline = Date().addingTimeInterval(2)
        while viewModel.modelResidencyPresentation.value(for: "Status") == nil {
            if Date() > deadline {
                XCTFail("Embedding rows stayed pending behind the stalled launchd sweep")
                break
            }
            await Task.yield()
        }

        XCTAssertEqual(viewModel.modelResidencyPresentation.value(for: "Status"), "Hotlane running · PID 4242")
        XCTAssertNotNil(viewModel.modelResidencyPresentation.value(for: "Resident memory"))
        XCTAssertTrue(viewModel.isRefreshingLaunchdStatus, "the all-jobs sweep is still stalled")
    }

    @MainActor
    func testASlowerOlderEmbeddingSampleNeverOverwritesANewerOne() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-embedding-order-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let probe = GatedResidencyProbe(first: .running(pid: 1111, residentBytes: 1_024), then: .stopped)
        defer { probe.releaseFirst() }
        let viewModel = BrainBarSettingsViewModel(
            store: store,
            launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
            embeddingResidencyProbe: probe,
            refreshStatusOnLoad: false
        )

        viewModel.refreshEmbeddingResidency() // blocks inside the probe until released
        // Both requests hop to detached tasks, so without this the second could reach the
        // probe first and be the one that blocks. Wait until request one is inside it.
        let entered = Date().addingTimeInterval(2)
        while !probe.firstCallEntered {
            if Date() > entered {
                XCTFail("First embedding sample never reached the probe")
                return
            }
            await Task.yield()
        }
        viewModel.refreshEmbeddingResidency() // returns .stopped immediately
        let deadline = Date().addingTimeInterval(2)
        while viewModel.embeddingProcess != .stopped {
            if Date() > deadline {
                XCTFail("Newer embedding sample never published")
                break
            }
            await Task.yield()
        }

        probe.releaseFirst()
        let settle = Date().addingTimeInterval(0.3)
        while Date() < settle { await Task.yield() }
        XCTAssertEqual(viewModel.embeddingProcess, .stopped, "the stale PID 1111 sample must be dropped")
    }

    // MARK: - Render

    @MainActor
    func testAdvancedEmbeddingRowsRenderAtEveryWindowWidth() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-embedding-render-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let scenarios: [(String, BrainBarEmbeddingProcessState)] = [
            ("running", .running(pid: 4242, residentBytes: 1_288_490_188)),
            ("stopped", .stopped),
        ]
        for (name, state) in scenarios {
            let viewModel = BrainBarSettingsViewModel(
                store: store,
                launchdStatusProvider: StaticBrainLayerLaunchdStatusProvider(states: [:]),
                embeddingResidencyProbe: StaticBrainBarEmbeddingResidencyProbe(state: state),
                initialEmbeddingProcess: state,
                refreshStatusOnLoad: false
            )
            for width in [760, 960, 1_280] {
                let view = NSHostingView(
                    rootView: BrainBarSettingsView(viewModel: viewModel, initialSection: .advanced)
                        .transaction { $0.disablesAnimations = true }
                )
                view.frame = NSRect(x: 0, y: 0, width: width, height: 720)
                view.layoutSubtreeIfNeeded()
                let bitmap = try XCTUnwrap(view.bitmapImageRepForCachingDisplay(in: view.bounds))
                view.cacheDisplay(in: view.bounds, to: bitmap)
                let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
                XCTAssertGreaterThan(png.count, 1_000, "\(name) @ \(width) rendered empty")
                if let dir = ProcessInfo.processInfo.environment["BRAINBAR_SETTINGS_RENDER_DIR"], !dir.isEmpty {
                    let url = URL(fileURLWithPath: dir, isDirectory: true)
                        .appendingPathComponent("settings-advanced-embedding-\(name)-\(width).png")
                    try FileManager.default.createDirectory(
                        at: url.deletingLastPathComponent(), withIntermediateDirectories: true
                    )
                    try png.write(to: url)
                }
            }
        }
    }

    private func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // BrainBarTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // brain-bar
            .deletingLastPathComponent() // repo root
    }
}

private final class SequencedProcessProbe: WatcherProcessProbing, @unchecked Sendable {
    private let lock = NSLock()
    private var results: [WatcherProcessProbeResult]

    init(_ results: [WatcherProcessProbeResult]) { self.results = results }

    func sample() -> WatcherProcessProbeResult {
        lock.lock()
        defer { lock.unlock() }
        return results.count > 1 ? results.removeFirst() : results[0]
    }
}

private final class PIDRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [pid_t] = []

    func record(_ pid: pid_t) {
        lock.lock()
        recorded.append(pid)
        lock.unlock()
    }

    var pids: [pid_t] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }
}

private final class StalledLaunchdStatusProvider: BrainLayerLaunchdStatusSampling, @unchecked Sendable {
    private let gate = DispatchSemaphore(value: 0)

    func sample() -> [BrainLayerLaunchdJob: BrainLayerLaunchdLoadState] {
        gate.wait()
        gate.signal()
        return [:]
    }

    func release() { gate.signal() }
}

private final class GatedResidencyProbe: BrainBarEmbeddingResidencySampling, @unchecked Sendable {
    private let lock = NSLock()
    private let gate = DispatchSemaphore(value: 0)
    private let first: BrainBarEmbeddingProcessState
    private let then: BrainBarEmbeddingProcessState
    private var calls = 0
    private var entered = false

    init(first: BrainBarEmbeddingProcessState, then: BrainBarEmbeddingProcessState) {
        self.first = first
        self.then = then
    }

    func sample() -> BrainBarEmbeddingProcessState {
        lock.lock()
        calls += 1
        let isFirst = calls == 1
        lock.unlock()
        guard isFirst else { return then }
        lock.lock()
        entered = true
        lock.unlock()
        gate.wait()
        return first
    }

    var firstCallEntered: Bool {
        lock.lock()
        defer { lock.unlock() }
        return entered
    }

    func releaseFirst() { gate.signal() }
}
