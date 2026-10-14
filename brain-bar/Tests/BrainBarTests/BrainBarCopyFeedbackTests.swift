import AppKit
import SwiftUI
import Vision
import Foundation
import XCTest
@testable import BrainBar

@MainActor
final class BrainBarCopyFeedbackTests: XCTestCase {
    @MainActor
    private final class Scheduler {
        var delays: [TimeInterval] = []
        var callbacks: [@MainActor @Sendable () -> Void] = []
        func schedule(_ delay: TimeInterval, _ callback: @escaping @MainActor @Sendable () -> Void) {
            delays.append(delay)
            callbacks.append(callback)
        }
    }

    private func model(_ scheduler: Scheduler) throws -> (BrainBarSettingsViewModel, BrainBarRecordingWorkspace) {
        let root = brainBarTestTemporaryDirectory().appendingPathComponent("brainbar-copy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: root) }
        let store = BrainLayerConfigStore(configURL: root.appendingPathComponent("brainlayer.env"))
        try store.save(.defaultConfig)
        let workspace = BrainBarRecordingWorkspace()
        return (BrainBarSettingsViewModel(workspace: workspace, store: store, refreshStatusOnLoad: false,
            copyFeedbackScheduler: { delay, action in scheduler.schedule(delay, action) }), workspace)
    }

    func testCopyRecordsValueShowsCopiedAndRevertsAfterInjectedTimeout() throws {
        let scheduler = Scheduler()
        let (model, workspace) = try model(scheduler)
        XCTAssertNil(model.copiedTechnicalDetailID)
        model.copyTechnicalDetail(.init(label: "Drive file", value: "fixture-drive-file"))
        XCTAssertEqual(workspace.recorded.map { $0.0 }, ["copy"])
        XCTAssertEqual(workspace.recorded.map { $0.1 }, ["fixture-drive-file"])
        XCTAssertEqual(model.copiedTechnicalDetailID, "Drive file")
        XCTAssertEqual(scheduler.delays, [1.5])
        try XCTUnwrap(scheduler.callbacks.first)()
        XCTAssertNil(model.copiedTechnicalDetailID)
    }

    func testAnOlderTimeoutCannotClearNewCopyFeedback() throws {
        let scheduler = Scheduler()
        let (model, workspace) = try model(scheduler)
        model.copyTechnicalDetail(.init(label: "First", value: "first"))
        model.copyTechnicalDetail(.init(label: "Second", value: "second"))
        scheduler.callbacks[0]()
        XCTAssertEqual(model.copiedTechnicalDetailID, "Second")
        scheduler.callbacks[1]()
        XCTAssertNil(model.copiedTechnicalDetailID)
        XCTAssertEqual(workspace.recorded.map { $0.1 }, ["first", "second"])
    }

    func testRepeatedCopyOfSameValueRestartsItsConfirmation() throws {
        let scheduler = Scheduler()
        let (model, _) = try model(scheduler)
        model.copyTechnicalDetail(.init(label: "Same", value: "same"))
        model.copyTechnicalDetail(.init(label: "Same", value: "same"))
        scheduler.callbacks[0]()
        XCTAssertEqual(model.copiedTechnicalDetailID, "Same")
        scheduler.callbacks[1]()
        XCTAssertNil(model.copiedTechnicalDetailID)
    }

    func testEqualValuesOnlyConfirmTheClickedRow() throws {
        let scheduler = Scheduler()
        let (model, _) = try model(scheduler)
        model.copyTechnicalDetail(.init(label: "First", value: "same"))
        model.copyTechnicalDetail(.init(label: "Second", value: "same"))
        XCTAssertEqual(model.copiedTechnicalDetailID, "Second")
        scheduler.callbacks[0]()
        XCTAssertEqual(model.copiedTechnicalDetailID, "Second")
        scheduler.callbacks[1]()
        XCTAssertNil(model.copiedTechnicalDetailID)
    }

    private func card(_ model: BrainBarSettingsViewModel) -> BrainBarBackupChecksCard {
        BrainBarBackupChecksCard(checks: .init(summary: "Verified", checks: [], alert: nil,
            alertTone: nil, details: [.init(label: "Drive file", value: "fixture-drive-file")],
            attentionSentence: nil), hiddenAlert: nil, detailsExpanded: .constant(true),
            copiedDetailID: model.copiedTechnicalDetailID,
            copy: { model.copyTechnicalDetail($0) })
    }

    private func renderedText(_ model: BrainBarSettingsViewModel, scheme: ColorScheme,
                              name: String) throws -> String {
        let view = card(model).frame(width: 600, height: 180, alignment: .topLeading)
            .foregroundStyle(Color.brainBarTextPrimary)
            .padding(16).background(Color.brainBarBackgroundBase).environment(\.colorScheme, scheme)
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: 632, height: 212)
        host.appearance = NSAppearance(named: scheme == .dark ? .darkAqua : .aqua)
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let cg = try XCTUnwrap(bitmap.cgImage)
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesCPUOnly = true
        try VNImageRequestHandler(cgImage: cg, options: [:]).perform([request])
        if let output = ProcessInfo.processInfo.environment["BRAINBAR_SNAPSHOT_DIR"] {
            try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(
                to: URL(fileURLWithPath: output).appendingPathComponent("copy-\(name)-\(scheme).png"))
        }
        return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ")
    }

    func testActualCardShowsCopiedThenRevertsInBothAppearances() throws {
        for scheme in [ColorScheme.light, .dark] {
            let scheduler = Scheduler()
            let (model, workspace) = try model(scheduler)
            XCTAssertFalse(try renderedText(model, scheme: scheme, name: "before").contains("Copied"))
            card(model).copy(.init(label: "Drive file", value: "fixture-drive-file"))
            XCTAssertEqual(workspace.recorded.map { $0.1 }, ["fixture-drive-file"])
            XCTAssertTrue(try renderedText(model, scheme: scheme, name: "copied").contains("Copied"))
            scheduler.callbacks[0]()
            XCTAssertFalse(try renderedText(model, scheme: scheme, name: "after").contains("Copied"))
        }
    }
}
