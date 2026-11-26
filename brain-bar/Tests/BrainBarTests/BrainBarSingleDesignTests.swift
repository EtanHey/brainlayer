import AppKit
import SwiftUI
import Vision
import XCTest
@testable import BrainBar

@MainActor final class BrainBarSingleDesignTests: XCTestCase {
    func testStoredClassicAndUnknownValuesResolveToNewWithoutWriting() {
        let name = "BrainBarSingleDesignTests." + UUID().uuidString
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        for stored in ["classic", "new", "unknown"] {
            defaults.set(stored, forKey: BrainBarDesign.key)
            XCTAssertEqual(BrainBarDesign.current(in: defaults), .new, stored)
            XCTAssertEqual(defaults.string(forKey: BrainBarDesign.key), stored)
        }
    }

    func testDrawnSettingsHasAccountAndDiagnosticsWithoutDesignChoice() throws {
        for (name, appearance) in NDRenderHarness.appearances {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
            let view = NDSettingsView(store: store)
            let size = NDSettingsWindowController.contentSize(of: NSHostingView(rootView: view))
            let rep = try NDRenderHarness.host(view, size: size, appearance: appearance)
            if let directory = ProcessInfo.processInfo.environment["BRAINBAR_SINGLE_DESIGN_PROOF"] {
                let url = URL(fileURLWithPath: directory, isDirectory: true)
                try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
                try NDRenderHarness.write(rep, name: "settings-" + name, in: url)
            }
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            try VNImageRequestHandler(cgImage: XCTUnwrap(rep.cgImage)).perform([request])
            let words = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
            XCTAssertTrue(words.contains("Drive"), words)
            XCTAssertTrue(words.contains("Diagnostics"), words)
            XCTAssertFalse(words.contains("Design"), words)
            XCTAssertFalse(words.contains("Classic"), words)
            XCTAssertFalse(words.contains("switch back"), words)
        }
    }
}
