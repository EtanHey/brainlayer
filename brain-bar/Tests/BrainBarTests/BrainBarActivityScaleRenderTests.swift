import XCTest
import SwiftUI
@testable import BrainBar

/// w315: the Activity chart's heights read honestly. This file uses only API that exists on main as well, so the same
/// code renders the before and after proof and runs the main-compatible red check.
final class BrainBarActivityScaleRenderTests: XCTestCase {
    enum Case: String, CaseIterable {
        /// The owner's moment: one save, one read by an agent and three session lines in one bucket (index 40), with
        /// session lines reaching 30 elsewhere in the hour.
        case ownerCase = "owner-case"
        /// One save in a bucket while reads by agents reach 8 elsewhere: the two up series on one scale or not.
        case sharedUp = "shared-up"
        /// The busy 24 h fixture (calm, its 24 h wave).
        case busyDay = "busy-24h"

        var range: NDRange { self == .busyDay ? .day : .hour }
        var hover: Int { self == .busyDay ? 48 : 40 }

        /// R2 (Cursor C3): each half's expected cue: (count, set by reads by agents) above, count below.
        var expectedUp: (count: Int, byReads: Bool) {
            switch self {
            case .ownerCase: (1, false)  // a 1-1 tie between saves and reads is the saves'
            case .sharedUp: (8, true)
            case .busyDay: (288, true)
            }
        }
        var expectedDown: Int { self == .busyDay ? 1_511 : 30 }
    }

    @MainActor
    static func inputs(_ c: Case) -> NDInputs {
        var inputs = BrainBarDesignFixture.inputs(.calm)
        guard c != .busyDay else { return inputs }
        let end = inputs.heroWaves[.hour]?.endsAt
        var saves = Array(repeating: 0, count: 60), lines = saves, calls = saves
        for index in 0..<60 {
            lines[index] = 5 + (index * 7) % 26
            if index.isMultiple(of: 9) { saves[index] = 1 }
            if index % 11 == 3 { calls[index] = c == .sharedUp ? 8 : 1 }
        }
        lines[12] = 30
        saves[40] = 1; calls[40] = 1; lines[40] = 3
        inputs.heroWaves[.hour] = NDWave(saves: saves, reads: lines, agentReads: calls, agentReadsPeakPerMinute: calls.max(),
                                         readable: true, endsAt: end)
        return inputs
    }

    /// Fails on main: one save drew as tall as eight reads by agents (each series to its own maximum).
    @MainActor
    func testOneSaveDrawsOneEighthOfTheUpHalfWhenReadsByAgentsReachEight() throws {
        let wave = try XCTUnwrap(Self.inputs(.sharedUp).heroWaves[.hour])
        XCTAssertEqual(wave.savesPeak, 1)
        XCTAssertEqual(wave.agentReads?.max(), 8)
        XCTAssertEqual(NDWaveform.savedHeight(wave, saves: 1, size: .hero), NDWaveform.halves(.hero).up / 8, accuracy: 0.001)
    }

    /// Before and after renders, dark and light, at rest and with the case's column hovered.
    /// `BRAINBAR_ACTIVITY_SCALE_PROOF_DIR=<abs dir>` writes them.
    @MainActor
    func testRenderActivityScaleProof() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let output = ProcessInfo.processInfo.environment["BRAINBAR_ACTIVITY_SCALE_PROOF_DIR"]
        let canvas = NDActivitySection.canvas
        let cy = NDWaveform.plot(canvas, nowTick: true).cy
        let axis = NDWaveform.baseline(canvas, nowTick: true).midY
        let (up, down) = NDWaveform.halves(.hero)
        for c in Case.allCases {
            let store = NDStore(fixture: Self.inputs(c), page: .overview)
            store.range = c.range
            // R2 (Cursor C3): the wave actually hosted below carries the expected cues and heights.
            let wave = try XCTUnwrap(store.inputs.heroWaves[c.range], c.rawValue)
            XCTAssertEqual(NDActivitySection.scaleTicks(wave),
                           [.init(half: .up, count: c.expectedUp.count, y: cy - 0.5 - up, setByAgentReads: c.expectedUp.byReads),
                            .init(half: .down, count: c.expectedDown, y: cy + 0.5 + down)], c.rawValue)
            let saves = wave.saves[c.hover]
            let expected = saves == 0 ? 0 : max(NDWaveform.minVisible(.hero), CGFloat(saves) / CGFloat(c.expectedUp.count) * up)
            XCTAssertEqual(NDWaveform.savedHeight(wave, saves: saves, size: .hero), expected, accuracy: 0.001, c.rawValue)
            XCTAssertEqual(NDWaveform.columnTop(wave, index: c.hover, canvas: canvas, nowTick: true, size: .hero),
                           cy - 0.5 - expected, accuracy: 0.001, "\(c.rawValue): the readout sits on the drawn bar")
            if let calls = wave.agentReads, calls[c.hover] == saves {
                XCTAssertEqual(axis - NDActivitySection.agentReadY(count: calls[c.hover], peak: NDWaveform.upPeak(wave), canvas: canvas),
                               expected, accuracy: 0.001, "\(c.rawValue): equal counts above the line, equal heights")
            }
            for (name, appearance) in NDRenderHarness.appearances {
                for (mode, hover) in [("rest", nil), ("hover", c.hover)] as [(String, Int?)] {
                    let rep = try NDRenderHarness.host(
                        NDActivitySection(store: store, hover: hover).padding(NDGrid.pageSide)
                            .frame(width: NDGrid.windowWidth, height: 250, alignment: .topLeading).background(ND.bg),
                        size: NSSize(width: NDGrid.windowWidth, height: 250), appearance: appearance)
                    XCTAssertNotNil(rep.cgImage)
                    if let output {
                        let directory = URL(fileURLWithPath: output)
                        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                        try NDRenderHarness.write(rep, name: "\(c.rawValue)-\(mode)-\(name)", in: directory)
                    }
                }
            }
        }
    }
}
