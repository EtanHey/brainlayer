import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// L63 geometry through the same measured panel path used by the offscreen hover renderer.
@MainActor final class BrainBarPipelineSpaceTests: XCTestCase {
    func testReturnLaneFollowsTheLowestDrawnWords() throws {
        for name in BrainBarDesignFixture.Name.allCases {
            let state = NDPipelineState.make(BrainBarDesignFixture.inputs(name))
            let bottom = NDStation.allCases.filter { $0 != .search }.flatMap {
                NDPipelineLayout.labelRects($0, state)
            }.map(\.maxY).max() ?? 0
            let captionTop = NDPipelineLayout.yAnswers - 8
            XCTAssertGreaterThanOrEqual(captionTop - bottom, 32)
            XCTAssertLessThanOrEqual(captionTop - bottom, 34, name.rawValue)
        }
        let L = NDPipelineLayout.self
        let state = NDPipelineState.make(BrainBarDesignFixture.inputs(.calm))
        let chart = NDPipelineChart(state: state, memories: "123", interactive: false)
            .background(ND.card).frame(width: L.size.width, height: L.size.height)
        let rep = try NDRenderHarness.host(chart, size: L.size, appearance: .darkAqua)
        let background = try XCTUnwrap(rep.colorAt(x: 0, y: 0)?.usingColorSpace(.deviceRGB))
        func containsInk(_ rect: CGRect) -> Bool {
            for y in Int(rect.minY * 2)..<Int(rect.maxY * 2) {
                for x in Int(rect.minX * 2)..<Int(rect.maxX * 2) {
                    guard let c = rep.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { continue }
                    if max(abs(c.redComponent - background.redComponent),
                           abs(c.greenComponent - background.greenComponent),
                           abs(c.blueComponent - background.blueComponent)) > 0.04 { return true }
                }
            }
            return false
        }
        XCTAssertTrue(containsInk(try XCTUnwrap(L.labelRects(.jobs, state).last)), "the chart draws the lowest status")
        XCTAssertTrue(containsInk(CGRect(x: 400, y: L.yAnswers - 2, width: 200, height: 4)), "the chart draws the return line at the measured position")
    }

    func testJobStationsSpreadBeyondTheMemoryBox() {
        let L = NDPipelineLayout.self
        let distance = L.point(.weekly).x - L.point(.jobs).x
        print("PIPELINE card=\(L.size.height) jobs=\(distance)")
        XCTAssertGreaterThan(distance, 186)
        XCTAssertLessThan(L.point(.jobs).x, L.hub.minX)
        XCTAssertGreaterThan(L.point(.weekly).x, L.hub.maxX)
    }

    func testEveryDrawnHoverClearsEveryOtherStation() {
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories,
                items: store.r1.stationItems, backupProgress: store.inputs.backupProgress)
            for station in NDStation.allCases {
                let content = AnyView(NDStationPopover(store: store, station: station))
                let panel = chart.panel(station, content: content)
                XCTAssertFalse(panel.scrolls, "\(name.rawValue) \(station): full content must fit")
                XCTAssertGreaterThanOrEqual(panel.body.height + 0.01, NDStationPopover.measure(content, width: panel.body.width).height)
                for other in NDStation.allCases where other != station {
                    let context = "\(name.rawValue) \(station) covering \(other)"
                    XCTAssertFalse(panel.body.intersects(NDPipelineLayout.circle(other)), context)
                    for words in chart.displayedPanelWords[other] ?? NDPipelineLayout.labelRects(other, store.r1.pipeline) {
                        XCTAssertFalse(panel.outer.intersects(words), context)
                    }
                }
            }
        }
    }

    func testOversizedPanelsStillUseCheckedClearance() {
        let L = NDPipelineLayout.self
        for name in BrainBarDesignFixture.Name.allCases {
            let state = NDPipelineState.make(BrainBarDesignFixture.inputs(name))
            for station in NDStation.allCases {
                let panel = L.panel(station, state: state, actions: [], height: 2000)
                XCTAssertTrue(panel.scrolls)
                XCTAssertFalse(panel.outer.intersects(L.pipelineHeading))
                for other in NDStation.allCases {
                    XCTAssertFalse(panel.body.intersects(L.circle(other)), "\(station) over \(other)")
                    for words in L.labelRects(other, state) {
                        XCTAssertFalse(panel.outer.intersects(words), "\(station) over \(other)")
                    }
                }
            }
        }
    }

    func testRenderRequestedProof() throws {
        guard let path = ProcessInfo.processInfo.environment["PIPELINE_SPACE_PROOF"] else { return }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (suffix, appearance) in NDRenderHarness.appearances {
            let calm = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
            _ = try NDRenderHarness.window(calm, name: "overview-\(suffix)", appearance: appearance, in: directory)
            for station: NDStation in [.memory, .jobs, .weekly] {
                _ = try NDRenderHarness.hover(calm, station: station, name: "hover-\(station.rawValue)-\(suffix)", appearance: appearance, in: directory)
            }
            let faulty = NDStore(fixture: BrainBarDesignFixture.inputs(.driveSignedOut), page: .overview)
            _ = try NDRenderHarness.window(faulty, name: "faulty-\(suffix)", appearance: appearance, in: directory)
        }
    }
}
