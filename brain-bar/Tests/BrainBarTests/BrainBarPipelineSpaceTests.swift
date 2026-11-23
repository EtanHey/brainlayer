import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// L63 geometry through the same measured panel path used by the offscreen hover renderer.
@MainActor final class BrainBarPipelineSpaceTests: XCTestCase {
    func testStationRowsAndActivityGapStayAtTheOriginalPositions() {
        let L = NDPipelineLayout.self
        XCTAssertEqual(L.yTop, 44)
        XCTAssertEqual(L.ySaves, 114)
        XCTAssertEqual(L.hub.minY, 24)
        XCTAssertEqual(L.keptSafe.minY, 18)
        XCTAssertEqual(L.point(.jobs).y, 138)
        XCTAssertEqual(L.point(.weekly).y, 138)
        XCTAssertEqual(NDGrid.activityToPipeline, 0)
    }

    func testWholeOverviewIsShorterThanTheOriginalWindow() {
        for (name, baseline): (BrainBarDesignFixture.Name, CGFloat) in [(.calm, 791), (.driveSignedOut, 841)] {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
            let height = NDPageHeight.window(store, page: .overview, screen: 2000, titleBarHeight: 52)
            XCTAssertLessThan(height, baseline, name.rawValue)
        }
    }

    func testMemoryPrefersTheClearPocketBelowItsBox() {
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories,
                items: store.r1.stationItems, backupProgress: store.inputs.backupProgress)
            let panel = chart.panel(.memory, content: AnyView(NDStationPopover(store: store, station: .memory)))
            XCTAssertEqual(panel.edge, .bottom, name.rawValue)
            XCTAssertLessThan(panel.body.width, 300)
            let left = NDPipelineLayout.labelRects(.jobs, store.r1.pipeline).map(\.maxX).max()! + 4
            let right = NDPipelineLayout.labelRects(.weekly, store.r1.pipeline).map(\.minX).min()! - 4
            XCTAssertGreaterThanOrEqual(panel.body.minX, left, name.rawValue)
            XCTAssertLessThanOrEqual(panel.body.maxX, right, name.rawValue)
            XCTAssertFalse(panel.scrolls, name.rawValue)
        }
    }

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
                    XCTAssertFalse(panel.body.intersects(NDPipelineLayout.circle(other).insetBy(dx: -4, dy: -4)), context)
                    for words in chart.displayedPanelWords[other] ?? NDPipelineLayout.labelRects(other, store.r1.pipeline) {
                        XCTAssertFalse(panel.outer.intersects(words.insetBy(dx: -4, dy: -4)), context)
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
                    let icon = other == station ? L.circle(other) : L.circle(other).insetBy(dx: -4, dy: -4)
                    XCTAssertFalse(panel.body.intersects(icon), "\(station) over \(other)")
                    for words in L.labelRects(other, state) {
                        XCTAssertFalse(panel.outer.intersects(other == station ? words : words.insetBy(dx: -4, dy: -4)), "\(station) over \(other)")
                    }
                }
            }
        }
    }

    func testScreenFitKeepsTheBodyInsideItsClearNaturalFrame() {
        let L = NDPipelineLayout.self
        let screen = CGRect(x: -60, y: -200, width: 940, height: L.size.height + 210)
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories,
                items: store.r1.stationItems, backupProgress: store.inputs.backupProgress)
            for station in NDStation.allCases {
                let natural = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                let fitted = L.fitted(natural, within: screen)
                XCTAssertTrue(natural.body.insetBy(dx: -0.01, dy: -0.01).contains(fitted.body), "\(name.rawValue) \(station)")
                XCTAssertEqual(fitted.tip, natural.tip)
                for other in NDStation.allCases where other != station {
                    XCTAssertFalse(fitted.body.intersects(L.circle(other).insetBy(dx: -4, dy: -4)), "\(station) over \(other)")
                    for words in chart.displayedPanelWords[other] ?? L.labelRects(other, store.r1.pipeline) {
                        XCTAssertFalse(fitted.outer.intersects(words.insetBy(dx: -4, dy: -4)), "\(station) over \(other)")
                    }
                }
            }
        }
    }

    func testBodiesClearTheKeptSafeOutline() {
        let L = NDPipelineLayout.self
        // The one-point outline extends half a point beyond its frame.
        let outline = L.keptSafe.insetBy(dx: -4.5, dy: -4.5)
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories,
                items: store.r1.stationItems, backupProgress: store.inputs.backupProgress)
            for station in NDStation.allCases {
                let panel = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                XCTAssertFalse(panel.body.intersects(outline), "\(name.rawValue) \(station): 4 pt from the outline")
            }
        }
    }

    func testWeeklyAndSearchUseSeparateMemoryPorts() throws {
        let L = NDPipelineLayout.self
        let weekly = try XCTUnwrap(L.connections.first { $0.pipe == .weeklyMemory }?.line.points.last)
        let search = try XCTUnwrap(L.connections.first { $0.pipe == .memorySearch }?.line.points.first)
        XCTAssertGreaterThanOrEqual(abs(weekly.x - search.x), 16)
        XCTAssertEqual(weekly.y, L.hub.maxY + L.edgeGap)
    }

    func testRenderRequestedProof() throws {
        guard let path = ProcessInfo.processInfo.environment["PIPELINE_SPACE_PROOF"] else { return }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (suffix, appearance) in NDRenderHarness.appearances {
            let calm = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
            _ = try NDRenderHarness.window(calm, name: "overview-\(suffix)", appearance: appearance, in: directory)
            for station: NDStation in [.memory, .jobs, .weekly, .queue, .read, .search] {
                _ = try NDRenderHarness.hover(calm, station: station, name: "hover-\(station.rawValue)-\(suffix)", appearance: appearance, in: directory)
            }
            let faulty = NDStore(fixture: BrainBarDesignFixture.inputs(.driveSignedOut), page: .overview)
            _ = try NDRenderHarness.window(faulty, name: "faulty-\(suffix)", appearance: appearance, in: directory)
            let searchFault = NDStore(fixture: BrainBarDesignFixture.inputs(.searchFailing), page: .overview)
            _ = try NDRenderHarness.hover(searchFault, station: .search, name: "hover-search-fault-\(suffix)", appearance: appearance, in: directory)
            if let widePath = ProcessInfo.processInfo.environment["PIPELINE_SPACE_WIDE_PROOF"] {
                let wide = URL(fileURLWithPath: widePath, isDirectory: true)
                try FileManager.default.createDirectory(at: wide, withIntermediateDirectories: true)
                for (fixture, station, label): (BrainBarDesignFixture.Name, NDStation, String) in [
                    (.calm, .mac, "mac"), (.driveSignedOut, .drive, "drive-fault"), (.lagStalled, .read, "read-lag"),
                ] {
                    let store = NDStore(fixture: BrainBarDesignFixture.inputs(fixture), page: .overview)
                    _ = try NDRenderHarness.hover(store, station: station, name: "hover-\(label)-\(suffix)", appearance: appearance, in: wide)
                }
            }
        }
    }
}
