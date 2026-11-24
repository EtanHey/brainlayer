import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Invented fixtures, offscreen hosting only. No windows or production services.
@MainActor final class BrainBarBubblePlacementTests: XCTestCase {
    func testAgentsBodyCentresWhenItsWidthFitsTheCard() throws {
        let L = NDPipelineLayout.self
        let state = NDPipelineState.make(BrainBarDesignFixture.inputs(.calm))
        let panel = try XCTUnwrap(L.panelCandidate(.agents, edge: .bottom, state: state, actions: [], height: 40, width: 80))
        XCTAssertEqual(panel.body.midX, L.point(.agents).x, accuracy: 0.01)
        XCTAssertEqual(panel.tip.x, panel.body.midX, accuracy: 0.01)
    }

    func testBodiesStayInsideCardAndClearBothCaptions() {
        let L = NDPipelineLayout.self
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories,
                items: store.r1.stationItems, backupProgress: store.inputs.backupProgress)
            for station: NDStation in [.agents, .queue, .search] {
                let p = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                let context = "\(name.rawValue) \(station) \(p.body)"
                XCTAssertFalse(p.scrolls, context)
                XCTAssertGreaterThanOrEqual(p.body.minX, -16, context)
                XCTAssertLessThanOrEqual(p.body.maxX, L.size.width + 16, context)
                XCTAssertLessThanOrEqual(p.body.maxY, L.size.height + L.verticalPadding + 12, context)
                for (_, caption) in L.laneLabelRects() {
                    XCTAssertFalse(p.intersects(caption.insetBy(dx: -4, dy: -4)), context)
                }
                if station == .agents, p.edge == .top || p.edge == .bottom {
                    XCTAssertEqual(p.body.minX, max(-16, L.point(.agents).x - p.body.width / 2), accuracy: 1, context)
                }
            }
        }
    }

    func testMinimumBodyDoesNotEscapeShortScreen() throws {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
        let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
        let content = AnyView(NDStationPopover(store: store, station: .queue))
        let screen = CGRect(x: -16, y: -100, width: NDPipelineLayout.size.width + 32, height: 280)
        let fitted = try XCTUnwrap(chart.panel(.queue, content: content, within: screen))
        XCTAssertTrue(screen.contains(fitted.outer), "\(fitted)")
        XCTAssertGreaterThanOrEqual(fitted.body.height, 32)
        XCTAssertFalse(fitted.body.intersects(NDPipelineLayout.circle(.queue)))
        for (suffix, appearance) in NDRenderHarness.appearances {
            let origin = CGPoint(x: 60, y: 120)
            func render(_ shown: Bool) throws -> NSBitmapImageRep {
                let view = ZStack(alignment: .topLeading) {
                    ND.card
                    Rectangle().stroke(Color.orange, lineWidth: 1)
                        .frame(width: screen.width, height: screen.height)
                        .offset(x: screen.minX + origin.x, y: screen.minY + origin.y)
                    if shown {
                        NDPanelChrome(layout: fitted, content: content)
                            .offset(x: fitted.outer.minX + origin.x, y: fitted.outer.minY + origin.y)
                    }
                }.frame(width: 940, height: 560)
                return try NDRenderHarness.host(view, size: CGSize(width: 940, height: 560), appearance: appearance)
            }
            let blank = try render(false), rep = try render(true)
            var outside = 0
            let firstRow = Int((screen.maxY + origin.y + 1) * 2)
            for y in firstRow..<(firstRow + 60) {
                for x in 0..<rep.pixelsWide {
                    let a = try XCTUnwrap(blank.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                    let b = try XCTUnwrap(rep.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                    if max(abs(a.redComponent - b.redComponent), abs(a.greenComponent - b.greenComponent),
                           abs(a.blueComponent - b.blueComponent)) > 0.08 { outside += 1 }
                }
            }
            XCTAssertEqual(outside, 0, "no bubble ink below the visible screen")
            if let path = ProcessInfo.processInfo.environment["BUBBLE_PLACEMENT_PROOF"] {
                let directory = URL(fileURLWithPath: path, isDirectory: true)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try NDRenderHarness.write(rep, name: "short-screen-\(suffix)", in: directory)
            }
        }
    }

    func testOffscreenTargetDoesNotProduceAnOffscreenMinimumBody() {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
        let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
        XCTAssertNil(chart.panel(.queue, content: AnyView(NDStationPopover(store: store, station: .queue)),
                                 within: CGRect(x: -16, y: -100, width: 852, height: 80)))
    }

    func testOlderLabelIsActuallyDrawnBeforeRail() throws {
        for (_, appearance) in NDRenderHarness.appearances {
            func render(_ label: String?) throws -> NSBitmapImageRep {
                let rail = NDRail(past: 24, ahead: 12, events: [.init(hours: -144, kind: .fail)], olderLabel: label)
                let view = rail.frame(width: 200).offset(x: 80, y: 20)
                    .frame(width: 300, height: 54, alignment: .topLeading).background(ND.card)
                return try NDRenderHarness.host(view, size: CGSize(width: 300, height: 54), appearance: appearance)
            }
            let blank = try render(nil), label = try render("6 d")
            var left = 0, onRail = 0
            for y in 35..<75 {
                for x in 0..<600 {
                    let a = try XCTUnwrap(blank.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                    let b = try XCTUnwrap(label.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                    if max(abs(a.redComponent - b.redComponent), abs(a.greenComponent - b.greenComponent),
                           abs(a.blueComponent - b.blueComponent)) > 0.08 {
                        if x < 160 { left += 1 } else { onRail += 1 }
                    }
                }
            }
            XCTAssertGreaterThan(left, 15, "label ink belongs before the rail")
            XCTAssertEqual(onRail, 0, "label ink must never occupy the rail")
            if let path = ProcessInfo.processInfo.environment["BUBBLE_PLACEMENT_PROOF"] {
                let directory = URL(fileURLWithPath: path, isDirectory: true)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try NDRenderHarness.write(label, name: "older-\(appearance.rawValue)", in: directory)
            }
        }
    }

    func testRenderPlacementProof() throws {
        guard let path = ProcessInfo.processInfo.environment["BUBBLE_PLACEMENT_PROOF"] else { return }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (suffix, appearance) in NDRenderHarness.appearances {
            for (name, station): (BrainBarDesignFixture.Name, NDStation) in [(.calm, .agents), (.calm, .queue), (.searchFailing, .search)] {
                let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
                _ = try NDRenderHarness.hover(store, station: station, name: "\(station.rawValue)-\(suffix)", appearance: appearance, in: directory)
            }
        }
    }

    func testDrawnBubblesPreserveCaptionInk() throws {
        let L = NDPipelineLayout.self
        let margin = CGPoint(x: 80, y: 160)
        let size = CGSize(width: 1000, height: 620)
        for (_, appearance) in NDRenderHarness.appearances {
            for (name, station): (BrainBarDesignFixture.Name, NDStation) in [(.calm, .agents), (.calm, .queue), (.searchFailing, .search)] {
                let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
                func render(_ hover: NDStation?) throws -> NSBitmapImageRep {
                    let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories,
                        items: store.r1.stationItems, interactive: false,
                        popover: { AnyView(NDStationPopover(store: store, station: $0)) }, previewHover: hover)
                    let view = ZStack(alignment: .topLeading) {
                        ND.card
                        chart.offset(x: margin.x, y: margin.y)
                    }.frame(width: size.width, height: size.height)
                    return try NDRenderHarness.host(view, size: size, appearance: appearance)
                }
                let plain = try render(nil), hovered = try render(station)
                var protected = L.laneLabelRects()
                for other in NDStation.allCases where other != station {
                    protected.append((other.rawValue + " icon", L.circle(other)))
                    protected += L.labelRects(other, store.r1.pipeline).map { (other.rawValue + " words", $0) }
                }
                protected += [
                    ("left card edge", CGRect(x: -60, y: 0, width: 43, height: L.cardInside.maxY + 24)),
                    ("bottom card edge", CGRect(x: -16, y: L.cardInside.maxY + 1, width: L.size.width + 32, height: 20)),
                ]
                for (words, rect) in protected {
                    var changed = 0
                    let area = rect.offsetBy(dx: margin.x, dy: margin.y)
                    for y in Int(area.minY * 2)..<Int(area.maxY * 2) {
                        for x in Int(area.minX * 2)..<Int(area.maxX * 2) {
                            let a = try XCTUnwrap(plain.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                            let b = try XCTUnwrap(hovered.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                            if max(abs(a.redComponent - b.redComponent), abs(a.greenComponent - b.greenComponent),
                                   abs(a.blueComponent - b.blueComponent)) > 0.08 { changed += 1 }
                        }
                    }
                    XCTAssertEqual(changed, 0, "\(station) must preserve drawn \(words)")
                }
            }
        }
    }
}
