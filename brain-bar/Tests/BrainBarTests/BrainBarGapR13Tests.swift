import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Offscreen native text and hosting measurements only; no windows or panels are opened.
@MainActor final class BrainBarGapR13Tests: XCTestCase {
    func testPurposeFieldsFitTheirNativeDrawingWidth() throws {
        let view = NDPipelineLabelView()
        for station: NDStation in [.jobs, .weekly] {
            let words = try XCTUnwrap(NDPipelineLayout.purpose(station))
            let field = try XCTUnwrap(view.subviews.compactMap { $0 as? NSTextField }.first { $0.stringValue == words })
            XCTAssertGreaterThanOrEqual(field.frame.width, ceil(field.fittingSize.width), words)
            XCTAssertGreaterThanOrEqual(field.frame.height, ceil(field.fittingSize.height), words)
            let font = try XCTUnwrap(field.font)
            let glyphs = NSAttributedString(string: words, attributes: [.font: font]).size()
            XCTAssertGreaterThanOrEqual(field.frame.width, ceil(glyphs.width) + 4, "native cell insets")
        }
    }

    func testJobPanelsOpenBelowWordsClearCaptionsAndFitAllTheirText() throws {
        typealias L = NDPipelineLayout
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name))
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
            for station: NDStation in [.jobs, .weekly] {
                let content = AnyView(NDStationPopover(store: store, station: station))
                let panel = chart.panel(station, content: content)
                let measured = NDStationPopover.measure(content, width: panel.body.width)
                let context = "\(name.rawValue) \(station) measured=\(measured) panel=\(panel.body)"
                XCTAssertEqual(panel.edge, .bottom, context)
                XCTAssertFalse(panel.scrolls, context)
                XCTAssertGreaterThanOrEqual(panel.body.height + 0.01, measured.height, context)
                XCTAssertLessThanOrEqual(panel.body.maxY, L.size.height, context)
                XCTAssertGreaterThan(panel.body.minY, L.labelRects(station, store.r1.pipeline).map(\.maxY).max() ?? 0, context)
                for other in NDStation.allCases {
                    for rect in L.labelRects(other, store.r1.pipeline) { XCTAssertFalse(panel.outer.intersects(rect), context) }
                    XCTAssertFalse(panel.body.intersects(L.circle(other)), context)
                }
                for (_, rect) in L.laneLabelRects() { XCTAssertFalse(panel.outer.intersects(rect.insetBy(dx: -4, dy: -4)), context) }
            }
        }
    }
}
