import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Geometry and content only; no windows, menus or status items.
@MainActor final class BrainBarFirstLookBubbleTests: XCTestCase {
    func testOutlineHasOneClosedSubpathAndNoNotchBase() {
        let body = CGRect(x: 30, y: 30, width: 300, height: 180)
        for edge: Edge in [.top, .bottom, .leading, .trailing] {
            let tip: CGPoint = switch edge {
            case .top: CGPoint(x: body.midX, y: body.maxY + 8)
            case .bottom: CGPoint(x: body.midX, y: body.minY - 8)
            case .leading: CGPoint(x: body.maxX + 8, y: body.midY)
            case .trailing: CGPoint(x: body.minX - 8, y: body.midY)
            }
            let path = NDPanelShape(body: body, tip: tip, edge: edge).path(in: body)
            var moves = 0, closes = 0
            var previous = CGPoint.zero
            path.forEach { element in
                switch element {
                case .move(to: let p): moves += 1; previous = p
                case .closeSubpath: closes += 1
                case .line(to: let p):
                    let base = edge == .top ? body.maxY : body.minY
                    let horizontal = edge == .top || edge == .bottom
                    if horizontal && previous.y == base && p.y == base {
                        XCTAssertFalse(min(previous.x, p.x) < tip.x && max(previous.x, p.x) > tip.x)
                    } else if !horizontal {
                        let x = edge == .trailing ? body.minX : body.maxX
                        if previous.x == x && p.x == x {
                            XCTAssertFalse(min(previous.y, p.y) < tip.y && max(previous.y, p.y) > tip.y)
                        }
                    }
                    previous = p
                case .quadCurve(to: let p, control: _): previous = p
                case .curve(to: let p, control1: _, control2: _): previous = p
                }
            }
            XCTAssertEqual(moves, 1, "\(edge)")
            XCTAssertEqual(closes, 1, "\(edge)")
        }
    }

    func testEveryStationPointsAtItsCentreAndShiftsOnlyForClearance() {
        let L = NDPipelineLayout.self
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
            let state = store.r1.pipeline
            let chart = NDPipelineChart(state: state, memories: store.model.memories, items: store.r1.stationItems, backupProgress: store.inputs.backupProgress)
            for station in NDStation.allCases {
                let p = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                let horizontal = p.edge == .top || p.edge == .bottom
                XCTAssertEqual(horizontal ? p.tip.x : p.tip.y, horizontal ? L.circle(station).midX : L.circle(station).midY,
                               accuracy: 1, "\(station)")
                let desired = horizontal ? p.tip.x - p.body.width / 2 : p.tip.y - p.body.height / 2
                let placed = horizontal ? p.body.minX : p.body.minY
                let delta = desired - placed
                if abs(delta) > 1 {
                    let nearerBody = p.body.offsetBy(dx: horizontal ? (delta > 0 ? 1 : -1) : 0,
                                                 dy: horizontal ? 0 : (delta > 0 ? 1 : -1))
                    let arrow = horizontal
                        ? CGRect(x: p.tip.x - L.arrowHalfWidth, y: p.edge == .top ? p.tip.y - L.arrowLength : p.tip.y,
                                 width: 2 * L.arrowHalfWidth, height: L.arrowLength)
                        : CGRect(x: p.edge == .leading ? p.tip.x - L.arrowLength : p.tip.x, y: p.tip.y - L.arrowHalfWidth,
                                 width: L.arrowLength, height: 2 * L.arrowHalfWidth)
                    let nearer = nearerBody.union(arrow)
                    let obstacles = L.panelObstacles(station, state: state, actions: [], actionWidths: [:],
                                                    interior: p.clearanceException == nil, displayedWords: chart.displayedPanelWords)
                    let ceiling = station == .search ? L.keptSafe.maxY + 4 : L.activityLabelsBottom + 4
                    let blocked = nearer.minX < L.windowInside.minX || nearer.maxX > L.windowInside.maxX
                        || nearer.minY < ceiling || nearer.maxY > L.footerTop - 4
                        || obstacles.contains { nearer.intersects($0) }
                        || L.panelBodyObstacles(station, interior: p.clearanceException == nil).contains { nearerBody.intersects($0) }
                    XCTAssertTrue(blocked, "\(station): moving 1 pt nearer the centre must violate clearance")
                }
            }
        }
    }

    func testTopPanelsContainMeasuredContentInsideWindow() {
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems, backupProgress: store.inputs.backupProgress)
            for station: NDStation in [.agents, .read, .memory, .mac, .drive] {
                let content = AnyView(NDStationPopover(store: store, station: station))
                let panel = chart.panel(station, content: content)
                let measured = NDStationPopover.measure(content, width: panel.body.width).height
                let context = "\(name.rawValue) \(station) \(panel.body) measured=\(measured)"
                XCTAssertGreaterThanOrEqual(panel.body.height + 0.01, measured, context)
                XCTAssertFalse(panel.scrolls, context)
                XCTAssertFalse(panel.outer.intersects(NDPipelineLayout.pipelineHeading), context)
                if panel.clearanceException == .topRowBelowBothSidesBlocked {
                    XCTAssertEqual(panel.edge, .bottom, context)
                    for side: Edge in [.top, .trailing, .leading, .bottom] {
                        XCTAssertNil(NDPipelineLayout.panelCandidate(station, edge: side, state: store.r1.pipeline,
                                                                     actions: [], height: measured, width: panel.body.width,
                                                                     displayedWords: chart.displayedPanelWords), context)
                    }
                    for width in NDPipelineLayout.widenedPanelWidths(station) {
                        XCTAssertNil(NDPipelineLayout.panelCandidate(station, edge: .top, state: store.r1.pipeline,
                            actions: [], height: NDStationPopover.measure(content, width: width).height, width: width,
                            displayedWords: chart.displayedPanelWords), context)
                    }
                }
                XCTAssertGreaterThanOrEqual(panel.body.minY, NDPipelineLayout.activityLabelsBottom + 4, context)
                XCTAssertLessThanOrEqual(panel.outer.maxY, NDPipelineLayout.windowInside.maxY, context)
                XCTAssertGreaterThanOrEqual(panel.outer.minX, NDPipelineLayout.windowInside.minX, context)
                XCTAssertLessThanOrEqual(panel.outer.maxX, NDPipelineLayout.windowInside.maxX, context)
            }
        }
    }

    func testCalmUploadPanelClearsTheDisplayedProgressRow() throws {
        var inputs = BrainBarDesignFixture.inputs(.calm)
        inputs.backupProgress = try XCTUnwrap(BrainBarDesignFixture.inputs(.uploadProgress).backupProgress)
        let store = NDStore(fixture: inputs, page: .overview)
        let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories,
                                    items: store.r1.stationItems, backupProgress: inputs.backupProgress)
        let row = NDPipelineChart.progressFrame(store.r1.pipeline)
        XCTAssertEqual(row.minY, 81)
        XCTAssertEqual(row.maxY, 114)
        XCTAssertTrue(chart.displayedPanelWords[.drive]?.contains(row) == true)
        let panel = chart.panel(.drive, content: AnyView(NDStationPopover(store: store, station: .drive)))
        XCTAssertFalse(panel.outer.intersects(row), "the notch and body must clear the words actually drawn")
        XCTAssertFalse(panel.body.intersects(row))
        if panel.edge == .bottom {
            XCTAssertGreaterThanOrEqual(panel.tip.y, row.maxY)
            XCTAssertGreaterThan(panel.body.minY, row.maxY)
            XCTAssertEqual(panel.tip.x, NDPipelineLayout.circle(.drive).midX, accuracy: 1)
        }
    }

    func testL58bCalmPlacementCases() {
        let L = NDPipelineLayout.self
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
        let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
        for station: NDStation in [.agents, .read, .mac, .drive] {
            let content = AnyView(NDStationPopover(store: store, station: station))
            let normal = L.panelSide(station, hasAction: false).width
            func candidate(_ edge: Edge, width: CGFloat, overlap: Bool = false) -> NDPipelineLayout.Panel? {
                L.panelCandidate(station, edge: edge, state: store.r1.pipeline, actions: [],
                    height: NDStationPopover.measure(content, width: width).height, width: width,
                    allowsInteriorOverlap: overlap)
            }
            let aboveNormal = candidate(.top, width: normal)
            let aboveWidened = L.widenedPanelWidths(station).compactMap { candidate(.top, width: $0) }.first
            let expected: NDPipelineLayout.Panel? = aboveNormal ?? aboveWidened ?? candidate(.trailing, width: normal)
                ?? candidate(.leading, width: normal) ?? candidate(.bottom, width: normal)
                ?? candidate(.bottom, width: normal, overlap: true)
            let panel = chart.panel(station, content: content)
            if let expected {
                XCTAssertEqual(panel, expected, "\(station): normal above, widened above, right, left, below")
            }
            XCTAssertNil(panel.clearanceException, "L63 withdraws the interior-overlap exception")
            XCTAssertFalse(panel.scrolls)
        }
    }

    func testProviderLabelsCannotInterceptTheAgentsStation() throws {
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign/NDPipelineChart.swift"), encoding: .utf8)
        let start = try XCTUnwrap(source.range(of: "ForEach(NDPipelineLayout.agentRows"))
        let end = try XCTUnwrap(source.range(of: "if let previewHover { preview(previewHover) }"))
        let providers = source[start.lowerBound..<end.lowerBound]
        XCTAssertFalse(providers.contains(".help("))
        XCTAssertFalse(providers.contains("NDStationButton("))
        XCTAssertTrue(providers.contains(".allowsHitTesting(false)"))
        XCTAssertTrue(providers.contains(".accessibilityLabel(NDProvider.name(family))"))
        let state = NDPipelineState.make(BrainBarDesignFixture.inputs(.calm))
        let target = NDPipelineChart.stationTarget(.agents, state: state)
        let path = target.shape.path(in: CGRect(origin: .zero, size: target.frame.size))
        for x in stride(from: 0.5, through: target.frame.width - 0.5, by: 1) {
            for y in stride(from: 0.5, through: target.frame.height - 0.5, by: 1) {
                XCTAssertTrue(path.contains(CGPoint(x: x, y: y)), "\(x), \(y)")
            }
        }
    }
}
