import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// w307: running agents per harness in the Pipeline's agents column. A harness with N > 0 running agents is drawn in its
/// colour with N beside its icon; at 0 (or unmeasured) it stays neutral with no number. The hover panel lists every
/// harness with its count, VoiceOver reads "<harness>, N agents running", and a count never reaches a connector.
@MainActor final class BrainBarAgentsPerHarnessTests: XCTestCase {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    typealias L = NDPipelineLayout
    typealias Fixture = BrainBarDesignFixture

    private func inputs(_ sessions: [AgentFamily: Int?], fixture: Fixture.Name = .calm) -> NDInputs {
        BrainBarAgentsPerHarnessRenderTests.inputs(fixture, sessions)
    }

    private let mixed: [AgentFamily: Int?] = [.claude: 7, .codex: 2, .cursor: 0, .gemini: 1]
    private let zero: [AgentFamily: Int?] = [.claude: 0, .codex: 0, .cursor: 0, .gemini: 0]
    private let unmeasured: [AgentFamily: Int?] = [.claude: nil, .codex: nil, .cursor: nil, .gemini: nil]

    func testTheChartCarriesTheMonitorsCounts() {
        let state = NDPipelineState.make(inputs(mixed))
        XCTAssertEqual(state.agentsRunning, [.claude: 7, .codex: 2, .cursor: 0, .gemini: 1])
        XCTAssertEqual(NDPipelineState.make(inputs(unmeasured)).agentsRunning, [:], "not measured is not zero")
        // The count is BrainBar's own measurement, not the verdict's: a stale verdict keeps it.
        let stale = NDPipelineState.make(inputs(mixed, fixture: .notConfirmed))
        XCTAssertTrue(stale.allUnconfirmed)
        XCTAssertEqual(stale.agentsRunning[.claude], 7)
    }

    func testCountsAppearOnlyForRunningHarnesses() {
        let state = NDPipelineState.make(inputs(mixed))
        XCTAssertEqual(AgentFamily.allCases.map { NDPipelineChart.agentCount($0, state: state) }, [7, 2, nil, 1])
        for sessions in [zero, unmeasured] {
            let quiet = NDPipelineState.make(inputs(sessions))
            XCTAssertTrue(AgentFamily.allCases.allSatisfy { NDPipelineChart.agentCount($0, state: quiet) == nil })
        }
    }

    func testIconIsInItsHarnessColourOnlyWhileAgentsRun() {
        let state = NDPipelineState.make(inputs(mixed))
        XCTAssertEqual(AgentFamily.allCases.map { NDPipelineChart.agentTone($0, state: state) }, [.harness, .harness, .neutral, .harness])
        for sessions in [zero, unmeasured] {
            let quiet = NDPipelineState.make(inputs(sessions))
            XCTAssertTrue(AgentFamily.allCases.allSatisfy { NDPipelineChart.agentTone($0, state: quiet) == .neutral })
        }
        // A fault keeps its own treatment; the count stays beside it.
        let faulted = NDPipelineState.make(inputs([.claude: 4, .codex: 23, .cursor: 0, .gemini: 0], fixture: .agentNotRead))
        XCTAssertFalse(faulted.agentFaults.isEmpty, "the agent-not-read fixture names a provider")
        for (family, sev) in faulted.agentFaults {
            XCTAssertEqual(NDPipelineChart.agentTone(family, state: faulted), .fault(sev))
        }
        XCTAssertEqual(NDPipelineChart.agentCount(.codex, state: faulted), 23)
    }

    func testVoiceOverReadsEachHarnessWithItsCount() {
        let state = NDPipelineState.make(inputs(mixed))
        XCTAssertEqual(AgentFamily.allCases.map { NDPipelineChart.providerLabel($0, state: state) },
                       ["Claude, 7 agents running", "Codex, 2 agents running", "Cursor, no agents running", "Antigravity, 1 agent running"])
        let unknown = NDPipelineState.make(inputs(unmeasured))
        XCTAssertEqual(NDPipelineChart.providerLabel(.codex, state: unknown), "Codex, running agents not known")
    }

    func testTheHoverPanelListsEveryHarnessWithItsCount() {
        let i = inputs(mixed)
        let rows = NDStationPopover.rows(.agents, i, NDModel.make(i))
        XCTAssertEqual(rows.map(\.label), ["Claude", "Codex", "Cursor", "Antigravity"])
        XCTAssertEqual(rows.map(\.value), ["7 running", "2 running", "none running", "1 running"])
        let u = inputs(unmeasured)
        XCTAssertEqual(NDStationPopover.rows(.agents, u, NDModel.make(u)).map(\.value), Array(repeating: "not known", count: 4))
        // The calm panel carries them; a fault panel stays the fault's (title, who is still read, Copy report), which has
        // no room for four more rows under the column. The counts stay on the icons there.
        let calm = NDStore(fixture: inputs(mixed), page: .overview)
        let explain = NDStationPopover.model(.agents, item: calm.r1.stationItems[.agents], calm.inputs, calm.model, calm.r1.pipeline)
        XCTAssertEqual(explain.rows.map(\.value), ["7 running", "2 running", "none running", "1 running"])
        let store = NDStore(fixture: inputs(mixed, fixture: .agentNotRead), page: .overview)
        let fault = NDStationPopover.model(.agents, item: store.r1.stationItems[.agents], store.inputs, store.model, store.r1.pipeline)
        XCTAssertNotNil(fault.action, "the fault panel")
        XCTAssertEqual(fault.rows, [])
    }

    /// The connectors' strokes (at their widest, 2.5 pt) and their arrowheads, as rectangles.
    private func connectorRects() -> [(String, CGRect)] {
        var out: [(String, CGRect)] = []
        for c in L.connections {
            let pts = c.line.points
            for (a, b) in zip(pts, pts.dropFirst()) {
                out.append(("\(c.pipe)", CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y))
                    .insetBy(dx: -1.25, dy: -1.25)))
            }
            if let first = pts.first, let last = pts.last {
                out.append(("\(c.pipe) end", CGRect(x: last.x - 6, y: last.y - 6, width: 12, height: 12)))
                if c.bothWays { out.append(("\(c.pipe) start", CGRect(x: first.x - 6, y: first.y - 6, width: 12, height: 12))) }
            }
        }
        return out
    }

    func testACountNeverReachesAConnectorAnIconOrAnotherCount() {
        let lines = connectorRects()
        XCTAssertFalse(lines.isEmpty)
        let icons = L.agentRows.map { family, _ -> CGRect in
            let c = L.iconCentre(family)
            return CGRect(x: c.x - L.iconSize / 2, y: c.y - L.iconSize / 2, width: L.iconSize, height: L.iconSize)
        }
        for count in [7, 42, 999] {
            for faulted in [false, true] {
                let masks = L.agentRows.map { L.countMask($0.0, count: count, faulted: faulted) }
                for (index, (family, _)) in L.agentRows.enumerated() {
                    let mask = masks[index]
                    let label = "\(family) \(count) faulted \(faulted)"
                    for (name, line) in lines { XCTAssertFalse(mask.intersects(line), "\(label) reaches \(name)") }
                    for icon in icons { XCTAssertFalse(mask.intersects(icon), "\(label) covers an icon") }
                    for (j, other) in masks.enumerated() where j != index {
                        XCTAssertFalse(mask.intersects(other), "\(label) touches another count")
                    }
                    XCTAssertGreaterThanOrEqual(mask.minX, 0)
                    XCTAssertLessThan(mask.maxX, L.point(.read).x - L.lineEnd, "\(label) stays left of Read")
                    if faulted {
                        let c = L.iconCentre(family)
                        let ring = CGRect(x: c.x - 12.75, y: c.y - 12.75, width: 25.5, height: 25.5)
                        XCTAssertGreaterThanOrEqual(L.countRect(family, count: count, faulted: true).minX, ring.maxX + 2, "\(label) clears the ring")
                    }
                }
            }
        }
        // The counts are the only new thing: the column and every connector are where they were.
        XCTAssertEqual(L.agentsFrame, CGRect(x: 16, y: 18, width: 44, height: 108))
        XCTAssertEqual(L.agentRows.map(\.1), [36, 60, 84, 108])
    }

    func testACountUsesTheChartsFigureStyle() {
        XCTAssertEqual(L.countText(1234), "1234")
        // Monospaced figures: every 3-digit count is as wide as every other.
        XCTAssertEqual(L.countRect(.claude, count: 111, faulted: false).width, L.countRect(.claude, count: 888, faulted: false).width)
        XCTAssertLessThan(L.countRect(.claude, count: 7, faulted: false).width, L.countRect(.claude, count: 77, faulted: false).width)
    }

    /// WCAG relative luminance of a dynamic colour in an appearance.
    private func luminance(_ color: NSColor, _ appearance: NSAppearance.Name) -> CGFloat {
        var out: CGFloat = 0
        NSAppearance(named: appearance)!.performAsCurrentDrawingAppearance {
            let c = color.usingColorSpace(.sRGB)!
            func lin(_ v: CGFloat) -> CGFloat { v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4) }
            out = 0.2126 * lin(c.redComponent) + 0.7152 * lin(c.greenComponent) + 0.0722 * lin(c.blueComponent)
        }
        return out
    }

    private func contrast(_ a: NSColor, _ b: NSColor, _ appearance: NSAppearance.Name) -> CGFloat {
        let x = luminance(a, appearance), y = luminance(b, appearance)
        return (max(x, y) + 0.05) / (min(x, y) + 0.05)
    }

    func testHarnessColoursHoldTheirContrast() {
        for appearance in [NSAppearance.Name.darkAqua, .aqua] {
            var seen: Set<String> = []
            for family in AgentFamily.allCases {
                let color = NDColor.harness(family)
                // The number's bar, 4.5 : 1, which also covers the icon's 3 : 1.
                XCTAssertGreaterThanOrEqual(contrast(color, NDColor.card, appearance), 4.5, "\(family) on the card, \(appearance.rawValue)")
                // Distinct from each other and from the fault and notice colours they sit beside.
                NSAppearance(named: appearance)!.performAsCurrentDrawingAppearance {
                    let c = color.usingColorSpace(.sRGB)!
                    seen.insert(String(format: "%.2f %.2f %.2f", c.redComponent, c.greenComponent, c.blueComponent))
                }
                for other in [NDColor.red, NDColor.orange, NDColor.focus] {
                    NSAppearance(named: appearance)!.performAsCurrentDrawingAppearance {
                        let a = color.usingColorSpace(.sRGB)!, b = other.usingColorSpace(.sRGB)!
                        let hueGap = abs(a.hueComponent - b.hueComponent)
                        XCTAssertGreaterThanOrEqual(min(hueGap, 1 - hueGap), 10.0 / 360, "\(family) is not read as \(other)")
                    }
                }
            }
            XCTAssertEqual(seen.count, AgentFamily.allCases.count, "four distinct colours in \(appearance.rawValue)")
        }
    }
}

/// w307: the agents panel with its four count rows opens whole under the column (no scrolling), in every fixture.
@MainActor final class BrainBarAgentsPanelFitTests: XCTestCase {
    func testTheAgentsPanelFitsWithItsRows() {
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: nil, items: store.r1.stationItems, interactive: false)
            let panel = chart.panel(.agents, content: AnyView(NDStationPopover(store: store, station: .agents)))
            XCTAssertFalse(panel.scrolls, "\(name.rawValue): the agents panel is cut short")
        }
    }
}
