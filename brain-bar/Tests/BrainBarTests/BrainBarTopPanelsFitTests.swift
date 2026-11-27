import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// w303: the two copy stations' hover panels (Local copy, Google Drive) are ordinary content-fit popovers: at most 320 pt
/// wide, stacked, opening upward from their own station over the bottom of the Activity card. Before, the room above the
/// top row (R1.12 § B kept every panel out of the Activity card) was too short for the stacked body, so the layout widened
/// it into a one-row strip at least 760 pt wide that covered the Pipeline heading's row and the "Session backups" line.
/// Also: no panel's button is cut short (the Nightly check-up's "Copy a report for your agent" read "Copy a report f…").
@MainActor
final class BrainBarTopPanelsFitTests: XCTestCase {
    private typealias L = NDPipelineLayout

    /// The top of the Activity card's plot in chart coordinates, from the card's own geometry: its bottom edge, 16 pt
    /// padding, the axis row (4 + 14), the plot with its now tick (112 + 8) and the 14 pt above the plot. Above it sit the
    /// card's totals row and its heading, which a panel never covers.
    static var activityPlotTop: CGFloat {
        L.activityBottom - 16 - 18 - (NDActivitySection.plotHeight + 8) - 14
    }

    private func chart(_ store: NDStore) -> NDPipelineChart {
        NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems,
                        backupProgress: store.inputs.backupProgress)
    }

    /// Every designed fixture, plus Drive uploading at 45 % on a calm day.
    private func stores() throws -> [(String, NDStore)] {
        var out = BrainBarDesignFixture.Name.allCases.map { ($0.rawValue, NDStore(fixture: BrainBarDesignFixture.inputs($0), page: .overview)) }
        var uploading = BrainBarDesignFixture.inputs(.calm)
        var progress = try XCTUnwrap(BrainBarDesignFixture.inputs(.uploadProgress).backupProgress)
        progress.confirmed = progress.total.map { $0 * 45 / 100 }
        uploading.backupProgress = progress
        out.append(("calm-uploading-45", NDStore(fixture: uploading, page: .overview)))
        return out
    }

    func testCopyPanelsFitTheirContentAndOpenAboveTheirStation() throws {
        for (name, store) in try stores() {
            let chart = chart(store)
            let state = NDPipelineChart.drawingState(store.r1.pipeline, progress: store.inputs.backupProgress)
            for station: NDStation in [.mac, .drive] {
                let content = AnyView(NDStationPopover(store: store, station: station))
                let panel = chart.panel(station, content: content)
                let context = "\(name) \(station) body=\(panel.body)"
                XCTAssertLessThanOrEqual(panel.body.width, 320, "\(context): a popover, not a strip")
                XCTAssertEqual(panel.edge, .top, "\(context): opens upward (below lands on Weekly clean's words)")
                XCTAssertEqual(panel.tip.x, L.circle(station).midX, accuracy: 0.5, "\(context): the arrow is on the station")
                XCTAssertFalse(panel.scrolls, context)
                let measured = NDStationPopover.measure(content, width: panel.body.width).height
                XCTAssertGreaterThanOrEqual(panel.body.height + 0.01, measured, "\(context): holds all of its content")
                // Above the station: clear of its own words and of every station on the Pipeline, words included.
                XCTAssertLessThanOrEqual(panel.body.maxY, L.circle(station).minY, context)
                let words = chart.displayedPanelWords[station] ?? L.labelRects(station, state)
                for rect in words { XCTAssertFalse(panel.outer.intersects(rect), "\(context): covers its own words \(rect)") }
                for other in NDStation.allCases where other != station {
                    XCTAssertFalse(panel.body.intersects(L.circle(other)), "\(context): covers \(other)")
                    for rect in L.labelRects(other, state) {
                        XCTAssertFalse(panel.outer.intersects(rect), "\(context): covers \(other)'s words \(rect)")
                    }
                }
                XCTAssertFalse(panel.outer.intersects(L.pipelineHeading), "\(context): covers the Pipeline heading")
                for (_, caption) in L.laneLabelRects() { XCTAssertFalse(panel.outer.intersects(caption), "\(context): covers a route caption") }
                // Over the bottom of the Activity card, never its totals row or anything above it.
                XCTAssertGreaterThanOrEqual(panel.body.minY, Self.activityPlotTop, context)
                XCTAssertGreaterThanOrEqual(panel.outer.minX, L.windowInside.minX, context)
                XCTAssertLessThanOrEqual(panel.outer.maxX, L.windowInside.maxX, context)
            }
        }
    }

    /// The copy panels never draw the one-row strip: the stacked body is what they measure and show.
    func testCopyPanelsUseTheStackedBody() throws {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
        for station: NDStation in [.mac, .drive] {
            let content = AnyView(NDStationPopover(store: store, station: station))
            let strip = NDStationPopover.measure(content, width: 800).height
            let stacked = NDStationPopover.measure(content, width: 320).height
            // At 800 pt the strip would be one row (~50 pt); the stacked body keeps its heading, lines and rows on their own
            // lines at any width, so it is no shorter at 800 than its rows alone.
            XCTAssertGreaterThan(strip, 80, "\(station): no one-row strip even with room for one (strip=\(strip))")
            XCTAssertLessThanOrEqual(stacked, 240, "\(station) stacked=\(stacked)")
        }
    }

    /// Every panel's button shows its whole label at the width the panel opens at: on one line, or, for Copy report in a
    /// panel too narrow for that, on two centred lines across the panel's width (no word broken, nothing cut).
    func testNoPanelButtonIsCutShort() throws {
        for (name, store) in try stores() {
            let chart = chart(store)
            for station in NDStation.allCases {
                let model = NDStationPopover.model(station, item: store.r1.stationItems[station], store.inputs, store.model,
                                                   store.r1.pipeline)
                guard let action = model.action else { continue }
                let panel = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                // The button sits inside the body's 14 pt side padding; its label has 10 pt each side.
                let room = panel.body.width - 28 - 20
                let label = NDPipelineChart.shortLabel(action)
                let context = "\(name) \(station): \"\(label)\" in a \(panel.body.width) pt panel"
                // Copy report reserves the wider of its two faces: its label, and the tick (12 pt + 4) with "Copied".
                let copied = L.textWidth(BrainBarMenuPresenter.SurfaceSay.copiedLabel, size: 13, weight: .medium) + 16
                XCTAssertLessThanOrEqual(copied, room, context)
                let lines = Self.wrap(label, width: room)
                XCTAssertLessThanOrEqual(lines.count, action.kind == .copyReport ? 2 : 1, "\(context) wraps to \(lines)")
            }
        }
    }

    /// The label set at 13 pt Medium, broken between words at `width`: a word wider than the width stays one (cut) line.
    static func wrap(_ text: String, width: CGFloat) -> [String] {
        var lines: [String] = []
        var line = ""
        for word in text.split(separator: " ").map(String.init) {
            let candidate = line.isEmpty ? word : line + " " + word
            if line.isEmpty || L.textWidth(candidate, size: 13, weight: .medium) <= width {
                line = candidate
            } else {
                lines.append(line)
                line = word
            }
            if L.textWidth(line, size: 13, weight: .medium) > width { return lines + [line, "(cut)"] }
        }
        return lines + [line]
    }

    /// The wrapped button really lays out on two lines, whole, at the narrowest panel that carries it (160 pt).
    func testTwoLineCopyReportShowsTheWholeLabel() throws {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.nightlyFailing), page: .overview)
        let narrow = NDStationPopover.measure(AnyView(NDStationPopover(store: store, station: .jobs)), width: 160).height
        let wide = NDStationPopover.measure(AnyView(NDStationPopover(store: store, station: .jobs)), width: 400).height
        XCTAssertGreaterThan(narrow, wide, "the narrow panel is taller (more lines), never clipped")
        let button = NDStationPopover.measure(AnyView(NDCopyReportButton(label: "Copy a report for your agent", spoken: "",
                                                                         lines: 2) {}), width: 132)
        XCTAssertGreaterThanOrEqual(button.height, 2 * 15, "two lines of 13 pt (button=\(button))")
    }

    /// w303 renders (offscreen, NDRenderHarness's host): the Activity card, the Pipeline heading and the Pipeline card
    /// laid out as the Overview lays them out, with one station's panel drawn at its layout frame. Runs only when
    /// `BRAINBAR_W303_RENDER_DIR` names an absolute directory.
    func testRenderTopPanelsForReview() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_W303_RENDER_DIR"], !path.isEmpty else {
            throw XCTSkip("set BRAINBAR_W303_RENDER_DIR to render")
        }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var uploading = BrainBarDesignFixture.inputs(.calm)
        var progress = try XCTUnwrap(BrainBarDesignFixture.inputs(.uploadProgress).backupProgress)
        progress.confirmed = progress.total.map { $0 * 45 / 100 }
        uploading.backupProgress = progress
        let cases: [(String, NDInputs, NDStation)] = [
            ("local-copy-calm", BrainBarDesignFixture.inputs(.calm), .mac),
            ("drive-calm", BrainBarDesignFixture.inputs(.calm), .drive),
            ("drive-uploading-45", uploading, .drive),
            ("drive-error-signed-out", BrainBarDesignFixture.inputs(.driveSignedOut), .drive),
            ("nightly-failing", BrainBarDesignFixture.inputs(.nightlyFailing), .jobs),
            ("nightly-calm", BrainBarDesignFixture.inputs(.calm), .jobs),
        ]
        for (label, inputs, station) in cases {
            for (suffix, appearance) in NDRenderHarness.appearances {
                let store = NDStore(fixture: inputs, page: .overview)
                let r1 = store.r1
                let width = NDGrid.windowWidth
                let size = NSSize(width: width, height: 620)
                let view = ZStack(alignment: .topLeading) {
                    ND.bg
                    VStack(alignment: .leading, spacing: 0) {
                        NDActivitySection(store: store)
                        NDBlockHeading(title: BrainBarMenuPresenter.SurfaceSay.pipeline).padding(.top, NDGrid.activityToPipeline)
                        NDCard(padding: EdgeInsets(top: L.verticalPadding, leading: 16, bottom: L.verticalPadding, trailing: 16)) {
                            NDPipelineChart(state: r1.pipeline, memories: store.model.memories, items: r1.stationItems, interactive: false,
                                            popover: { s in AnyView(NDStationPopover(store: store, station: s)) }, previewHover: station,
                                            backupProgress: store.inputs.backupProgress)
                        }
                    }
                    .frame(width: width - 2 * NDGrid.pageSide)
                    .offset(x: NDGrid.pageSide, y: 20)
                }
                .frame(width: size.width, height: size.height)
                let name = "\(label)-\(station.rawValue)-\(suffix)"
                try NDRenderHarness.write(NDRenderHarness.host(view, size: size, appearance: appearance), name: name, in: directory)
                let chart = NDPipelineChart(state: r1.pipeline, memories: store.model.memories, items: r1.stationItems,
                                            backupProgress: store.inputs.backupProgress)
                let panel = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                print("[w303-render] \(name) edge=\(panel.edge) body=\(panel.body) outer=\(panel.outer)")
            }
        }
    }
}
