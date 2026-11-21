import AppKit
import XCTest
@testable import BrainBar

/// Owner-desk safe: pure presenters, layout data and offscreen image contexts only.
@MainActor
final class BrainBarGapClarityTests: XCTestCase {
    typealias R113 = BrainBarNewDesignR113Tests
    typealias Say = BrainBarMenuPresenter.Say

    /// Gap 79 / L56: reproduce the geometry pin without running R1's window-owning test class on the owner's desk.
    func testJobLinksAndChartHeightAfterPurposeLines() throws {
        typealias L = NDPipelineLayout
        for (pipe, station) in [(NDPipe.jobsMemory, NDStation.jobs), (.weeklyMemory, .weekly)] {
            let link = try XCTUnwrap(L.connections.first { $0.pipe == pipe })
            let anchorX = station == .jobs ? L.hub.minX + 12 : L.hub.maxX - 12
            XCTAssertTrue(link.bothWays)
            XCTAssertEqual(link.line.points.first?.x, L.point(station).x)
            XCTAssertEqual(link.line.points.last, CGPoint(x: anchorX, y: L.hub.maxY + L.edgeGap))
            for (a, b) in zip(link.line.points, link.line.points.dropFirst()) {
                XCTAssertTrue(a.x == b.x || a.y == b.y, "orthogonal elbow")
            }
        }
        // The compact M1 layout still fits the lowest station's complete status line.
        XCTAssertEqual(L.size.height, 345) // R14: wider full-text panels reclaim 9 pt; L57 fits the page.
        for fixture in BrainBarDesignFixture.Name.allCases {
            let p = NDPipelineState.make(BrainBarDesignFixture.inputs(fixture))
            for station in NDStation.allCases {
                for rect in L.labelRects(station, p) { XCTAssertLessThanOrEqual(rect.maxY, L.size.height) }
            }
        }
    }

    func testDrainingLagNeverAsksForAnAgentOnEitherSeveritySetting() throws {
        for severity: BrainBarMenuPresenter.LagSeverity in [.needsYouPastLimit, .noticeWhileDraining] {
            let c = BrainBarNewDesignR111Tests.lag(clock: 2400, severity: severity)
            let i = NDInputs(capture: c), m = NDModel.make(i), r = NDR1.make(i, m)
            let item = try XCTUnwrap(r.items.first { $0.id == "reader" })
            XCTAssertNil(item.action)
            XCTAssertFalse(item.detail?.contains(Say.agent.text) == true)
            XCTAssertFalse(m.presentation.rows.flatMap(\.strings).contains(Say.agent.text))
            // This action fix never changes the severity decision, including the historical red side.
            XCTAssertEqual(r.redCount, severity == .needsYouPastLimit ? 1 : 0)
        }
    }

    func testRestingCaptionsAndPurposeLinesKeepTheirOwnSpace() throws {
        typealias L = NDPipelineLayout
        typealias Deck = BrainBarMenuPresenter.SurfaceSay
        let captions = L.laneLabelRects()
        XCTAssertEqual(captions.map(\.0), [Deck.sessionBackups, Deck.searchAnswers])
        XCTAssertEqual(captions.map { $0.1.midY }, [L.yRail, L.yAnswers])
        let nightly = try XCTUnwrap(L.purposeRect(.jobs)), weekly = try XCTUnwrap(L.purposeRect(.weekly))
        XCTAssertGreaterThanOrEqual(weekly.minX - nightly.maxX, 16)
        for fixture in BrainBarDesignFixture.Name.allCases {
            let i = BrainBarDesignFixture.inputs(fixture), p = NDPipelineState.make(i)
            for s: NDStation in [.jobs, .weekly] {
                let purpose = try XCTUnwrap(L.purposeRect(s))
                XCTAssertGreaterThanOrEqual(L.statusTop(s) - purpose.maxY, 4)
                XCTAssertLessThan(L.statusTop(s) + 13, L.yAnswers - 8)
                for other in NDStation.allCases where other != s {
                    for rect in L.labelRects(other, p) { XCTAssertFalse(rect.intersects(purpose), "\(s) vs \(other)") }
                }
                // Pure layout check; actual panel sizing and interaction stay on M1.
                for panelStation: NDStation in [.queue, .jobs, .weekly, .search] {
                    let panel = L.panel(panelStation, state: p, actions: [], height: 100)
                    XCTAssertFalse(panel.body.intersects(purpose), "\(panelStation) covers \(s)'s purpose")
                }
            }
        }
    }

    func testMenuMarkContrastFromRenderedPixels() throws {
        for dark in [false, true] {
            // The menu bar can be light over wallpaper while the app is dark (and conversely).
            let appearance = try XCTUnwrap(NSAppearance(named: dark ? .aqua : .darkAqua))
            let barAppearance = try XCTUnwrap(NSAppearance(named: dark ? .darkAqua : .aqua))
            for fixture: BrainBarDesignFixture.Name in [.calm, .notConfirmed, .meaningBehind, .driveSignedOut, .twoFaults] {
                let i = BrainBarDesignFixture.inputs(fixture), m = NDModel.make(i), r = NDR1.make(i, m)
                let rep = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 100, pixelsHigh: 44,
                    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
                rep.size = NSSize(width: 50, height: 22)
                let context = try XCTUnwrap(NSGraphicsContext(bitmapImageRep: rep))
                NSGraphicsContext.saveGraphicsState()
                NSGraphicsContext.current = context
                appearance.performAsCurrentDrawingAppearance {
                    let image = NDIcon.image(for: m, r, appearance: barAppearance)
                    image.draw(in: NSRect(origin: .zero, size: image.size))
                    if image.isTemplate {
                        (dark ? NSColor.white : NSColor.black).setFill()
                        NSRect(origin: .zero, size: image.size).fill(using: .sourceIn)
                    }
                }
                context.flushGraphics()
                NSGraphicsContext.restoreGraphicsState()
                func luminance(_ v: Double) -> Double { v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4) }
                let bar = dark ? 0.11 : 0.96
                let bg = luminance(bar)
                func contrast(in range: Range<Int>) throws -> Double {
                    var values: [Double] = []
                    for x in range {
                        for y in 0..<44 {
                            let c = try XCTUnwrap(rep.colorAt(x: x, y: y)?.usingColorSpace(.sRGB))
                            let a = Double(c.alphaComponent)
                            guard a > 0.4 else { continue }
                            let rgb = [c.redComponent, c.greenComponent, c.blueComponent].map { luminance(Double($0) * a + bar * (1 - a)) }
                            let l = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
                            values.append((max(bg, l) + 0.05) / (min(bg, l) + 0.05))
                        }
                    }
                    XCTAssertGreaterThan(values.count, 10)
                    // Median of the strongest half discards antialiased edge pixels, without counting the badge.
                    let sorted = values.sorted()
                    return sorted.isEmpty ? 0 : sorted[sorted.count * 3 / 4]
                }
                let brain = try contrast(in: 2..<24)
                XCTAssertGreaterThanOrEqual(brain, 3, "\(fixture) dark=\(dark)")
                if r.redCount > 1 {
                    let count = try contrast(in: 50..<90)
                    XCTAssertGreaterThanOrEqual(count, 3, "count dark=\(dark)")
                }
            }
        }
    }

    func testUnnamedRedLagOffersReportAcrossBothDesigns() throws {
        var c = R113.capture(ingest: .red, reasons: ["coverage_drop"],
                            catchUp: .init(state: "catching_up", bytesBehind: R113.mb117))
        c.watcher?.dbProbeFailed = true
        let p = BrainBarMenuPresenter.present(c)
        let row = try XCTUnwrap(p.rows.first { $0.kind == .needsYou })
        XCTAssertEqual(row.strings.first, Say.lagBehind.text)
        XCTAssertEqual(row.action?.kind, .copyReport)
        XCTAssertTrue(row.strings.contains(Say.agent.text))
        let i = NDInputs(capture: c), m = NDModel.make(i)
        let r = NDR1.make(i, m)
        let item = try XCTUnwrap(r.items.first { $0.concerns.contains("ingest") })
        XCTAssertEqual(item.title, row.strings.first)
        XCTAssertEqual(item.action?.kind, .copyReport)
        XCTAssertTrue(item.detail?.contains(Say.agent.text) == true)
        XCTAssertEqual(r.redCount, 1)
    }

    func testAmberIngestKeepsClassicWordsAcrossNewSurfaces() throws {
        for reasons in [["file_ingestion_failure"], [String]()] {
            var c = R113.capture(ingest: .amber, reasons: reasons,
                                catchUp: .init(state: "catching_up", bytesBehind: R113.mb117))
            if reasons.isEmpty {
                let checks = c.verdict!.checks.map { check -> BrainBarHealthVerdict.Check in
                    var k = check
                    if k.name == "ingest" { k.contextCodes = ["unindexed_agent_running"] }
                    return k
                }
                c.verdict = .init(dot: .amber, checks: checks, notes: [], evaluatedAt: c.now)
            }
            let i = NDInputs(capture: c), m = NDModel.make(i)
            let r = NDR1.make(i, m)
            let item = try XCTUnwrap(r.items.first { $0.concerns.contains("ingest") })
            let expected = reasons.isEmpty ? Say.lagBehind.text : Say.unreadFiles.text
            XCTAssertEqual(item.title, expected)
            XCTAssertEqual(item.menuTitle ?? item.title, expected)
            XCTAssertEqual(r.headline, expected)
            XCTAssertEqual(item.severity, .orange)
            XCTAssertNil(item.action)
            XCTAssertEqual(r.redCount, 0)
        }
    }
}
