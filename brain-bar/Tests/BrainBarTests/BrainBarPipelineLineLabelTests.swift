import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// The two route captions ("Session backups" on the rail to Google Drive, "Search answers" on the return to the agents)
/// sit ON their lines: the line passes behind the words' vertical centre, through a card-coloured knock-out a few points
/// wider than the words, and the words are centred on the line's straight run. Checked as layout and as drawn pixels,
/// dark and light. The straight runs are computed here from the drawn connections, independently of the layout's helper.
@MainActor final class BrainBarPipelineLineLabelTests: XCTestCase {
    typealias L = NDPipelineLayout
    typealias Say = BrainBarMenuPresenter.SurfaceSay

    /// The straight part of a pipe's longest segment: between the ends of its rounded corners, each corner's radius
    /// clamped to half of its two segments (as `NDPolyline.path` draws it).
    static func run(_ pipe: NDPipe) throws -> (start: CGPoint, end: CGPoint) {
        let c = try XCTUnwrap(L.connections.first { $0.pipe == pipe })
        let p = c.line.points
        func len(_ k: Int) -> CGFloat { hypot(p[k].x - p[k - 1].x, p[k].y - p[k - 1].y) }
        let i = try XCTUnwrap((1..<p.count).max { len($0) < len($1) })
        func radius(at k: Int) -> CGFloat { k == 0 || k == p.count - 1 ? 0 : min(c.corner, len(k) / 2, len(k + 1) / 2) }
        let a = p[i - 1], b = p[i], l = len(i)
        let s = radius(at: i - 1) / l, e = radius(at: i) / l
        return (CGPoint(x: a.x + (b.x - a.x) * s, y: a.y + (b.y - a.y) * s), CGPoint(x: b.x - (b.x - a.x) * e, y: b.y - (b.y - a.y) * e))
    }

    static let captions: [(String, NDPipe)] = [(Say.sessionBackups, .rail), (Say.searchAnswers, .answers)]

    /// Where both captions' centres belong, one shared column so they read as a pair: the middle of their straight runs,
    /// moved left only as far as the job panels force. Nightly check-up's panel (its arrow included) opens beside its
    /// purpose line and crosses the return line, so the knock-out of "Search answers" may end no further right than that
    /// line ends.
    static func expectedMidX(_ words: String, _ pipe: NDPipe) throws -> CGFloat {
        let mids = try captions.map { try run($0.1) }.map { ($0.start.x + $0.end.x) / 2 }
        let edge = try XCTUnwrap(L.purposeRect(.jobs)).maxX
        return min(mids.min() ?? 0, edge - 6 - L.textWidth(Say.searchAnswers, size: 11, weight: .medium) / 2)
    }

    /// The knock-out the brief asks for: 6 pt past the words on each side, 16 pt tall, centred on the words.
    static func knockout(_ words: CGRect) -> CGRect {
        CGRect(x: words.minX - 6, y: words.midY - 8, width: words.width + 12, height: 16)
    }

    /// Layout: each caption's centre lies on its own line segment within 0.5 pt, and both share one column (x 348 today:
    /// the runs' middle, moved left only as far as the job panels force), so they align vertically.
    func testEachCaptionIsCentredOnItsOwnLine() throws {
        let rects = Dictionary(uniqueKeysWithValues: L.laneLabelRects())
        XCTAssertEqual(rects.count, 2)
        let top = try XCTUnwrap(rects[Say.sessionBackups]), bottom = try XCTUnwrap(rects[Say.searchAnswers])
        XCTAssertEqual(top.midX, bottom.midX, accuracy: 0.01, "the two route captions share one column")
        XCTAssertEqual(top.midX, 348, accuracy: 0.5, "the shared column is the forced Search answers centre")
        for (words, pipe) in Self.captions {
            let rect = try XCTUnwrap(rects[words], words)
            let run = try Self.run(pipe)
            XCTAssertEqual(run.start.y, run.end.y, "\(words): its run is horizontal")
            XCTAssertEqual(rect.midY, run.start.y, accuracy: 0.5, "\(words): the line passes through the words' centre")
            XCTAssertEqual(rect.midX, try Self.expectedMidX(words, pipe), accuracy: 0.5, "\(words): on the shared column")
            XCTAssertGreaterThan(rect.minX, min(run.start.x, run.end.x) + 16, "\(words): the line shows on its left")
            XCTAssertLessThan(rect.maxX, max(run.start.x, run.end.x) - 16, "\(words): the line shows on its right")
        }
    }

    /// The knock-out is a few points wider than the words on each side (the line visibly enters and leaves it), is
    /// centred on the line, and clears every station, frame and the not-confirmed caption at its widest.
    func testKnockoutIsAFewPointsWiderThanTheWordsAndClearsTheChart() throws {
        let unconfirmedWidest = L.textWidth(Say.valuesLastKnownAt("12:58 PM"), size: 11, weight: .medium)
        let unconfirmedMask = CGRect(x: L.size.width - unconfirmedWidest - 6, y: -2, width: unconfirmedWidest + 12, height: 16)
        for (words, rect) in L.laneLabelRects() {
            let k = L.laneLabelKnockout(rect)
            XCTAssertEqual(k, Self.knockout(rect), words)
            XCTAssertEqual(rect.minX - k.minX, k.maxX - rect.maxX, accuracy: 0.01, words)
            XCTAssertGreaterThanOrEqual(rect.minX - k.minX, 4, words)
            XCTAssertLessThanOrEqual(rect.minX - k.minX, 8, words)
            XCTAssertEqual(k.midY, rect.midY, accuracy: 0.01, words)
            XCTAssertGreaterThanOrEqual(k.height, 2.5 + 4, "\(words): covers the widest stroke with room")
            for s in NDStation.allCases { XCTAssertFalse(k.intersects(L.circle(s).insetBy(dx: -3, dy: -3)), "\(words) vs \(s)") }
            XCTAssertFalse(k.intersects(L.keptSafe), words)
            XCTAssertFalse(k.intersects(unconfirmedMask), words)
            for fixture in BrainBarDesignFixture.Name.allCases {
                let state = NDPipelineState.make(BrainBarDesignFixture.inputs(fixture))
                for s in NDStation.allCases {
                    for box in L.labelRects(s, state) { XCTAssertFalse(k.intersects(box), "\(fixture.rawValue): \(words) vs \(s)'s words") }
                }
            }
        }
    }

    /// Pixels: in the drawn chart, each caption's first capital (the "S" both start with) is centred on its line within
    /// 0.5 pt, its words are centred on the run within 1 pt, and the line itself shows just outside the knock-out but
    /// not between the words and the knock-out's edge. Dark and light.
    func testCaptionsAreDrawnOnTheirLines() throws {
        let state = NDPipelineState.make(BrainBarDesignFixture.inputs(.calm))
        for (suffix, appearance) in NDRenderHarness.appearances {
            let chart = NDPipelineChart(state: state, memories: "297,412", interactive: false)
                .background(ND.card).frame(width: L.size.width, height: L.size.height)
            let rep = try NDRenderHarness.host(chart, size: L.size, appearance: appearance)
            let card = try XCTUnwrap(rep.colorAt(x: 2 * Int(L.size.width) - 4, y: 2 * Int(L.size.height) - 4)?.usingColorSpace(.deviceRGB))
            /// How far a pixel is from the card colour (sum over RGB); ink is a clear difference.
            func weight(_ x: Int, _ y: Int) -> CGFloat {
                guard x >= 0, y >= 0, x < rep.pixelsWide, y < rep.pixelsHigh,
                      let c = rep.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { return 0 }
                return abs(c.redComponent - card.redComponent) + abs(c.greenComponent - card.greenComponent)
                    + abs(c.blueComponent - card.blueComponent)
            }
            func ink(_ x: Int, _ y: Int) -> Bool { weight(x, y) > 0.45 }
            let rects = Dictionary(uniqueKeysWithValues: L.laneLabelRects())
            for (words, pipe) in Self.captions {
                let run = try Self.run(pipe)
                let lineY = run.start.y, mid = try Self.expectedMidX(words, pipe)
                let w = L.textWidth(words, size: 11, weight: .medium)
                // Search where the words could be: around the line, across the whole run, away from its corners.
                let xs = Int(2 * (min(run.start.x, run.end.x) + 6))..<Int(2 * (max(run.start.x, run.end.x) - 6))
                let ys = Int(2 * (lineY - 8))..<Int(2 * (lineY + 8))
                let lineRows = Int(2 * (lineY - 1.5))...Int(2 * (lineY + 1.5))
                // Columns with ink outside the line's own rows: the words' horizontal extent.
                let cols = xs.filter { x in ys.contains { y in !lineRows.contains(y) && ink(x, y) } }
                let first = try XCTUnwrap(cols.first, "\(suffix) \(words): no words drawn"), last = try XCTUnwrap(cols.last)
                XCTAssertEqual(CGFloat(first + last + 1) / 4, mid, accuracy: 0.75, "\(suffix) \(words): words centred on the run")
                XCTAssertEqual(CGFloat(last - first + 1) / 2, w, accuracy: 3, "\(suffix) \(words): one run of words")
                // The "S": its first 5 pt of columns. It is symmetric top to bottom, so its ink-weighted centre is the
                // middle of its cap band.
                let sCols = first..<(first + 10)
                var mass: CGFloat = 0, moment: CGFloat = 0
                for y in ys {
                    for x in sCols { let m = weight(x, y); mass += m; moment += m * (CGFloat(y) + 0.5) }
                }
                let capCentre = moment / max(mass, 0.0001) / 2
                XCTAssertEqual(capCentre, lineY, accuracy: 0.5, "\(suffix) \(words): its capital's centre \(capCentre) on the line \(lineY)")
                // The line enters and leaves the knock-out: drawn 2 pt outside it, absent between the words and its edge.
                let rect = try XCTUnwrap(rects[words]), k = Self.knockout(rect)
                let row = Int(2 * lineY)
                XCTAssertTrue(ink(Int(2 * (k.minX - 2)), row) && ink(Int(2 * (k.maxX + 2)), row), "\(suffix) \(words): line outside")
                XCTAssertFalse(ink(Int(2 * (k.minX + 2)), row) || ink(Int(2 * (k.maxX - 2)), row), "\(suffix) \(words): no line inside")
            }
        }
    }

    /// `PIPELINE_LABELS_RENDER_DIR=<absolute dir>`: the Overview in the states the captions touch (calm, not confirmed,
    /// a fault) and the two job panels that open across the return line, dark and light, offscreen.
    func testRenderRequestedLineLabels() throws {
        guard let path = ProcessInfo.processInfo.environment["PIPELINE_LABELS_RENDER_DIR"] else { return }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        typealias F = BrainBarDesignFixture
        for (suffix, appearance) in NDRenderHarness.appearances {
            for fixture: F.Name in [.calm, .notConfirmed, .driveSignedOut] {
                let store = NDStore(fixture: F.inputs(fixture), page: .overview)
                _ = try NDRenderHarness.window(store, name: "overview-\(fixture.rawValue)-\(suffix)", appearance: appearance, in: directory)
            }
            for station: NDStation in [.jobs, .weekly, .agents] {
                let store = NDStore(fixture: F.inputs(.nightlyFailing), page: .overview)
                _ = try NDRenderHarness.hover(store, station: station, name: "hover-\(station.rawValue)-nightly-failing-\(suffix)",
                                              appearance: appearance, in: directory)
            }
        }
    }

    /// The not-confirmed caption is a line label too: the rail passes through the middle of its digits ("06:00", the
    /// height of capitals), its knock-out runs 6 pt past its words, and it ends at the chart's right edge. Dark and light.
    func testNotConfirmedCaptionIsDrawnOnTheRail() throws {
        let state = NDPipelineState.make(BrainBarDesignFixture.inputs(.notConfirmed))
        XCTAssertTrue(state.allUnconfirmed)
        let words = try XCTUnwrap(state.lastKnownAt.map(Say.valuesLastKnownAt))
        XCTAssertTrue(words.hasSuffix(":00"), words)
        let lineY = L.yRail
        for (suffix, appearance) in NDRenderHarness.appearances {
            let chart = NDPipelineChart(state: state, memories: "297,412", interactive: false)
                .background(ND.card).frame(width: L.size.width, height: L.size.height)
            let rep = try NDRenderHarness.host(chart, size: L.size, appearance: appearance)
            let card = try XCTUnwrap(rep.colorAt(x: 2 * Int(L.size.width) - 4, y: 2 * Int(L.size.height) - 4)?.usingColorSpace(.deviceRGB))
            func weight(_ x: Int, _ y: Int) -> CGFloat {
                guard x >= 0, y >= 0, x < rep.pixelsWide, y < rep.pixelsHigh,
                      let c = rep.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { return 0 }
                return abs(c.redComponent - card.redComponent) + abs(c.greenComponent - card.greenComponent)
                    + abs(c.blueComponent - card.blueComponent)
            }
            // The last glyph, a "0": the rightmost inked columns at the chart's right edge, 5 pt of them.
            let ys = Int(2 * (lineY - 8))..<Int(2 * (lineY + 8))
            let xs = (Int(2 * (L.size.width - 30))..<Int(2 * L.size.width)).filter { x in ys.contains { weight(x, $0) > 0.45 } }
            let last = try XCTUnwrap(xs.last, "\(suffix): no caption drawn")
            var mass: CGFloat = 0, moment: CGFloat = 0
            for y in ys { for x in (last - 9)...last { let m = weight(x, y); mass += m; moment += m * (CGFloat(y) + 0.5) } }
            let centre = moment / max(mass, 0.0001) / 2
            XCTAssertEqual(centre, lineY, accuracy: 0.5, "\(suffix): the digits' centre \(centre) on the rail \(lineY)")
            XCTAssertEqual(CGFloat(last + 1) / 2, L.size.width, accuracy: 1.5, "\(suffix): it ends at the chart's right edge")
        }
    }
}
