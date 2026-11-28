import XCTest
import SwiftUI
@testable import BrainBar

/// w315: how the Activity chart turns counts into heights. Saves and reads by agents share the upper half's scale;
/// session lines keep the lower half's own; each half's full-height count is drawn beside it.
@MainActor final class BrainBarActivityScaleTests: XCTestCase {
    private let up = NDWaveform.halves(.hero).up
    private let down = NDWaveform.halves(.hero).down

    /// The owner's case: one save, one read by an agent and three session lines in one bucket.
    private func ownerWave(lines: [Int] = [3, 30], calls: [Int] = [1, 8], saves: [Int] = [1, 0]) -> NDWave {
        NDWave(saves: saves, reads: lines, agentReads: calls, readable: true)
    }

    func testOneSaveAndOneAgentReadDrawTheSameHeightWhateverEachSeriesPeaks() {
        let wave = ownerWave()
        let canvas = NDActivitySection.canvas
        let axis = NDWaveform.baseline(canvas, nowTick: true).midY
        let save = NDWaveform.savedHeight(wave, saves: 1, size: .hero)
        let read = axis - NDActivitySection.agentReadY(count: 1, peak: NDWaveform.upPeak(wave), canvas: canvas)
        XCTAssertEqual(save, read, accuracy: 0.001)
        XCTAssertEqual(save, up / 8, accuracy: 0.001, "one of eight, the upper half's shared peak")
        // Eight reads reach the full half; so would eight saves.
        XCTAssertEqual(axis - NDActivitySection.agentReadY(count: 8, peak: 8, canvas: canvas), up, accuracy: 0.001)
    }

    func testHeightsAreProportionalToCountsWithinEachHalfsScale() {
        let wave = ownerWave()
        XCTAssertEqual(NDWaveform.upPeak(wave), 8)
        XCTAssertEqual(NDWaveform.downPeak(wave), 30)
        XCTAssertEqual(NDWaveform.readHeight(wave, reads: 3, size: .hero), down * 3 / 30, accuracy: 0.001)
        XCTAssertEqual(NDWaveform.readHeight(wave, reads: 30, size: .hero), down, accuracy: 0.001)
        // Within a half, the ratio of heights is the ratio of counts.
        let sharedPeak = NDWave(saves: [2, 6], reads: [10, 20], agentReads: [4, 0], readable: true)
        XCTAssertEqual(NDWaveform.savedHeight(sharedPeak, saves: 6, size: .hero)
                       / NDWaveform.savedHeight(sharedPeak, saves: 2, size: .hero), 3, accuracy: 0.001)
        XCTAssertEqual(NDWaveform.readHeight(sharedPeak, reads: 20, size: .hero)
                       / NDWaveform.readHeight(sharedPeak, reads: 10, size: .hero), 2, accuracy: 0.001)
        // When the session-line peak is the owner's three, three lines fill their half.
        let quiet = ownerWave(lines: [3, 0], calls: [1, 0])
        XCTAssertEqual(NDWaveform.readHeight(quiet, reads: 3, size: .hero), down, accuracy: 0.001)
    }

    func testEachHalfCarriesItsScaleCueAtItsFullHeight() {
        let canvas = NDActivitySection.canvas
        let cy = NDWaveform.plot(canvas, nowTick: true).cy
        let ticks = NDActivitySection.scaleTicks(ownerWave())
        XCTAssertEqual(ticks, [.init(half: .up, count: 8, y: cy - 0.5 - up, setByAgentReads: true),
                               .init(half: .down, count: 30, y: cy + 0.5 + down)])
        // R1 (N1): the upper number is the reads' colour only when reads by agents set the shared peak.
        XCTAssertFalse(NDActivitySection.scaleTicks(ownerWave(calls: [1, 0], saves: [8, 0]))[0].setByAgentReads, "saves set it")
        XCTAssertFalse(NDActivitySection.scaleTicks(ownerWave(calls: [4, 0], saves: [4, 0]))[0].setByAgentReads, "a tie is the saves'")
        // The tallest bar in each half ends exactly at its tick.
        XCTAssertEqual(cy - 0.5 - NDWaveform.savedHeight(ownerWave(calls: [1, 0], saves: [8, 0]), saves: 8, size: .hero), ticks[0].y)
        XCTAssertEqual(cy + 0.5 + NDWaveform.readHeight(ownerWave(), reads: 30, size: .hero), ticks[1].y)
        XCTAssertEqual(NDActivitySection.scaleAccessibility(ownerWave(), known: true),
                       " Full height above the line is 8 saved or read by agents, below it 30 session lines.")
    }

    func testNoCueForAnEmptyHalfAnUnknownChartOrUnconfirmedCounts() {
        let canvas = NDActivitySection.canvas
        let cy = NDWaveform.plot(canvas, nowTick: true).cy
        XCTAssertEqual(NDActivitySection.scaleTicks(NDWave(saves: [0, 0], reads: [0, 0], agentReads: [0, 0])), [])
        XCTAssertEqual(NDActivitySection.scaleTicks(NDWave(saves: [0, 0], reads: [4, 0], agentReads: nil)),
                       [.init(half: .down, count: 4, y: cy + 0.5 + down)])
        XCTAssertEqual(NDActivitySection.scaleTicks(NDWave(saves: [0, 2], reads: [0, 0])),
                       [.init(half: .up, count: 2, y: cy - 0.5 - up)])
        XCTAssertEqual(NDActivitySection.scaleTicks(NDWave(saves: [3], reads: [3], readable: false)), [])
        XCTAssertEqual(NDActivitySection.scaleTicks(ownerWave(), known: false), [])
        XCTAssertEqual(NDActivitySection.scaleTicks(nil), [])
        XCTAssertEqual(NDActivitySection.scaleAccessibility(nil, known: true), "")
    }

    func testTinyCountsStayVisibleWithoutPassingForLargerOnes() {
        let wave = NDWave(saves: [1, 300], reads: [1, 3000], agentReads: [0, 0], readable: true)
        let min = NDWaveform.minVisible(.hero)
        XCTAssertEqual(NDWaveform.savedHeight(wave, saves: 1, size: .hero), min)
        XCTAssertEqual(NDWaveform.readHeight(wave, reads: 1, size: .hero), min)
        XCTAssertLessThanOrEqual(min / up, 0.05, "the floor is a sliver, not a bar")
        // The floor only lifts bars that would draw thinner; above it heights stay linear.
        XCTAssertEqual(NDWaveform.savedHeight(wave, saves: 150, size: .hero), up / 2, accuracy: 0.001)
        // Monotone: a larger count never draws shorter.
        var last: CGFloat = 0
        for count in 1...300 {
            let h = NDWaveform.savedHeight(wave, saves: count, size: .hero)
            XCTAssertGreaterThanOrEqual(h, last)
            XCTAssertLessThanOrEqual(h, up)
            last = h
        }
        // The agent-read line lifts a lone read off the axis by the same floor.
        let canvas = NDActivitySection.canvas
        let axis = NDWaveform.baseline(canvas, nowTick: true).midY
        XCTAssertEqual(axis - NDActivitySection.agentReadY(count: 1, peak: 300, canvas: canvas), min)
    }

    func testEmptyBucketsDrawNothing() {
        let wave = ownerWave()
        XCTAssertEqual(NDWaveform.savedHeight(wave, saves: 0, size: .hero), 0)
        XCTAssertEqual(NDWaveform.readHeight(wave, reads: 0, size: .hero), 0)
        XCTAssertEqual(NDWaveform.scaledHeight(0, peak: 0, extent: up, minVisible: 2), 0)
        XCTAssertEqual(NDWaveform.scaledHeight(5, peak: 0, extent: up, minVisible: 2), 0)
        let canvas = NDActivitySection.canvas
        XCTAssertEqual(NDActivitySection.agentReadY(count: 0, peak: 8, canvas: canvas), NDWaveform.baseline(canvas, nowTick: true).midY)
    }

    func testTheDropdownSparklineDrawsNoReadLineSoItsSavesKeepTheirOwnScale() {
        let wave = ownerWave()
        XCTAssertEqual(NDWaveform.upPeak(wave, size: .menu), 1)
        XCTAssertEqual(NDWaveform.savedHeight(wave, saves: 1, size: .menu), NDWaveform.halves(.menu).up)
    }
}
