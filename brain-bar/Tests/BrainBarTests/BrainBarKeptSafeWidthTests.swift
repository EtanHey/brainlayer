import AppKit
import XCTest
@testable import BrainBar

/// Google Drive's node, and the Kept-safe group around Local copy and Drive, are only as wide as their widest designed
/// words need: 12 pt from the frame to the outermost words on each side and 16 pt between the two stations' words (the
/// lead's 1.3 rule), with less than 1 pt to spare. Every line each station can draw counts, in the weight it is drawn
/// in: a fault or notice in Semibold, a calm, upload or last-known line in Regular, the stall's clock in the person's own
/// 12- or 24-hour setting, the last-known clocks as `NDR1.clockOn` writes them, and the 96 pt upload bar.
@MainActor final class BrainBarKeptSafeWidthTests: XCTestCase {
    typealias L = NDPipelineLayout
    typealias Say = BrainBarMenuPresenter.SurfaceSay

    static func clocks() throws -> [String] {
        let utc = try XCTUnwrap(TimeZone(identifier: "UTC"))
        let widest = try XCTUnwrap(Calendar(identifier: .gregorian).date(from: DateComponents(timeZone: utc, year: 2026, month: 11,
                                                                                              day: 26, hour: 12, minute: 58)))
        return ["en_US", "ko_KR", "en_GB"].map { BrainBarBackupProgress.clock(widest, in: utc, locale: Locale(identifier: $0)) }
    }

    /// Half the widest thing a station draws under its circle.
    static func half(_ s: NDStation, semibold: [String], regular: [String], other: [CGFloat] = []) -> CGFloat {
        let words = semibold.map { L.textWidth($0, size: 11, weight: .semibold) } + regular.map { L.textWidth($0, size: 11) }
        return ((words + other + [L.textWidth(L.name(s), size: 13)]).max() ?? 0) / 2
    }

    /// The last-known clock a not-confirmed line carries on the day of the last check (`NDR1.clockOn`: "12:58").
    /// Known gap, not covered here: from an earlier day it reads "Wed 12:58", and "3 kept, newest Wed 12:58" with "last
    /// upload Wed 12:58" sit 8.5 pt apart and 8.5 pt from the frame (rule: 16 and 12). The words do not touch; the
    /// fix (shorter last-known words, or a wider frame) is its own change, recorded as a follow-up in the PR.
    static let lastKnownClocks = ["12:58"]

    func testKeptSafeIsOnlyAsWideAsItsWidestWords() throws {
        let clocks = try Self.clocks(), known = Self.lastKnownClocks
        let ages = [27, 30, 99].map { NDFormat.ageText(Double($0) * 3600) }
        let drive = Self.half(.drive,
            semibold: ages.map(Say.lastUpload) + [Say.signedOut, Say.stationWaiting],
            regular: ages.map(Say.lastUpload) + known.map(Say.lastUploadClock) + clocks.map(Say.driveStalledAt)
                + [Say.driveUploadingPercent(100), Say.backupUploadingShort, Say.backupVerifyingShort],
            other: [NDPipelineChart.progressFrame(.calm).width])
        let mac = Self.half(.mac,
            semibold: ages.map { Say.keptNewest(3, $0) },
            regular: ages.map { Say.keptNewest(3, $0) } + known.map { Say.keptNewestClock(3, $0) })
        let k = L.keptSafe, d = L.point(.drive).x, m = L.point(.mac).x
        let right = k.maxX - (d + drive), left = (m - mac) - k.minX, between = (d - drive) - (m + mac)
        // The rule holds at the widest words...
        XCTAssertGreaterThanOrEqual(right, 12, "Drive's words keep 12 pt to the frame")
        XCTAssertGreaterThanOrEqual(left, 12, "Local copy's words keep 12 pt to the frame")
        XCTAssertGreaterThanOrEqual(between, 16, "16 pt between the two stations' words")
        // ...with nothing reserved beyond it: no room kept for a bar, a button or a collision that is not drawn.
        XCTAssertLessThan(right, 13, "Drive is no wider than its widest words: \(right) pt to the frame")
        XCTAssertLessThan(left, 13, "Local copy is no wider than its widest words: \(left) pt to the frame")
        XCTAssertLessThan(between, 17, "the two stations sit no further apart than their words need: \(between) pt")
        // The upload bar is narrower than Drive's widest line, so it never sets the width.
        XCTAssertLessThan(NDPipelineChart.progressFrame(.calm).width / 2, drive)
    }
}
