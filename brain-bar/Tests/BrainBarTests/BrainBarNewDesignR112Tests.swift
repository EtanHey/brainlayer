import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Revision 1.12 (ruling L54, CHANGES-R1.12): the headline and the type scale, one wording per fault, four red marks for two
/// problems, the four contradictions and the two status lines. The panels' placements are held by the panel tests in
/// BrainBarNewDesignTests (inside the window, clear of every station, the Activity card and the footer, in every fixture).
@MainActor
final class BrainBarNewDesignR112Tests: XCTestCase {
    typealias Fixture = BrainBarDesignFixture
    typealias Say = BrainBarMenuPresenter.SurfaceSay

    func r1(_ name: Fixture.Name) -> NDR1 { NDR1.make(Fixture.inputs(name), NDModel.make(Fixture.inputs(name))) }

    // MARK: A. the headline and the scale

    /// The scale is 24 / 17 / 13 / 11, as tokens; the headline is 24 Semibold on a 30 pt line at −0.4; every text size the
    /// new design sets or measures is one of the four (symbol glyphs, the menu-bar icon and drawn marks aside).
    func testTheScaleIsTokensUsedEverywhere() throws {
        XCTAssertEqual(NDScale.all, [24, 17, 13, 11])
        XCTAssertEqual(NDScale.headline, 24)
        XCTAssertEqual(NDType.headlineLine, 30)
        XCTAssertEqual(NDType.headlineTracking, -0.4)
        XCTAssertTrue(NDScale.all.contains(NDType.panelBodySize), "the panels' text is on the scale")
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBar/NewDesign")
        let files = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.hasSuffix(".swift") && $0 != "NDIcon.swift" }
        let textSize = try NSRegularExpression(pattern: #"(?:text\(|textWidth\(|Text\(|systemFont\(ofSize:|Font\.system\(size:)[^\n]*?(?:size|ofSize): ([0-9.]+)"#)
        var off: [String] = []
        for file in files {
            let source = try String(contentsOf: root.appendingPathComponent(file), encoding: .utf8)
            for (number, line) in source.components(separatedBy: "\n").enumerated() {
                let code = line.components(separatedBy: "//")[0]
                guard !code.contains("Image(systemName"), !code.contains("symbol("), !code.contains("jobGlyph"), !code.contains("checkmark") else { continue }
                for match in textSize.matches(in: code, range: NSRange(code.startIndex..., in: code)) {
                    let value = String(code[Range(match.range(at: 1), in: code)!])
                    if let size = Double(value), !NDScale.all.contains(CGFloat(size)) { off.append("\(file):\(number + 1) size \(value)") }
                }
            }
        }
        XCTAssertEqual(off, [], "text off the 24/17/13/11 scale")
        // The window's headline and the two page titles use the headline token; the section labels the 13 Semibold `t2`.
        let window = try String(contentsOf: root.appendingPathComponent("NDWindow.swift"), encoding: .utf8)
        XCTAssertEqual(window.components(separatedBy: "NDType.windowHeadline").count - 1, 2, "the Overview headline and the page titles")
        XCTAssertTrue(window.contains("Text(title).font(NDType.section).foregroundStyle(ND.t2)"), "Activity and Pipeline")
        XCTAssertEqual(NDGrid.headlineToRows, 14)
        XCTAssertEqual(NDGrid.toActivity, 24)
        XCTAssertEqual(NDGrid.activityToPipeline, 32)  // L54 clearance, retained by L63.
    }

    // MARK: C. one wording per fault

    /// One sentence for a fault on the row, its page, the dropdown and its station's panel; the station shows its tail.
    func testOneWordingPerFaultOnEverySurface() throws {
        var weeklyFailing = Fixture.inputs(.calm)
        let index = try XCTUnwrap(weeklyFailing.jobs?.firstIndex { $0.job == .maintenanceWeekly })
        weeklyFailing.jobs?[index].lastOutcome = .failed
        weeklyFailing.jobs?[index].failureStreak = 2
        let cases: [(String, NDInputs, String, NDStation, String)] = [
            ("drive", Fixture.inputs(.driveSignedOut), "Google Drive is signed out", .drive, "signed out"),
            ("nightly", Fixture.inputs(.nightlyFailing), "Nightly check-up failed 3 nights", .jobs, "failed 3 nights"),
            ("weekly", weeklyFailing, "Weekly clean failed 2 weeks", .weekly, "failed 2 weeks"),
            ("saves", Fixture.inputs(.savesStuck), "Saves are stuck", .queue, "stuck 2\u{00A0}h"),
            ("agent", Fixture.inputs(.agentNotRead), "Cursor\u{2019}s sessions aren\u{2019}t being read", .read, "Cursor not read"),
        ]
        for (id, inputs, sentence, station, tail) in cases {
            let m = NDModel.make(inputs), r1 = NDR1.make(inputs, m)
            let item = try XCTUnwrap(r1.items.first { $0.id == id }, id)
            XCTAssertEqual(item.title, sentence, "\(id): the page headline")
            XCTAssertEqual(item.menuTitle ?? item.title, sentence, "\(id): the dropdown")
            XCTAssertEqual(r1.problemRows.first { $0.id == id }?.words, sentence, "\(id): the row")
            if let s = item.station { XCTAssertEqual(NDStationPopover.faultTitle(s, item, inputs), sentence, "\(id): the panel") }
            XCTAssertEqual(r1.pipeline.status[station]?.text, tail, "\(id): the station's tail")
        }
        // Search by meaning catching up (a notice) and the local copy running late (a notice, L53 C4).
        let meaning = r1(.meaningBehind)
        XCTAssertEqual(meaning.items.first { $0.id == "meaning" }?.title, "Search by meaning is catching up")
        XCTAssertEqual(meaning.pipeline.status[.search]?.text, "catching up")
        let late = r1(.backupLate)
        XCTAssertEqual(late.headline, "The local copy is running late")
        XCTAssertEqual(late.pipeline.status[.mac]?.text, Say.keptNewest(2, NDFormat.ageText(30 * 3600)))
        // The report action, one label on every surface.
        let nightly = r1(.nightlyFailing)
        XCTAssertEqual(nightly.items.first { $0.id == "nightly" }?.action?.label, "Copy a report for your agent")
    }

    // MARK: D. two problems, four red marks

    /// Two problems draw four red marks: each row's station symbol and each station's fault treatment; the headline none,
    /// the tab badges none (a neutral count).
    func testTwoProblemsDrawFourRedMarks() {
        let store = NDStore(fixture: Fixture.inputs(.twoFaults), page: .overview)
        let r1 = store.r1
        XCTAssertNil(r1.headlineSymbol, "the headline: none (L53)")
        XCTAssertEqual(r1.problemRows.count, 2)
        XCTAssertEqual(r1.problemRows.compactMap(\.station), [.drive, .jobs], "each row's mark is its station's symbol")
        let stationMarks = r1.pipeline.faults.count
        XCTAssertEqual(stationMarks, 2, "each station's one fault treatment (L29)")
        XCTAssertEqual(r1.problemRows.count + stationMarks, 4)
        XCTAssertEqual(r1.backupsBadge + r1.jobsBadge, 2, "the tabs still say where each problem lives")
        XCTAssertEqual(NDSegments.badgeFill, NDColor.badge, "a neutral pill, not red")
        XCTAssertNotEqual(NDSegments.badgeFill, NDColor.red)
        XCTAssertEqual(NDSegments.badgeText, NDColor.t1)
        // The row's glyph is its station's: a drawn moon for the nightly check-up, the cloud for Google Drive.
        XCTAssertEqual(NDPipelineLayout.symbols[.drive], "icloud")
    }

    // MARK: E. the four contradictions

    /// E1: the Backups page's check row is the local copy's check, not memory's, and its help tag says so.
    func testE1TheCheckRowIsTheLocalCopyCheck() {
        let store = NDStore(fixture: Fixture.inputs(.nightlyFailing), page: .backups)
        let lane = NDBackupsPage(store: store).lanes(store.model).first { $0.id == "integrity" }
        XCTAssertEqual(lane?.name, "Local copy check")
        XCTAssertTrue(lane?.help?.hasPrefix("Checks that last night\u{2019}s local copy opens and is complete.") == true)
        XCTAssertTrue(store.model.areas.flatMap(\.rows).contains { $0.name == "Local copy check" })
    }

    /// E2: not confirmed, the stations keep their last-known values as clock times (never ages), none later than the caption.
    func testE2LastKnownValuesWithTimes() throws {
        let i = Fixture.inputs(.notConfirmed)
        let p = NDPipelineState.make(i)
        XCTAssertTrue(p.allUnconfirmed)
        let evaluated = try XCTUnwrap(i.capture.verdict?.evaluatedAt)
        XCTAssertEqual(p.lastKnownAt, NDR1.hhmm(evaluated, i.capture))
        let mac = try XCTUnwrap(p.status[.mac]?.text)
        XCTAssertTrue(mac.hasPrefix("3 kept, newest "), mac)
        XCTAssertFalse(mac.hasSuffix(" ago"), "a time, not an age: \(mac)")
        XCTAssertTrue(p.status[.drive]?.text.hasPrefix("last upload ") == true, p.status[.drive]?.text ?? "nil")
        XCTAssertTrue(p.status[.jobs]?.text.hasPrefix("last run ") == true, p.status[.jobs]?.text ?? "nil")
        for (station, status) in p.status { XCTAssertFalse(status.text.hasSuffix(" ago"), "\(station): \(status.text)") }
    }

    /// E3: the late local copy is one notice with a second line; Drive's older upload is not a second problem.
    func testE3BackupLateHasItsSecondLineWithoutASecondProblem() {
        let r1 = r1(.backupLate)
        XCTAssertEqual(r1.headline, "The local copy is running late")
        XCTAssertTrue(r1.headlineDetail?.hasPrefix("The newest is 30\u{00A0}h old") == true, r1.headlineDetail ?? "nil")
        XCTAssertEqual(r1.items.count, 1, "one notice")
        XCTAssertNil(r1.pipeline.status[.drive]?.severity, "Drive's line stays t2")
        XCTAssertFalse(r1.pipeline.notices.contains(.drive) || r1.pipeline.faults.contains(.drive), "no ring on Drive")
        let i = Fixture.inputs(.backupLate)
        XCTAssertEqual(NDStationPopover.lines(.drive, i, r1.pipeline), ["Nothing new to upload: the local copy is late."])
        XCTAssertEqual(Say.localLateLine("30 h", "tonight at 03:15"), "The newest is 30 h old; the next one is due tonight at 03:15.")
        XCTAssertEqual(Say.localLateLine("30 h", nil), "The newest is 30 h old.", "the next run only when the schedule says")
    }

    /// E4: a missing agent's status reads red like every fault status; Read gets no ring and no badge.
    func testE4AMissingAgentsStatusIsRed() {
        let p = r1(.agentNotRead).pipeline
        XCTAssertEqual(p.status[.read]?.text, "Cursor not read")
        XCTAssertEqual(p.status[.read]?.severity, .red)
        XCTAssertFalse(p.faults.contains(.read), "no ring, no badge on Read (L41)")
        XCTAssertFalse(p.agentFaults.isEmpty, "the provider's icon keeps the fault")
    }

    // MARK: F. the two status lines

    func testSaveQueueAndSearchStatusLines() {
        let calm = r1(.calm).pipeline
        XCTAssertEqual(calm.status[.queue]?.text, "empty")
        XCTAssertEqual(calm.status[.search]?.text, "99% searchable")
        XCTAssertEqual(r1(.savesStuck).pipeline.status[.queue], .init(text: "stuck 2\u{00A0}h", severity: .red))
        XCTAssertEqual(r1(.searchFailing).pipeline.status[.search], .init(text: "failed 3 checks", severity: .red))
        var waiting = Fixture.inputs(.calm)
        waiting.savesWaiting = 4
        XCTAssertEqual(NDPipelineState.make(waiting).status[.queue]?.text, "4 waiting")
    }
}
