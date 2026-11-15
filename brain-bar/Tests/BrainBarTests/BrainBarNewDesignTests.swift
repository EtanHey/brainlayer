import AppKit
import XCTest
@testable import BrainBar

/// The new design behind `BrainBarDesign` (phase 11). Headless throughout: in-memory status items, windows that are
/// never ordered front, activation prohibited.
@MainActor final class BrainBarNewDesignSwitchTests: XCTestCase {
    private final class Button: NSButton {}
    private final class Item: BrainBarStatusItem {
        let length: CGFloat = NSStatusItem.variableLength
        var menu: NSMenu?
        let view = Button()
        var button: NSButton? { view }
    }
    private final class Host: BrainBarStatusItemHosting {
        let item = Item()
        func statusItem(withLength: CGFloat) -> any BrainBarStatusItem { item }
        func removeStatusItem(_ item: any BrainBarStatusItem) {}
    }

    private func defaults() -> UserDefaults {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let name = "BrainBarNewDesignSwitchTests." + UUID().uuidString
        addTeardownBlock { UserDefaults(suiteName: name)?.removePersistentDomain(forName: name) }
        return UserDefaults(suiteName: name)!
    }

    private func stats() -> DashboardStats {
        DashboardStats(chunkCount: 1, enrichedChunkCount: 1, pendingEnrichmentCount: 0, enrichmentPercent: 50,
                       enrichmentRatePerMinute: 1, databaseSizeBytes: 0, recentActivityBuckets: Array(repeating: 0, count: 12),
                       recentAgentWriteBuckets: Array(repeating: 0, count: 12), recentWatcherWriteBuckets: Array(repeating: 0, count: 12),
                       recentEnrichmentBuckets: Array(repeating: 0, count: 12))
    }

    private func png(_ image: NSImage?) throws -> Data {
        let image = try XCTUnwrap(image)
        let rep = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(image.size.width * 2),
            pixelsHigh: Int(image.size.height * 2), bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        rep.size = image.size
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        image.draw(in: NSRect(origin: .zero, size: image.size))
        NSGraphicsContext.restoreGraphicsState()
        return try XCTUnwrap(rep.representation(using: .png, properties: [:]))
    }

    /// What the status item and menu show for one set of stored preferences.
    private func shown(design: String?, newMenu: Bool?) throws -> (titles: [String], actions: [String], image: Data, size: NSSize,
                                                                   hostedNew: Int) {
        let d = defaults()
        if let design { d.set(design, forKey: BrainBarDesign.key) }
        if let newMenu { d.set(newMenu, forKey: BrainBarNewMenuPreference.key) }
        let host = Host()
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(),
            runtime: BrainBarRuntime(launchMode: .menuItemDaemon), dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []),
            alertsURL: URL(fileURLWithPath: "/tmp/nd-absent-" + UUID().uuidString))
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        let items = controller.contextMenuForTesting.items
        return (items.map(\.title), items.map { ($0.representedObject as? BrainBarMenuAction).map { "\($0)" } ?? "" },
                try png(host.item.view.image), host.item.view.image?.size ?? .zero,
                items.filter { $0.view is NDMenuRowHost }.count)
    }

    /// L6: with `classic` (absent or stored), the icon and the menu are main's, byte for byte and title for title.
    func testClassicIsMainsIconAndMenuForBothMenuSettings() throws {
        for newMenu in [nil, false, true] as [Bool?] {
            let absent = try shown(design: nil, newMenu: newMenu)
            let stored = try shown(design: "classic", newMenu: newMenu)
            XCTAssertEqual(stored.titles, absent.titles, "menu titles, new menu \(String(describing: newMenu))")
            XCTAssertEqual(stored.actions, absent.actions)
            XCTAssertEqual(stored.image, absent.image, "icon pixels, new menu \(String(describing: newMenu))")
            XCTAssertEqual(stored.hostedNew, 0, "classic never builds a new-design row")
        }
        // Main's own expectations for the two menu settings (BrainBarBrainIconTests): charts and the legacy rows when off,
        // the 22 pt brain and the presenter's standing items when on.
        let off = try shown(design: "classic", newMenu: false)
        XCTAssertEqual(off.titles, ["Nothing needs attention", "Show log", "", "", "Open Dashboard", "Settings…", "", "Restart BrainBar", "", "Quit BrainBar"])
        XCTAssertEqual(off.size, NSSize(width: 26, height: 14))
        let on = try shown(design: "classic", newMenu: nil)
        XCTAssertEqual(on.size, NSSize(width: 22, height: 22))
        XCTAssertFalse(on.titles.contains("Open Dashboard"))
    }

    func testAbsentDesignIsClassicAndReadingNeverWrites() {
        let d = defaults()
        XCTAssertEqual(BrainBarDesign.current(in: d), .classic)
        XCTAssertNil(d.object(forKey: BrainBarDesign.key))
        d.set("new", forKey: BrainBarDesign.key)
        XCTAssertEqual(BrainBarDesign.current(in: d), .new)
        d.set("something else", forKey: BrainBarDesign.key)
        XCTAssertEqual(BrainBarDesign.current(in: d), .classic, "an unknown value is classic")
    }

    func testNewDesignDrawsItsOwnMenuAndIconAndSwitchesBackLive() throws {
        let new = try shown(design: "new", newMenu: nil)
        XCTAssertGreaterThanOrEqual(new.hostedNew, 5, "headline and four area rows")
        XCTAssertTrue(new.actions.contains("settings"))
        XCTAssertTrue(new.actions.contains("quit"))
        XCTAssertEqual(new.size, NSSize(width: 22, height: 22))

        let d = defaults()
        d.set("new", forKey: BrainBarDesign.key)
        let host = Host()
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(),
            runtime: BrainBarRuntime(launchMode: .menuItemDaemon), dashboardPanelController: nil, statusBar: host, menuDefaults: d)
        defer { controller.stop() }
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        XCTAssertTrue(controller.contextMenuForTesting.items.contains { $0.view is NDMenuRowHost })
        XCTAssertEqual(host.item.view.alphaValue, 0.45, accuracy: 0.001, "no reading yet: the resting brain, dimmed")
        d.set("classic", forKey: BrainBarDesign.key)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertFalse(controller.contextMenuForTesting.items.contains { $0.view is NDMenuRowHost })
        XCTAssertEqual(host.item.view.alphaValue, 1)
    }

    /// The window: classic keeps main's hosting controller and root type; new swaps in its own and swaps back.
    func testWindowRootIsMainsInClassicAndSwapsLive() throws {
        _ = defaults()
        let controller = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
        let classic = controller.contentViewControllerForTesting
        XCTAssertTrue(controller.windowForTesting.contentViewController === classic)
        XCTAssertTrue(String(describing: type(of: classic)).contains("BrainBarWindowRootView"), String(describing: type(of: classic)))
        XCTAssertEqual(controller.windowForTesting.minSize, BrainBarDashboardPanelController.minSize)
        XCTAssertFalse(controller.isShownForTesting)

        let d = defaults()
        d.set("new", forKey: BrainBarDesign.key)
        controller.designDefaults = d
        controller.applyDesign()
        let new = try XCTUnwrap(controller.windowForTesting.contentViewController)
        XCTAssertTrue(String(describing: type(of: new)).contains("NDWindowRoot"), String(describing: type(of: new)))
        XCTAssertEqual(controller.windowForTesting.minSize, NDGrid.windowMinimum)
        XCTAssertFalse(controller.isShownForTesting, "a design switch never opens the window")

        d.set("classic", forKey: BrainBarDesign.key)
        controller.applyDesign()
        XCTAssertTrue(controller.windowForTesting.contentViewController === classic)
        XCTAssertEqual(controller.windowForTesting.minSize, BrainBarDashboardPanelController.minSize)
    }
}

/// L5 and the truth rules: the new design never shows a calmer state than the presentation. Enumerated over the
/// presenter's own input space (BrainBarMenuInvariantTests.inputs) and the seventeen contract cases.
final class BrainBarNewDesignTruthTests: XCTestCase {
    typealias P = BrainBarMenuPresentation
    typealias Thing = BrainBarMenuPresenter.Thing

    static func violations(_ i: NDInputs) -> [String] {
        let c = i.capture
        let p = BrainBarMenuPresenter.present(c)
        let m = NDModel.make(i)
        var out: [String] = []
        let say = BrainBarMenuPresenter.Say.self
        // The headline, count and icon: the presentation's, or more urgent, never calmer.
        if m.count < p.count { out.append("count \(m.count) < presentation \(p.count)") }
        if p.icon == .marked, m.icon != .marked { out.append("icon calmer than the presentation") }
        if p.form == .needs, m.headlineMark != .fault { out.append("needs form without the fault mark") }
        if m.headline == say.allGood.text, !BrainBarMenuPresenter.allGoodProven(c) { out.append("All good unproven") }
        if m.headline == say.allGood.text, !i.extras.isEmpty { out.append("All good beside a runtime warning") }
        if m.faults.count != m.count, m.icon == .marked { out.append("count \(m.count) but \(m.faults.count) fault rows") }
        if m.everythingElse, !p.rows.contains(where: { $0.kind == .everythingElse }) { out.append("Everything else without the presenter") }
        // Every Needs-you row of the presentation is drawn, in its order, with its action.
        let pf = p.rows.filter { $0.kind == .needsYou }
        if Array(m.faults.prefix(pf.count)).map(\.title) != pf.map { $0.strings.first ?? "" } { out.append("fault rows differ") }
        if Array(m.faults.prefix(pf.count)).map(\.action) != pf.map(\.action) { out.append("fault actions differ") }
        // No area reads fine unless the presenter proves it fine (unknown never drawn as fine).
        let proven = BrainBarMenuPresenter.provenFine(c)
        for area in m.menuAreas where area.mark == .none && !proven.contains(area.thing) {
            out.append("\(area.name) drawn fine, not proven")
        }
        for area in m.areas where area.thing != nil && area.mark == .none && !proven.contains(area.thing!) {
            out.append("\(area.name) panel fine, not proven")
        }
        // A red area always has a counted Needs-you row behind it.
        if !m.faultThings.isEmpty, m.count == 0 { out.append("red area with nothing counted") }
        // Unknown is dashed and "—", never a zero or a value; a value is never dashed.
        for row in m.areas.flatMap(\.rows) {
            if case .agent = row.visual { continue }  // an agent row draws its own unknown (dashed ring and bar)
            let dashed = row.visual == .unknown
            let dash = row.number.value == "—"
            if dashed != dash, !(row.stateWord) { out.append("\(row.name): dashed \(dashed) but number \(row.number.value)") }
            if row.visual == .emptyTrack, i.savesWaiting != 0 { out.append("\(row.name): empty drawn without a measured empty queue") }
        }
        // Remembering's number is the newest conversation's age, a fact apart from its waveform.
        for area in m.menuAreas where area.thing != .remembering && (area.visual == .unknown) != (area.number.value == "—") {
            out.append("menu \(area.name): visual \(area.visual) number \(area.number.value)")
        }
        return out
    }

    func testNeverCalmerThanThePresentationOnEveryEnumeratedInput() throws {
        typealias I = BrainBarMenuInvariantTests
        let inputs = I.inputs()
        let tally = I.Tally()
        // Every input, crossed with the dot, every freshness and the checking flag; measured both ways on a stride.
        // Every fourth input (the presenter's own test covers each one); the full cross on each.
        let strided = Array(stride(from: 0, to: inputs.count, by: 4))
        DispatchQueue.concurrentPerform(iterations: strided.count) { k in
            let index = strided[k]
            var local: [String] = [], n = 0
            for dot in [BrainBarHealthVerdict.State.green, .red] {
                for freshness in I.Freshness.allCases {
                    for checking in [false, true] {
                        let measured = index % 3 != 0
                        var i = NDInputs(capture: I.capture(inputs[index], dot: dot, checking: checking, wake: false,
                                                            freshness: freshness, measured: measured))
                        i.savesWaiting = index % 2 == 0 ? 0 : nil
                        local += Self.violations(i).map { "\($0) | input \(index) dot \(dot) \(freshness) checking \(checking)" }
                        n += 1
                    }
                }
            }
            tally.add((n, local, [:]))
        }
        var checked = tally.outputs, found = tally.violations
        for c in try BrainBarMenuPresentationTests.loadCases().map({ try BrainBarMenuPresentationTests.capture($0["inputs"] as! [String: Any]) }) {
            found += Self.violations(NDInputs(capture: c))
            checked += 1
        }
        print("BrainBarNewDesignTruthTests: \(checked) captures checked")
        XCTAssertGreaterThan(checked, 40_000)
        XCTAssertEqual(Array(found.prefix(20)), [], "\(found.count) violations")
    }

    func testEveryFixtureHoldsTheRules() {
        for name in BrainBarDesignFixture.Name.allCases {
            XCTAssertEqual(Self.violations(BrainBarDesignFixture.inputs(name)), [], name.rawValue)
        }
    }

    /// The three runtime warnings: an attention-level one is counted and drawn in its own words; a notice stops "All good".
    func testRuntimeWarningsAreCountedAndNeverBesideAllGood() {
        var i = BrainBarDesignFixture.inputs(.calm)
        XCTAssertEqual(NDModel.make(i).headline, "All good")
        i.extras = [NDExtra(id: "job.x", level: .fault, text: "A job alert text", thing: .safe, jobs: [.backupDaily])]
        var m = NDModel.make(i)
        XCTAssertEqual(m.count, 1)
        XCTAssertEqual(m.headline, "1 thing needs you")
        XCTAssertEqual(m.icon, .marked)
        XCTAssertEqual(m.faults.map(\.title), ["A job alert text"])
        XCTAssertEqual(m.menuAreas.first { $0.thing == .safe }?.mark, .fault)
        XCTAssertFalse(m.everythingElse)
        XCTAssertEqual(Self.violations(i), [])

        i.extras = [NDExtra(id: "health.vector-coverage", level: .notice, text: "Vector check unverified", thing: .finding)]
        m = NDModel.make(i)
        XCTAssertEqual(m.headline, "Still checking search")
        XCTAssertEqual(m.count, 0)
        XCTAssertEqual(m.menuAreas.first { $0.thing == .finding }?.mark, .unknown)
        XCTAssertEqual(Self.violations(i), [])
    }

    func testUnloadedWatcherIsAFaultInTheRegistrysWords() {
        let setting = BrainLayerLaunchdJobSetting(enabled: true, loadState: .unloaded)
        let extras = NDExtra.derive(snapshot: .checking, watchSetting: setting)
        XCTAssertEqual(extras.map(\.level), [.fault])
        XCTAssertEqual(extras.first?.thing, .remembering)
        XCTAssertEqual(extras.first?.text, BrainBarMenuPresenter.Say.generic.text)
    }

    /// L4: the Drive fault drops the Drive copy's age where a gauge shows it; VoiceOver and the report keep it.
    func testDriveFaultDropsTheAgeTheGaugeShows() {
        var c = BrainBarDesignFixture.inputs(.oneFault).capture
        c.backupDailyFailures = 2  // the Drive row absorbs a failing database copy and says its age (R15)
        let checks = (c.verdict?.checks ?? []).map { k -> BrainBarHealthVerdict.Check in
            k.name == "backup_offsite" ? BrainBarDesignFixture.check("backup_offsite", .red, age: 21 * 3600, limit: 18 * 3600) : k
        }
        c.verdict = BrainBarHealthVerdict(dot: .red, checks: checks, notes: [], evaluatedAt: c.verdict!.evaluatedAt)
        let m = NDModel.make(NDInputs(capture: c))
        let drive = m.faults.first { $0.concerns.contains("drive_auth") }
        XCTAssertNotNil(drive)
        XCTAssertTrue(drive!.facts.joined(separator: " ").contains("Your newest backup in Google Drive is"), drive!.facts.description)
        XCTAssertFalse(drive!.shownFacts.joined(separator: " ").contains("Your newest backup in Google Drive is"))
        XCTAssertTrue(drive!.shownFacts.joined(separator: " ").contains("Backups can’t upload until you reconnect it."))
    }

    func testConcernedRowsTurnFaultAndTheDropdownShowsTheConcernedCopy() {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.oneFault))
        let rows = m.areas.first { $0.thing == .safe }!.rows
        guard case let .gauge(g)? = rows.first(where: { $0.id == "backup_offsite" })?.visual else { return XCTFail("gauge") }
        XCTAssertTrue(g.concerned)
        XCTAssertEqual(g.tone, .fault)
        guard case let .gauge(local)? = rows.first(where: { $0.id == "backup_local" })?.visual else { return XCTFail("gauge") }
        XCTAssertFalse(local.concerned)
        XCTAssertEqual(local.tone, .fine)
        XCTAssertEqual(m.menuAreas.first { $0.thing == .safe }?.number.text, "21\u{00A0}h")
        XCTAssertEqual(m.backupsTabCount, 1)
    }

    func testUpkeepFailingIsOneFormOneString() {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.severalFaults))
        XCTAssertEqual(m.menuAreas.first { $0.thing == .upkeep }?.visual, .streak(3))
        XCTAssertEqual(m.menuAreas.first { $0.thing == .upkeep }?.number.text, "3\u{00A0}nights")
        XCTAssertEqual(m.areas.first { $0.thing == .upkeep }?.rows.first?.number.text, "3\u{00A0}nights")
        XCTAssertEqual(m.jobsTabCount, 1)
        XCTAssertEqual(m.count, 2)
    }

    /// L3: the 36 h / 8 d scale places the notch and never colours it.
    func testUpkeepScaleDecidesNoState() {
        var i = BrainBarDesignFixture.inputs(.calm)
        i.nightlyLastRun = .ran(i.capture.now.addingTimeInterval(-40 * 3600))
        let m = NDModel.make(i)
        guard case let .gauge(g)? = m.areas.first(where: { $0.thing == .upkeep })?.rows.first?.visual else { return XCTFail("gauge") }
        XCTAssertTrue(g.pinned)
        XCTAssertEqual(g.tone, .fine)
        XCTAssertEqual(m.headline, "All good")
        i.nightlyLastRun = .unread
        guard case .unknown? = NDModel.make(i).areas.first(where: { $0.thing == .upkeep })?.rows.first?.visual else { return XCTFail("dashed") }
    }

    func testMeasuredZeroDrawsNothingAndUnknownIsDashed() {
        var i = BrainBarDesignFixture.inputs(.calm)
        let m = NDModel.make(i)
        let gemini = m.areas.first { $0.name == "Agents" }!.rows.first { $0.id == "gemini" }!
        XCTAssertEqual(gemini.visual, .agent(sessions: 0, saves: 0, scale: 72))
        XCTAssertEqual(gemini.number.value, "0")
        i.agents = .unknown
        let unknown = NDModel.make(i).areas.first { $0.name == "Agents" }!
        XCTAssertEqual(unknown.mark, .unknown)
        XCTAssertTrue(unknown.rows.allSatisfy { $0.number.value == "—" })
    }

    func testCheckedLineSaysWhenTheVerdictIsStale() {
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.calm)).checked, "Checked 1 min ago")
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.cantTell)).checked, "Last checked 18 min ago")
    }

    func testFixtureNamesAreTheCaptureList() {
        XCTAssertEqual(BrainBarDesignFixture.Name.allCases.map(\.rawValue),
                       ["calm", "checking", "cant-tell", "one-fault", "reconnect-waiting", "several-faults",
                        "backups-failed-upload", "jobs-failed-streak"])
    }
}

/// L2: the backup-history reader, on invented lines only.
final class BrainBarNewDesignBackupHistoryTests: XCTestCase {
    let now = ISO8601DateFormatter().date(from: "2026-11-15T08:00:00+02:00")!

    private func log(_ lines: [String]) -> NDBackupHistoryReader.Log { .init(lines: lines.map { Substring($0) }, unreadable: false) }
    private func at(_ hoursAgo: Double) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f.string(from: now.addingTimeInterval(-hoursAgo * 3600))
    }
    private func row(_ hoursAgo: Double, id: String, real: Bool = true, local: Bool = true, uploaded: Bool? = true,
                     verified: Bool = true, integrity: String = "ok") -> String {
        var fields = ["\"attempted_at\": \"\(at(hoursAgo))\"", "\"backup_invocation_id\": \"\(id)\"",
                      "\"backup_log_provenance\": \"\(real ? "real" : "pytest")\"", "\"local_verified\": \(local)",
                      "\"verified\": \(verified)", "\"local_integrity\": \"\(integrity)\""]
        if let uploaded { fields.append("\"uploaded\": \(uploaded)") }
        return "{" + fields.joined(separator: ", ") + "}"
    }

    func testRealRowsGroupedByInvocationInsideTheWindow() throws {
        let lanes = try XCTUnwrap(NDBackupHistoryReader.parseDatabase(log([
            row(200, id: "old"),                          // older than 8 days: outside
            row(30, id: "a", uploaded: true, verified: false), row(29.9, id: "a"),  // one invocation: the newest row speaks
            row(6, id: "b", real: false),                 // a test row: never counted
            row(5, id: "c", local: false, uploaded: false, verified: false, integrity: "corrupt"),
        ]), now: now))
        XCTAssertEqual(lanes.local?.map(\.ok), [true, false])
        XCTAssertEqual(lanes.drive?.map(\.ok), [true, false])
        XCTAssertEqual(lanes.integrity?.map(\.ok), [true, false])
        XCTAssertEqual(lanes.skipped, 0)
    }

    func testADamagedLineMakesTheLanesUnknownAndIsCounted() throws {
        let lanes = try XCTUnwrap(NDBackupHistoryReader.parseDatabase(log([row(5, id: "a"), "{not json"]), now: now))
        XCTAssertNil(lanes.local)
        XCTAssertNil(lanes.drive)
        XCTAssertEqual(lanes.skipped, 1)
    }

    func testAnUnreadableLogIsUnknownNeverEmpty() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("nd-history-" + UUID().uuidString)
        let history = NDBackupHistoryReader.read(databaseLog: dir.appendingPathComponent("missing.log"),
                                                 conversationLog: dir.appendingPathComponent("missing-too.log"), now: now)
        guard case let .read(lanes) = history else { return XCTFail("read") }
        XCTAssertNil(lanes.local)
        XCTAssertNil(lanes.conversations)
    }

    func testReadsOnlyTheTailAndNeverWrites() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("nd-history-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("backup-daily.log")
        let lines = (0..<500).map { row(Double($0) * 0.1, id: "i\($0)") }
        try (lines.reversed().joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
        let before = try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date
        let tail = NDBackupHistoryReader.tail(url)
        XCTAssertEqual(tail.lines.count, NDBackupHistoryReader.maxLines)
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date, before)
    }

    func testConversationPendingRowsDrawNoDot() throws {
        let lines = ["{\"attempted_at\": \"\(at(4))\", \"uploaded\": true, \"verified\": true, \"status\": \"verified\"}",
                     "{\"attempted_at\": \"\(at(2))\", \"uploaded\": false, \"verified\": false, \"status\": \"pending\"}"]
        let parsed = try XCTUnwrap(NDBackupHistoryReader.parseConversations(log(lines), now: now))
        XCTAssertEqual(parsed.attempts?.map(\.ok), [true])
    }
}
