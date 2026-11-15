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
        // Review blocker 1: one unified 52 pt row, the pages control centred in the toolbar.
        XCTAssertEqual(controller.windowForTesting.toolbarStyle, .unified)
        XCTAssertEqual(controller.windowForTesting.toolbar?.centeredItemIdentifiers, [NDToolbar.pages])
        // 780 wide; AppKit adds up to 20 pt of unified-toolbar chrome to the 560 pt frame minimum (measured 580).
        XCTAssertEqual(controller.windowForTesting.minSize.width, NDGrid.windowMinimum.width)
        XCTAssertGreaterThanOrEqual(controller.windowForTesting.minSize.height, NDGrid.windowMinimum.height)
        XCTAssertLessThanOrEqual(controller.windowForTesting.minSize.height, NDGrid.windowMinimum.height + 20)
        XCTAssertFalse(controller.isShownForTesting, "a design switch never opens the window")

        d.set("classic", forKey: BrainBarDesign.key)
        controller.applyDesign()
        XCTAssertTrue(controller.windowForTesting.contentViewController === classic)
        XCTAssertEqual(controller.windowForTesting.minSize, BrainBarDashboardPanelController.minSize)
        XCTAssertNil(controller.windowForTesting.toolbar, "Classic's chrome is main's: no toolbar")
    }

    /// Review blocker 2: Settings is 520 pt wide and as tall as its three rows, fixed before it is shown.
    func testSettingsWindowIsSizedToItsRows() {
        _ = defaults()
        let settings = NDSettingsWindowController(store: NDStore(fixture: BrainBarDesignFixture.inputs(.calm)))
        let size = settings.window.contentRect(forFrameRect: settings.window.frame).size
        XCTAssertEqual(size.width, 520)
        XCTAssertGreaterThanOrEqual(size.height, 130)
        XCTAssertLessThanOrEqual(size.height, 200)
        XCTAssertFalse(settings.window.styleMask.contains(.resizable))
        XCTAssertEqual(settings.window.contentMaxSize, size)
        XCTAssertEqual(settings.window.identifier?.rawValue, "brainbar.nd.settings")
        XCTAssertEqual(settings.window.title, "BrainBar Settings")
        XCTAssertFalse(settings.window.isVisible)
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
            // Design pre-pass 5a: "none" rests on a fresh, known stores check.
            if row.visual == .emptyTrack, !(0...900).contains(c.verdictAwakeAgeS) || c.verdict == nil {
                out.append("\(row.name): none on a stale reading")
            }
        }
        // Design pre-pass 5b: a ring is explained by a dashed row, or says why in its help tag and VoiceOver.
        for area in m.areas where area.mark == .unknown && !area.rows.contains(where: { $0.visual == .unknown }) && area.unknownReason == nil {
            out.append("\(area.name): ring with no dashed row and no reason")
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

    /// Design pre-pass 7: beside "Everything else is fine." the fault drops its own reassurance; 22 words at most.
    func testOneUpkeepFaultKeepsToTheWordsBudget() {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.jobsFailedStreak))
        XCTAssertTrue(m.everythingElse)
        let shown = m.faults.flatMap(\.shownFacts).joined(separator: " ")
        XCTAssertFalse(shown.contains("still being saved"), shown)
        XCTAssertLessThanOrEqual(shown.split(separator: " ").count, 22, shown)
        XCTAssertTrue(m.faults.flatMap(\.facts).joined(separator: " ").contains("still being saved"), "VoiceOver and the report keep it")
    }

    func testStaleReadingNeverSaysNone() {
        let row = NDModel.make(BrainBarDesignFixture.inputs(.checking)).areas.first { $0.thing == .remembering }!.rows.first { $0.id == "stores" }!
        XCTAssertEqual(row.visual, .unknown)
        XCTAssertEqual(row.number.value, "—")
    }

    func testCheckedLineSaysWhenTheVerdictIsStale() {
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.calm)).checked, "Checked 1 min ago")
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.checking)).checked, "Last checked 18 min ago")
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

    /// N01: a real row with none of the lanes' results is damage, not an empty night.
    func testASchemaInvalidRowIsDamage() throws {
        let lanes = try XCTUnwrap(NDBackupHistoryReader.parseDatabase(log([
            row(5, id: "a"), "{\"attempted_at\": \"\(at(3))\", \"backup_log_provenance\": \"real\"}"]), now: now))
        XCTAssertNil(lanes.local)
        XCTAssertEqual(lanes.skipped, 1)
    }

    func testConversationPendingRowsDrawNoDot() throws {
        let lines = ["{\"attempted_at\": \"\(at(4))\", \"uploaded\": true, \"verified\": true, \"status\": \"verified\"}",
                     "{\"attempted_at\": \"\(at(2))\", \"uploaded\": false, \"verified\": false, \"status\": \"pending\"}"]
        let parsed = try XCTUnwrap(NDBackupHistoryReader.parseConversations(log(lines), now: now))
        XCTAssertEqual(parsed.attempts?.map(\.ok), [true])
    }
}

// MARK: - Code review round 1 (CODE-REVIEW.md F01–F07, N01–N05)

@MainActor final class BrainBarNewDesignReviewTests: XCTestCase {
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
    private func suite() -> UserDefaults {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let name = "BrainBarNewDesignReviewTests." + UUID().uuidString
        addTeardownBlock { UserDefaults(suiteName: name)?.removePersistentDomain(forName: name) }
        return UserDefaults(suiteName: name)!
    }

    /// F01: Settings reads the same accepted Drive row as every surface: a stale green check is not "Connected", a
    /// duplicated name is not read, and Reconnect appears only with the counted Drive fault.
    func testSettingsDriveComesFromTheAcceptedRow() {
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.calm)).drive, .init(state: "Connected", action: nil))
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.checking)).drive.state, "—", "18 minutes old: not current")
        let fault = NDModel.make(BrainBarDesignFixture.inputs(.oneFault)).drive
        XCTAssertEqual(fault.state, "Disconnected")
        XCTAssertEqual(fault.action?.kind, .reconnectDrive)
        var twice = BrainBarDesignFixture.inputs(.calm)
        let checks = twice.capture.verdict!.checks + [BrainBarDesignFixture.check("drive_auth", .red)]
        twice.capture.verdict = BrainBarHealthVerdict(dot: .red, checks: checks, notes: [], evaluatedAt: twice.capture.verdict!.evaluatedAt)
        XCTAssertEqual(NDModel.make(twice).drive.state, "—", "a duplicated check is never read for its fields")
    }

    /// F02: a last-run time alone is never a success dot.
    func testJobsDrawASuccessOnlyForARecordedSuccess() {
        let now = BrainBarDesignFixture.now
        func observation(exit: Int32?, pid: Int32? = nil, damaged: Bool? = nil, failures: Int? = nil) -> BrainLayerLaunchdJobObservation {
            var o = BrainLayerLaunchdJobObservation(loadState: .loaded, runs: 1, lastExitCode: exit, lastRunAt: now,
                                                     nextRunAt: nil, isContinuous: false)
            o.pid = pid; o.calendarHistoryDamaged = damaged; o.calendarConsecutiveFailures = failures
            return o
        }
        XCTAssertEqual(NDStore.outcome(observation(exit: 0)), .ok)
        XCTAssertEqual(NDStore.outcome(observation(exit: 1)), .failed)
        XCTAssertEqual(NDStore.outcome(observation(exit: nil)), .unknown)
        XCTAssertEqual(NDStore.outcome(observation(exit: BrainLayerMaintenanceExit.deferred)), .unknown)
        XCTAssertEqual(NDStore.outcome(observation(exit: 0, damaged: true)), .unknown)
        XCTAssertEqual(NDStore.outcome(observation(exit: 0, failures: 1)), .failed)
        XCTAssertEqual(NDStore.outcome(observation(exit: 0, pid: 42)), .running)

        var inputs = BrainBarDesignFixture.inputs(.calm)
        inputs.jobs = inputs.jobs!.map { var j = $0; if j.job == .decay { j.lastOutcome = .unknown }; if j.job == .drain { j.lastOutcome = .failed }; return j }
        let store = NDStore(fixture: inputs, page: .jobs)
        let lanes = NDJobsPage(store: store).lanes(store.model)
        XCTAssertEqual(lanes.first { $0.job == .decay }?.events.first?.kind, .unknown)
        XCTAssertEqual(lanes.first { $0.job == .drain }?.events.first?.kind, .fail)
        XCTAssertEqual(lanes.first { $0.job == .backupDaily }?.events.first?.kind, .ok)
    }

    /// F03: the copied report says what the surfaces show, runtime warnings included.
    func testReportCarriesTheRuntimeWarningsTheDropdownShows() {
        var i = BrainBarDesignFixture.inputs(.calm)
        i.extras = [NDExtra(id: "job.x", level: .fault, text: "A job alert text", thing: .safe, jobs: [.backupDaily])]
        var report = NDStore(fixture: i).report
        XCTAssertTrue(report.contains("1 thing needs you"), report)
        XCTAssertTrue(report.contains("A job alert text"), report)
        XCTAssertFalse(report.contains("All good"), report)
        i.extras = [NDExtra(id: "health.vector-coverage", level: .notice, text: "Vector check unverified", thing: .finding)]
        report = NDStore(fixture: i).report
        XCTAssertTrue(report.contains("Notice: Vector check unverified"), report)
        XCTAssertTrue(report.contains("Still checking search"), report)
        XCTAssertFalse(report.contains("All good"), report)
        XCTAssertFalse(NDStore(fixture: BrainBarDesignFixture.inputs(.calm)).report.contains("Warnings:"))
    }

    /// F04: no subscription or task runs unless a surface shows the new design; a history read in flight when the
    /// window closes is cancelled and publishes nothing.
    func testNewDesignWorkStopsOnSwitchAndClose() async throws {
        let runtime = BrainBarRuntime()
        let store = NDStore(runtime: runtime, workspace: BrainBarRecordingWorkspace())
        XCTAssertEqual(store.liveWorkCount, 0, "a new store does nothing until asked")
        store.setIconActive(true)
        XCTAssertGreaterThan(store.liveWorkCount, 0)
        store.setIconActive(false)
        XCTAssertEqual(store.liveWorkCount, 0, "back to Classic: every subscription and task gone")

        let dir = URL(fileURLWithPath: "/nonexistent-nd-test")
        store.setBackupPathsForTesting(.init(launchAgents: dir, databaseLog: dir, archiveLog: dir, maintenanceLog: dir,
                                             snapshotDirectory: dir, archiveDirectory: dir))
        final class Flag: @unchecked Sendable {
            private let lock = NSLock(); private var on = false
            func set() { lock.withLock { on = true } }
            var value: Bool { lock.withLock { on } }
        }
        let gate = DispatchSemaphore(value: 0), started = Flag()
        store.readHistory = { @Sendable known, _, _ in
            started.set(); gate.wait()
            return (.read(NDBackupLanes(local: [], drive: [], conversations: [], integrity: [])), known!)
        }
        store.setWindowActive(true)
        var didStart = false
        // The read starts off the main actor; yield until it has (up to 20 s, so a loaded CI runner cannot fail it early;
        // the wait ends as soon as it starts).
        for _ in 0..<400 where !didStart {
            try await Task.sleep(for: .milliseconds(50))
            didStart = started.value
        }
        XCTAssertTrue(didStart, "the read started")
        store.setWindowActive(false)
        XCTAssertEqual(store.liveWorkCount, 0, "closing the window cancels the read in flight")
        gate.signal()
        try await Task.sleep(for: .milliseconds(300))
        XCTAssertEqual(store.inputs.backupHistory, .unread, "a read finishing after close publishes nothing")
    }

    func testClassicControllersCreateNoNewDesignStore() {
        let d = suite()
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let panel = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        panel.designDefaults = d
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: panel, statusBar: Host(), menuDefaults: d)
        defer { controller.stop() }
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        XCTAssertNil(NDStore.existing(for: runtime), "Classic never builds the new design's store or its work")

        d.set("new", forKey: BrainBarDesign.key)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        let store = try? XCTUnwrap(NDStore.existing(for: runtime))
        XCTAssertEqual(store?.iconActive, true)
        d.set("classic", forKey: BrainBarDesign.key)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        panel.applyDesign()
        XCTAssertEqual(store?.iconActive, false)
        XCTAssertEqual(store?.windowActive, false)
        XCTAssertEqual(store?.liveWorkCount, 0)
    }

    /// F06: a capture keeps its window frame and Design choice in its own suite; the app's defaults are untouched.
    func testCaptureWritesOnlyItsOwnPreferences() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        setenv(BrainBarDesignFixture.environmentKey, "calm", 1)
        defer { unsetenv(BrainBarDesignFixture.environmentKey) }
        let key = BrainBarDashboardPanelController.frameDefaultsKey
        let before = UserDefaults.standard.object(forKey: key) as? String
        let bundleID = Bundle.main.bundleIdentifier ?? ""
        XCTAssertNotEqual(BrainBarDesignFixture.captureSuiteName, "com.brainlayer.brainbar")
        XCTAssertNotEqual(BrainBarDesignFixture.captureSuiteName, "com.brainlayer.BrainBar")
        XCTAssertNotEqual(BrainBarDesignFixture.captureSuiteName, bundleID)
        let (panel, status) = BrainBarDesignFixture.makeCaptureSurfaces(runtime: BrainBarRuntime(), activationPolicy: nil, statusBar: Host())
        defer { status.stop() }
        XCTAssertTrue(panel.designDefaults === BrainBarDesignFixture.capturePreferences)
        XCTAssertTrue(panel.newDesignStore.preferences === BrainBarDesignFixture.capturePreferences)
        XCTAssertTrue(panel.newDesignStore.isFixture)
        panel.placeWindowForTesting()
        panel.prepareForClose()
        XCTAssertNotNil(BrainBarDesignFixture.capturePreferences.string(forKey: key), "the capture's frame went to its own suite")
        XCTAssertEqual(UserDefaults.standard.object(forKey: key) as? String, before, "and nothing reached the app's defaults")
        XCTAssertNil(UserDefaults.standard.object(forKey: BrainBarDesign.key))
    }

    /// F07: the row is named for what it measures: the conversation reader's heartbeat.
    func testReaderRowIsNamedForTheHeartbeat() {
        let row = NDModel.make(BrainBarDesignFixture.inputs(.calm)).areas.first { $0.thing == .remembering }!.rows.first!
        XCTAssertEqual(row.name, "Read new conversations")
        XCTAssertTrue(row.accessibility.contains("last reported"), row.accessibility)
        XCTAssertTrue(NDModel.make(BrainBarDesignFixture.inputs(.calm)).areas.flatMap(\.rows).allSatisfy { $0.name != "New conversations" })
    }

    /// N04: a failed database check says what failed to VoiceOver.
    func testFailedDatabaseCheckIsSpokenInRegistryWords() {
        var i = BrainBarDesignFixture.inputs(.calm)
        i.capture.verdict = BrainBarHealthVerdict(dot: .red, checks: BrainBarDesignFixture.replacing(BrainBarDesignFixture.calmChecks(),
            BrainBarDesignFixture.check("integrity", .red)), notes: [], evaluatedAt: i.capture.verdict!.evaluatedAt)
        let row = NDModel.make(i).areas.first { $0.thing == .safe }!.rows.first { $0.id == "integrity" }!
        XCTAssertEqual(row.number.value, NDModel.Number.faultMark.value)
        XCTAssertTrue(row.accessibility.contains(BrainBarMenuPresenter.Say.s19.text), row.accessibility)
    }

    /// Review blocker 3: "Newest copy" is now − the lane's age (within an hour), in every fixture.
    func testCaptureClockIsTheWallClock() {
        let wall = Date(timeIntervalSinceReferenceDate: 784_000_123)
        let capture = BrainBarDesignFixture.makeNow(capturing: true, wall: wall)
        XCTAssertLessThan(wall.timeIntervalSince(capture), 60, "a capture's now is its launch minute, not a frozen 08:00")
        XCTAssertGreaterThanOrEqual(wall.timeIntervalSince(capture), 0)
        XCTAssertEqual(BrainBarDesignFixture.makeNow(capturing: false, wall: wall), BrainBarDesignFixture.fixedNow)
    }

    func testNewestCopyTimeMatchesTheLaneAge() throws {
        for name in BrainBarDesignFixture.Name.allCases {
            let i = BrainBarDesignFixture.inputs(name)
            let local = try XCTUnwrap(i.capture.verdict?.checks.first { $0.name == "backup_local" }?.ageS)
            let newest = try XCTUnwrap(i.newestLocalCopyAt)
            XCTAssertEqual(i.capture.now.timeIntervalSince(newest), local, accuracy: 3600, name.rawValue)
        }
    }

    /// Design pre-pass 6: on Backups the newest dot and the number come from the same record (within an hour).
    func testNewestRailDotMatchesTheNumber() {
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .backups)
            for lane in NDBackupsPage(store: store).lanes(store.model) where lane.number.unit == "h" {
                let newest = lane.events.filter { $0.kind != .next }.map(\.hours).max()
                guard let newest, let hours = Double(lane.number.value) else { continue }
                XCTAssertEqual(-newest, hours, accuracy: 1, "\(name.rawValue) \(lane.name)")
            }
        }
    }
}

/// F05: Classic against an oracle recorded from origin/main's own code (an export of `origin/main` 84b0e7a5 with a
/// recorder test, `Fixtures/classic-main-golden.json`), never against the code under test: the menu's titles, actions,
/// enabled states and keys, the icon's size, template flag and pixels, and the window's root type and minimum size.
@MainActor final class BrainBarClassicGoldenTests: XCTestCase {
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

    static func golden() throws -> [String: Any] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/classic-main-golden.json")
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    /// Pixels compared on the alpha and colour channels with a small tolerance, so a different macOS's antialiasing passes
    /// and a changed drawing does not.
    static func pixelsMatch(_ a: Data, _ b: Data) -> (Bool, String) {
        guard let ra = NSBitmapImageRep(data: a), let rb = NSBitmapImageRep(data: b) else { return (false, "undecodable") }
        guard ra.pixelsWide == rb.pixelsWide, ra.pixelsHigh == rb.pixelsHigh else {
            return (false, "size \(ra.pixelsWide)x\(ra.pixelsHigh) vs \(rb.pixelsWide)x\(rb.pixelsHigh)")
        }
        var total = 0, worst = 0
        for y in 0..<ra.pixelsHigh {
            for x in 0..<ra.pixelsWide {
                guard let ca = ra.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
                      let cb = rb.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else { continue }
                let d = [abs(ca.alphaComponent - cb.alphaComponent), abs(ca.redComponent * ca.alphaComponent - cb.redComponent * cb.alphaComponent),
                         abs(ca.greenComponent * ca.alphaComponent - cb.greenComponent * cb.alphaComponent),
                         abs(ca.blueComponent * ca.alphaComponent - cb.blueComponent * cb.alphaComponent)].max()!
                let v = Int(d * 255)
                total += v
                worst = max(worst, v)
            }
        }
        let mean = Double(total) / Double(ra.pixelsWide * ra.pixelsHigh)
        return (mean < 2 && worst < 96, "mean \(mean) worst \(worst)")
    }

    /// What an `ImageRenderer` raster depends on: Classic's chart icon (the old menu) is rendered by SwiftUI at
    /// `NSScreen.main`'s backing scale in the app's appearance, on this OS's renderer. Measured, not assumed.
    @MainActor static func rasterEnvironment() -> String {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let os = ProcessInfo.processInfo.operatingSystemVersion.majorVersion
        let scale = NSScreen.main?.backingScaleFactor ?? 0
        let appearance = NSApplication.shared.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua])?.rawValue ?? "unknown"
        return "macOS \(os), screen scale \(scale), \(appearance)"
    }

    func testPrintRasterEnvironment() throws {
        guard ProcessInfo.processInfo.environment["ND_PRINT_RASTER_ENV"] != nil else { throw XCTSkip("probe only") }
        print("ND_RASTER_ENV=\(Self.rasterEnvironment())")
    }

    func testClassicMatchesTheGoldenRecordedFromMain() throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        let golden = try Self.golden()
        XCTAssertEqual(golden["source"] as? String, "origin/main")
        XCTAssertNotNil(golden["rasterEnvironment"] as? String, "the raster icon's recording environment")
        XCTAssertEqual(golden["commit"] as? String, "84b0e7a58c034e46cb9080071557e0574db31f3c", "re-record when main's classic path moves")
        for expected in try XCTUnwrap(golden["cases"] as? [[String: Any]]) {
            let newMenu = try XCTUnwrap(expected["newMenu"] as? String)
            for design in [nil, "classic"] as [String?] {
                let name = "BrainBarClassicGoldenTests." + UUID().uuidString
                let d = try XCTUnwrap(UserDefaults(suiteName: name))
                defer { d.removePersistentDomain(forName: name) }
                if newMenu != "absent" { d.set(newMenu == "true", forKey: BrainBarNewMenuPreference.key) }
                if let design { d.set(design, forKey: BrainBarDesign.key) }
                let host = Host()
                let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
                let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
                    dashboardPanelController: nil, statusBar: host, menuDefaults: d)
                controller.setBadgeForTesting(.init(badgeOn: false, reason: "", activeCodes: []),
                    alertsURL: URL(fileURLWithPath: "/tmp/nd-absent-golden"))
                controller.menuNeedsUpdate(controller.contextMenuForTesting)
                let zero = Array(repeating: 0, count: 12)
                controller.renderStatusIconForTesting(stats: DashboardStats(chunkCount: 1, enrichedChunkCount: 1, pendingEnrichmentCount: 0,
                    enrichmentPercent: 50, enrichmentRatePerMinute: 1, databaseSizeBytes: 0, recentActivityBuckets: zero,
                    recentAgentWriteBuckets: zero, recentWatcherWriteBuckets: zero, recentEnrichmentBuckets: zero), state: .idle)
                let items = controller.contextMenuForTesting.items
                let label = "new menu \(newMenu), design \(design ?? "absent")"
                XCTAssertEqual(items.map(\.title), expected["titles"] as? [String], label)
                XCTAssertEqual(items.map { ($0.representedObject as? BrainBarMenuAction).map { "\($0)" } ?? "" }, expected["actions"] as? [String], label)
                XCTAssertEqual(items.map(\.isEnabled), expected["enabled"] as? [Bool], label)
                XCTAssertEqual(items.map(\.keyEquivalent), expected["keys"] as? [String], label)
                XCTAssertEqual(items.filter { $0.view != nil }.count, expected["hosted"] as? Int, label)
                let image = try XCTUnwrap(host.item.view.image)
                XCTAssertEqual(image.size.width, expected["iconWidth"] as? CGFloat, label)
                XCTAssertEqual(image.size.height, expected["iconHeight"] as? CGFloat, label)
                XCTAssertEqual(image.isTemplate, expected["iconTemplate"] as? Bool, label)
                let rep = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(image.size.width * 2),
                    pixelsHigh: Int(image.size.height * 2), bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
                rep.size = image.size
                NSGraphicsContext.saveGraphicsState()
                NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
                image.draw(in: NSRect(origin: .zero, size: image.size))
                NSGraphicsContext.restoreGraphicsState()
                let goldenPNG = try XCTUnwrap(Data(base64Encoded: try XCTUnwrap(expected["iconPNG"] as? String)))
                // The brain is a vector drawing at a fixed 2x: its pixels are compared on every machine. The chart icon is an
                // ImageRenderer raster whose pixels follow the OS, the screen's scale and the appearance (the macos-15
                // runner differed by a mean 3.7 of 255): compared only where it was recorded; its size and template
                // flag, and the whole menu, are compared everywhere.
                let raster = !image.isTemplate
                if !raster || Self.rasterEnvironment() == golden["rasterEnvironment"] as? String {
                    let (same, detail) = Self.pixelsMatch(try XCTUnwrap(rep.representation(using: .png, properties: [:])), goldenPNG)
                    XCTAssertTrue(same, "icon pixels differ from main (\(detail)), \(label)")
                } else {
                    print("BrainBarClassicGoldenTests: chart-icon pixels not compared here (\(Self.rasterEnvironment())); recorded on \(golden["rasterEnvironment"] as? String ?? "?")")
                }
                XCTAssertNil(NDStore.existing(for: runtime), "Classic built new-design work, \(label)")
                controller.stop()
            }
        }
        let panel = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
        // The root's type, structurally everywhere (main's hosting controller around main's root view); its exact spelling,
        // which follows the Swift toolchain's type printing, only where the golden was recorded.
        let rootType = String(describing: type(of: panel.contentViewControllerForTesting))
        let goldenRootType = try XCTUnwrap(golden["windowRootType"] as? String)
        XCTAssertTrue(rootType.hasPrefix("NSHostingController<") && rootType.contains("BrainBarWindowRootView"), rootType)
        XCTAssertEqual(rootType.components(separatedBy: "BrainBarWindowRootView").count,
                       goldenRootType.components(separatedBy: "BrainBarWindowRootView").count, "one classic root, as on main")
        if Self.rasterEnvironment() == golden["rasterEnvironment"] as? String { XCTAssertEqual(rootType, goldenRootType) }
        XCTAssertEqual(panel.windowForTesting.minSize.width, golden["windowMinWidth"] as? CGFloat)
        XCTAssertEqual(panel.windowForTesting.minSize.height, golden["windowMinHeight"] as? CGFloat)
    }
}

/// #1810's rule, applied to the new design: no sentence is written in NewDesign/. Its sentences live with the wording
/// registry (`BrainBarMenuPresenter.SurfaceSay`, the presenter's `Say`); what stays here are the copy deck's labels and
/// state words, check names and VoiceOver fragments. A literal is a sentence when it starts with a capital (or an
/// interpolation), has a space, and ends a sentence ("." or "…") or holds one (". " between words). Comments are skipped, as in `testThePresenterWritesNoWordsOutsideTheRegistry`.
final class BrainBarNewDesignWordsTests: XCTestCase {
    func testTheNewDesignWritesNoSentencesOutsideTheRegistry() throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBar/NewDesign")
        let files = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".swift") }
        XCTAssertGreaterThanOrEqual(files.count, 12)
        let literal = try NSRegularExpression(pattern: #""((?:[^"\\\n]|\\.)*)""#)
        var found: [String] = []
        for file in files {
            let code = try String(contentsOf: dir.appendingPathComponent(file), encoding: .utf8).components(separatedBy: "\n")
                .map { $0.components(separatedBy: "//")[0] }.joined(separator: "\n")
            for match in literal.matches(in: code, range: NSRange(code.startIndex..., in: code)) {
                let s = String(code[Range(match.range(at: 1), in: code)!])
                let words = s.range(of: #"^[A-Z\\(]"#, options: .regularExpression) != nil && s.contains(" ")
                let sentence = words && (s.range(of: #"[a-z)][.…]$"#, options: .regularExpression) != nil
                    || s.range(of: #"[a-z]\. [A-Z]"#, options: .regularExpression) != nil)
                if sentence { found.append("\(file): \(s)") }
            }
        }
        XCTAssertEqual(found, [])
    }
}
