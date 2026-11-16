import AppKit
import XCTest
import SwiftUI
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
        XCTAssertEqual(host.item.view.alphaValue, 1, "L23: no reading yet is drawn into the image (45 % brain and a \"?\" badge)")
        d.set("classic", forKey: BrainBarDesign.key)
        controller.menuNeedsUpdate(controller.contextMenuForTesting)
        controller.renderStatusIconForTesting(stats: stats(), state: .idle)
        XCTAssertFalse(controller.contextMenuForTesting.items.contains { $0.view is NDMenuRowHost })
        XCTAssertEqual(host.item.view.alphaValue, 1)
    }

    private func segmentedControl(in view: NSView) -> NSSegmentedControl? {
        if let control = view as? NSSegmentedControl { return control }
        for sub in view.subviews { if let found = segmentedControl(in: sub) { return found } }
        return nil
    }

    /// D1 (real screen, round 1): a click on a tab changes the page, the selected segment and the window's height together.
    /// It drives the action the toolbar's segmented control sends (selection, then its target/action), three times, and
    /// after each checks the stored page, the page the window root drew, the control's selected segment and the height.
    func testATabClickChangesPageSegmentAndHeightTogether() throws {
        let d = defaults()
        d.set("new", forKey: BrainBarDesign.key)
        let controller = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: BrainBarRuntime())
        controller.designDefaults = d
        controller.applyDesign()
        let window = controller.windowForTesting
        let toolbar = try XCTUnwrap(window.toolbar)
        let delegate = try XCTUnwrap(toolbar.delegate)
        let item = try XCTUnwrap(delegate.toolbar?(toolbar, itemForItemIdentifier: NDToolbar.pages, willBeInsertedIntoToolbar: true))
        let host = try XCTUnwrap(item.view)
        host.frame = NSRect(origin: .zero, size: host.fittingSize)
        host.layoutSubtreeIfNeeded()
        let control = try XCTUnwrap(segmentedControl(in: host))
        let store = controller.newDesignStore
        func settle() {
            for _ in 0..<3 {
                RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
                window.contentView?.layoutSubtreeIfNeeded()
                host.layoutSubtreeIfNeeded()
            }
        }
        settle()
        for page in [NDPage.backups, .jobs, .overview] {
            let index = try XCTUnwrap(NDPage.allCases.firstIndex(of: page))
            control.selectedSegment = index
            XCTAssertTrue(NSApp.sendAction(try XCTUnwrap(control.action), to: control.target, from: control), "the control's own action")
            settle()
            XCTAssertEqual(store.page, page)
            XCTAssertEqual(store.drawnPageForTesting, page, "the window root drew the page the tab names")
            XCTAssertEqual(control.selectedSegment, index, "the selected segment names the same page")
            let height = NDPageHeight.window(store, page: page, screen: nil)
            let held = controller.windowWillResize(window, to: NSSize(width: 1200, height: 300)).height
            for measured in [window.frame.height, held] {
                XCTAssertGreaterThanOrEqual(measured, height - 0.5, "\(page): the height follows the same page")
                XCTAssertLessThanOrEqual(measured, height + NDGrid.titleBar, "\(page)")
            }
        }
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
        // L17: the window is 900 wide and takes the page's own height (Overview 716, R1.1), min = max; a tab switch snaps
        // it to the next page's height with the top edge fixed.
        // AppKit may add the unified toolbar's chrome to a content size it holds (20 pt on macOS 27 headless, 24 on CI's
        // macOS 15, 0 measured on the capture Mac's macOS 26), so heights are pinned to [page, page + one 52 pt row].
        let window = controller.windowForTesting
        func holds(_ page: NDPage, _ line: UInt = #line) {
            // R1.2 change 22 and L25: the page's measured height, capped at the screen's visible height minus 40 pt.
            let h = NDPageHeight.window(controller.newDesignStore, page: page,
                                        screen: window.screen?.visibleFrame.height ?? NSScreen.main?.visibleFrame.height)
            let held = controller.windowWillResize(window, to: NSSize(width: 1400, height: 200))
            XCTAssertEqual(held.width, 900, "not resizable: the delegate holds the page size", line: line)
            XCTAssertGreaterThanOrEqual(held.height, h, line: line)
            XCTAssertLessThanOrEqual(held.height, h + NDGrid.titleBar, line: line)
            XCTAssertEqual(window.frame.size.height, held.height, accuracy: 0.5, line: line)
        }
        holds(.overview)
        let top = window.frame.maxY
        controller.fitPage(.backups)
        holds(.backups)
        XCTAssertEqual(window.frame.maxY, top, accuracy: 0.5, "the top edge stays")
        controller.fitPage(.jobs)
        holds(.jobs)
        controller.fitPage(.overview)
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
        XCTAssertEqual(size.width, 480)
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
        var c = BrainBarDesignFixture.inputs(.driveSignedOut).capture
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
        let m = NDModel.make(BrainBarDesignFixture.inputs(.driveSignedOut))
        let rows = m.areas.first { $0.thing == .safe }!.rows
        guard case let .gauge(g)? = rows.first(where: { $0.id == "backup_offsite" })?.visual else { return XCTFail("gauge") }
        XCTAssertTrue(g.concerned)
        XCTAssertEqual(g.tone, .fault)
        guard case let .gauge(local)? = rows.first(where: { $0.id == "backup_local" })?.visual else { return XCTFail("gauge") }
        XCTAssertFalse(local.concerned)
        XCTAssertEqual(local.tone, .fine)
        XCTAssertEqual(m.menuAreas.first { $0.thing == .safe }?.number.text, "27\u{00A0}h")
        XCTAssertEqual(m.backupsTabCount, 1)
    }

    func testUpkeepFailingIsOneFormOneString() {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.twoFaults))
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

    func testAgentsColumnCountsMemories() throws {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.calm))
        let agents = try XCTUnwrap(m.areas.first { $0.name == "Agents" })
        XCTAssertEqual(agents.aside, BrainBarMenuPresenter.SurfaceSay.memoriesIn24h)
        XCTAssertEqual(agents.aside, "memories in 24 h")
        for row in agents.rows {
            XCTAssertTrue(row.accessibility.contains("memories"), row.accessibility)
            XCTAssertFalse(row.accessibility.contains("saves"), row.accessibility)
        }
        XCTAssertTrue(agents.rows.contains { $0.number.value.contains(",") }, "thousands separator like the title bar")
    }

    func testMeasuredZeroDrawsNothingAndUnknownIsDashed() {
        var i = BrainBarDesignFixture.inputs(.calm)
        let m = NDModel.make(i)
        let gemini = m.areas.first { $0.name == "Agents" }!.rows.first { $0.id == "gemini" }!
        // The scale is the busiest agent's count (the fixture's Agents rows sum to its 24 h memories, L5).
        XCTAssertEqual(gemini.visual, .agent(sessions: 0, saves: 0, scale: i.agents.agents.compactMap(\.saves).max()!))
        XCTAssertEqual(gemini.number.value, "0")
        i.agents = .unknown
        let unknown = NDModel.make(i).areas.first { $0.name == "Agents" }!
        XCTAssertEqual(unknown.mark, .unknown)
        XCTAssertTrue(unknown.rows.allSatisfy { $0.number.value == "—" })
    }

    /// Design pre-pass 7: beside "Everything else is fine." the fault drops its own reassurance; 22 words at most.
    func testOneUpkeepFaultKeepsToTheWordsBudget() {
        let m = NDModel.make(BrainBarDesignFixture.inputs(.nightlyFailing))
        XCTAssertTrue(m.everythingElse)
        let shown = m.faults.flatMap(\.shownFacts).joined(separator: " ")
        XCTAssertFalse(shown.contains("still being saved"), shown)
        XCTAssertLessThanOrEqual(shown.split(separator: " ").count, 22, shown)
        XCTAssertTrue(m.faults.flatMap(\.facts).joined(separator: " ").contains("still being saved"), "VoiceOver and the report keep it")
    }

    func testStaleReadingNeverSaysNone() {
        let row = NDModel.make(BrainBarDesignFixture.inputs(.notConfirmed)).areas.first { $0.thing == .remembering }!.rows.first { $0.id == "stores" }!
        XCTAssertEqual(row.visual, .unknown)
        XCTAssertEqual(row.number.value, "—")
    }

    func testCheckedLineSaysWhenTheVerdictIsStale() {
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.calm)).checked, "Checked 1 min ago")
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.notConfirmed)).checked, "Last checked 2 h ago")
    }

    func testFixtureNamesAreTheCaptureList() {
        XCTAssertEqual(BrainBarDesignFixture.Name.allCases.map(\.rawValue),
                       ["calm", "agent-not-read", "drive-signed-out", "saves-stuck", "meaning-behind", "nightly-failing",
                        "not-confirmed", "two-faults", "reconnect-waiting", "backups-failed-upload", "backup-late"])
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
        XCTAssertEqual(NDModel.make(BrainBarDesignFixture.inputs(.notConfirmed)).drive.state, "—", "18 minutes old: not current")
        let fault = NDModel.make(BrainBarDesignFixture.inputs(.driveSignedOut)).drive
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
        XCTAssertEqual(BrainBarDesignFixture.capturePreferences.string(forKey: BrainBarDesign.key), BrainBarDesign.new.rawValue,
                       "fix 4: the capture's Settings shows the design it draws")
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

    /// Fix 5 with review B2: a capture tick moves the fixture clock one minute; the 1 h hero shifts by exactly one bucket
    /// (and the first tick after opening slides: polish 7), every wave is read again off the one series, deterministically.
    func testCaptureTickShiftsTheHourByOneBucket() throws {
        let start = BrainBarDesignFixture.inputs(.calm)
        let store = NDStore(fixture: start), twin = NDStore(fixture: start)
        store.tickFixture(); twin.tickFixture()
        let hour = try XCTUnwrap(store.inputs.heroWaves[.hour]), was = try XCTUnwrap(start.heroWaves[.hour])
        XCTAssertEqual(hour.count, was.count)
        XCTAssertEqual(Array(hour.reads.dropLast()), Array(was.reads.dropFirst()))
        XCTAssertEqual(Array(hour.saves.dropLast()), Array(was.saves.dropFirst()))
        XCTAssertEqual(hour.sequence, try XCTUnwrap(was.sequence) + 1)
        XCTAssertEqual(hour.outgoing, NDWave.Bucket(saves: was.saves[0], reads: was.reads[0]), "the first tick slides")
        XCTAssertEqual(try XCTUnwrap(hour.endsAt).timeIntervalSince(try XCTUnwrap(was.endsAt)), 60, accuracy: 0.001)
        XCTAssertEqual(store.inputs.menuWave, twin.inputs.menuWave, "deterministic")
        XCTAssertEqual(store.inputs.heroWaves, twin.inputs.heroWaves, "deterministic")
        XCTAssertEqual(store.model, NDModel.make(store.inputs), "through the sample path: the model follows")
        store.tickFixture()
        XCTAssertEqual(store.fixtureTicks, 2)
        XCTAssertEqual(store.inputs.heroWaves[.hour]?.sequence, try XCTUnwrap(was.sequence) + 2)
    }

    /// Review B1, on the layout model: through one shift the baseline rect never moves; the moving layer holds n + 1
    /// buckets while it slides and n at rest; at t = 0 it stands exactly one bucket width right, so every bar is where it
    /// was before the data changed (no hole, no kick), and the outgoing bucket is the one that left.
    func testShiftSlidesOnePitchOverAStaticBaseline() throws {
        let t0 = BrainBarDesignFixture.fixedNow
        for (minutes, count, size) in [(60, 60, NDWaveform.Size.hero), (60, 24, .menu), (180, 60, .hero), (1_440, 60, .hero)] {
            let width = Double(minutes * 60) / Double(count)
            let before = NDFixtureTraffic.wave(minutes: minutes, buckets: count, now: t0)
            let after = NDFixtureTraffic.wave(minutes: minutes, buckets: count, now: t0.addingTimeInterval(width))
                .linked(after: before, minutes: minutes)
            let target = try XCTUnwrap(after.sequence)
            XCTAssertEqual(target, try XCTUnwrap(before.sequence) + 1)
            XCTAssertEqual(after.outgoing, NDWave.Bucket(saves: before.saves[0], reads: before.reads[0]))
            for canvas in [CGSize(width: 664, height: 114), CGSize(width: 664, height: 58), CGSize(width: 172, height: 20)] {
                let (bw, gap) = NDWaveform.layout(width: canvas.width, count: count, size: size)
                let pitch = bw + gap
                XCTAssertEqual(NDWaveform.offset(target: target, phase: Double(target - 1), pitch: pitch), pitch, "t = 0: +1 pitch")
                let mid = NDWaveform.offset(target: target, phase: Double(target) - 0.5, pitch: pitch)
                XCTAssertEqual(mid, pitch / 2, accuracy: 0.0001)
                XCTAssertEqual(NDWaveform.offset(target: target, phase: Double(target), pitch: pitch), 0)
                XCTAssertEqual(NDWaveform.baseline(canvas, nowTick: true), NDWaveform.baseline(canvas, nowTick: true))
                let atStart = NDWaveform.strip(after, width: canvas.width, size: size, offset: pitch)
                let during = NDWaveform.strip(after, width: canvas.width, size: size, offset: mid)
                let rest = NDWaveform.strip(after, width: canvas.width, size: size, offset: 0)
                let shown = NDWaveform.strip(before, width: canvas.width, size: size, offset: 0)
                XCTAssertEqual(atStart.count, count + 1)
                XCTAssertEqual(during.count, count + 1)
                XCTAssertEqual(rest.count, count)
                // At t = 0 the strip's first n entries are exactly the bars on screen before the change.
                for i in 0..<count {
                    XCTAssertEqual(atStart[i].x, shown[i].x, accuracy: 0.0001, "bar \(i) does not move at t = 0")
                    XCTAssertEqual(atStart[i].bucket, shown[i].bucket, "bar \(i) is the same bar")
                }
            }
        }
        // Not one bucket later: drawn in place, no outgoing bucket.
        let a = NDFixtureTraffic.wave(minutes: 60, buckets: 60, now: t0)
        XCTAssertNil(NDFixtureTraffic.wave(minutes: 60, buckets: 60, now: t0.addingTimeInterval(180)).linked(after: a, minutes: 60).outgoing)
        XCTAssertNil(a.linked(after: a, minutes: 60).outgoing)
        XCTAssertNil(a.linked(after: nil, minutes: 60).outgoing)
    }

    /// Review B2 in the fixture: across clocks, 1 h <= 3 h <= 24 h for both series, and the Agents rows sum to the 24 h
    /// saved plus the 24 h read (L5: they count memories). Captioned with the approved noun.
    func testFixtureTotalsNest() throws {
        for minutes in [0, 1, 2, 7, 61, 190, 1_441] {
            var i = BrainBarDesignFixture.inputs(.calm)
            BrainBarDesignFixture.traffic(&i, at: BrainBarDesignFixture.fixedNow.addingTimeInterval(Double(minutes) * 60 + 17))
            let h = try XCTUnwrap(i.heroWaves[.hour]), t = try XCTUnwrap(i.heroWaves[.threeHours]), d = try XCTUnwrap(i.heroWaves[.day])
            XCTAssertLessThanOrEqual(h.savesTotal, t.savesTotal, "\(minutes)")
            XCTAssertLessThanOrEqual(t.savesTotal, d.savesTotal, "\(minutes)")
            XCTAssertLessThanOrEqual(h.readsTotal, t.readsTotal, "\(minutes)")
            XCTAssertLessThanOrEqual(t.readsTotal, d.readsTotal, "\(minutes)")
            // L5 ruling: the Agents rows count memories (saved plus read), every one attributed in the fixture.
            XCTAssertEqual(i.agents.agents.compactMap(\.saves).reduce(0, +), d.savesTotal + d.readsTotal, "\(minutes)")
        }
    }

    /// Fix 6: every Reconnect a person can click (dropdown row, needs band, Backups row, Settings) reaches the one store
    /// action, and in a capture that shows the waiting line.
    func testEveryReconnectClickShowsTheWaitingLine() {
        for press in [{ (s: NDStore) in s.perform(.reconnectDrive) }, { $0.perform(.connectDrive) }, { $0.reconnectDrive() }] {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(.driveSignedOut))
            XCTAssertFalse(store.inputs.driveReconnecting)
            press(store)
            XCTAssertTrue(store.inputs.driveReconnecting)
        }
    }

    /// Empty Settings window: the app's SwiftUI scene graph holds no scene with a window (Settings, WindowGroup, Window,
    /// DocumentGroup), so neither AppKit's launch-time open-untitled request, a reopen, state restoration nor
    /// `showSettingsWindow:` reaching SwiftUI's own app delegate can present one; its one scene is a menu-bar extra that is
    /// never inserted (no status item). Both designs.
    func testTheAppHasNoSceneThatCanPresentAWindow() throws {
        let scenes = String(describing: BrainBarApp.Body.self)
        XCTAssertTrue(scenes.contains("MenuBarExtra"), scenes)
        for windowed in ["Settings<", "SettingsScene", "WindowGroup", "Window<", "DocumentGroup", "UtilityWindow"] {
            XCTAssertFalse(scenes.contains(windowed), "\(windowed) in \(scenes)")
        }
        let source = try String(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBar/BrainBarApp.swift"), encoding: .utf8)
        XCTAssertTrue(source.contains("MenuBarExtra(\"BrainBar\", isInserted: .constant(false))"), "never inserted: no status item")
        XCTAssertFalse(source.split(separator: "\n").contains { $0.trimmingCharacters(in: .whitespaces).hasPrefix("Settings {") },
                       "no SwiftUI Settings scene")
    }

    /// Fix 10: a switched-off job says "off" in Ink 2; its switch's help names the job and the action.
    func testSwitchedOffJobSaysOff() throws {
        var i = BrainBarDesignFixture.inputs(.calm)
        let index = try XCTUnwrap(i.jobs?.firstIndex { !$0.continuous })
        i.jobs![index].enabled = false
        let store = NDStore(fixture: i, page: .jobs)
        let lane = try XCTUnwrap(NDJobsPage(store: store).lanes(store.model).first { $0.job == i.jobs![index].job })
        XCTAssertEqual(lane.number.value, BrainBarMenuPresenter.SurfaceSay.off)
        XCTAssertTrue(lane.number.secondary)
        XCTAssertFalse(lane.events.contains { $0.kind == .next }, "a switched-off job has no next run")
        XCTAssertEqual(BrainBarMenuPresenter.SurfaceSay.turnOn(lane.name), "Turn on " + lane.name)
        XCTAssertEqual(BrainBarMenuPresenter.SurfaceSay.turnOff(lane.name), "Turn off " + lane.name)
    }

    /// Fix 11: the report sentence never breaks, so neither "an agent." nor "to an agent." can stand alone.
    func testReportSentenceIsOneUnit() {
        let report = BrainBarMenuPresenter.Say.giveReport.text
        XCTAssertFalse(keepingUnits("Fact one. " + report).hasSuffix(" " + report.split(separator: " ").suffix(2).joined(separator: " ")))
        XCTAssertFalse(keepingUnits(report).contains(" "))
    }

    /// R1 § 3.1: 6 pt bars with 2 pt corners in the window, 3 pt bars with 1 pt in the dropdown; never more than half a bar.
    func testShortHeroBarsAreNotBeads() {
        XCTAssertEqual(NDWaveform.barRadius(height: 40, width: 6, pill: false, corner: 2), 2)
        XCTAssertEqual(NDWaveform.barRadius(height: 2, width: 6, pill: false, corner: 2), 1, "a 2 pt bar is not a bead")
        XCTAssertEqual(NDWaveform.barRadius(height: 9, width: 3, pill: false, corner: 1), 1)
        XCTAssertEqual(NDWaveform.layout(width: 800, count: 60, size: .hero).bar, 6)
        XCTAssertEqual(NDWaveform.layout(width: 128, count: 24, size: .menu).bar, 3)
    }

    /// Fix 14: an area draws the unknown ring only when one of its rows is dashed; otherwise the reason is spoken.
    func testRingOnlyBesideADashedRow() {
        for name in BrainBarDesignFixture.Name.allCases {
            let m = NDModel.make(BrainBarDesignFixture.inputs(name))
            for area in m.areas {
                if area.shownMark == .unknown {
                    XCTAssertTrue(area.rows.contains { $0.visual == .unknown }, "\(name.rawValue) \(area.name)")
                } else if area.mark == .unknown {
                    XCTAssertNotNil(area.unknownReason, "\(name.rawValue) \(area.name)")
                }
            }
            for area in m.menuAreas where area.shownMark == .unknown {
                XCTAssertEqual(area.visual, .unknown, "\(name.rawValue) \(area.name)")
            }
        }
    }

    /// Fix 15: the last run and the next run both within 6 pt of now: only the next-run ring is drawn.
    func testRailDrawsOnlyTheNextRingBesideNow() {
        let close: [NDRail.Event] = [.init(hours: -0.05, kind: .ok), .init(hours: 0.05, kind: .next)]
        XCTAssertEqual(NDRail.shown(close, width: 480, past: 36, ahead: 12), [close[1]])
        let apart: [NDRail.Event] = [.init(hours: -5, kind: .ok), .init(hours: 0.05, kind: .next)]
        XCTAssertEqual(NDRail.shown(apart, width: 480, past: 36, ahead: 12), apart)
        let noNext: [NDRail.Event] = [.init(hours: -0.05, kind: .ok)]
        XCTAssertEqual(NDRail.shown(noNext, width: 480, past: 36, ahead: 12), noNext)
    }

    /// Review B3: on the Backups page "Newest copy" and the "Copy on this Mac" lane are both unknown, or both known and
    /// within an hour of each other, in every fixture, the stale one included.
    func testNewestCopyFollowsTheLaneOnTheBackupsPage() throws {
        let units: [String: Double] = ["s": 1, "min": 60, "h": 3600, "d": 86_400]
        var sawUnknown = false
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .backups)
            let page = NDBackupsPage(store: store)
            let lane = try XCTUnwrap(page.lanes(store.model).first { $0.id == "backup_local" })
            let fact = page.newestCopyFact(store.model)
            if lane.number.value == NDModel.Number.unknown.value {
                sawUnknown = true
                XCTAssertEqual(fact, "—", name.rawValue)
                continue
            }
            XCTAssertNotEqual(fact, "—", name.rawValue)
            let age = try XCTUnwrap(Double(lane.number.value)) * (try XCTUnwrap(units[lane.number.unit]))
            let newest = try XCTUnwrap(store.inputs.newestLocalCopyAt)
            XCTAssertEqual(store.inputs.capture.now.timeIntervalSince(newest), age, accuracy: 3600, name.rawValue)
        }
        XCTAssertTrue(sawUnknown, "the stale fixture is covered")
    }

    /// Review N1: on the Backups page a row's rail has filled dots (a confirmed copy, or a failed attempt) only if its
    /// newest cell is known; an unknown newest keeps the attempts' places as not-confirmed rings with a dashed line from
    /// the first to now. Every fixture, both uncertain states included.
    func testBackupRailClaimsNoCopyItsNewestCellCallsUnknown() throws {
        var unknownRows = 0
        for name in BrainBarDesignFixture.Name.allCases {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: .backups)
            for lane in NDBackupsPage(store: store).lanes(store.model) {
                let filled = lane.events.contains { $0.kind == .ok || $0.kind == .fail }
                let known = lane.number.value != NDModel.Number.unknown.value
                if !known {
                    unknownRows += 1
                    XCTAssertFalse(filled, "\(name.rawValue) \(lane.name)")
                    let past = lane.events.filter { $0.kind != .next }
                    XCTAssertTrue(past.allSatisfy { $0.kind == .unconfirmed && $0.hours <= 0 }, "\(name.rawValue) \(lane.name)")
                    XCTAssertEqual(lane.unconfirmedFrom, past.map(\.hours).min(), "\(name.rawValue) \(lane.name)")
                } else {
                    XCTAssertFalse(lane.events.contains { $0.kind == .unconfirmed }, "\(name.rawValue) \(lane.name)")
                    XCTAssertNil(lane.unconfirmedFrom)
                }
                XCTAssertTrue(lane.events.filter { $0.kind == .next }.allSatisfy { $0.hours >= 0 }, "next rings sit right of now")
            }
        }
        XCTAssertGreaterThan(unknownRows, 0, "checking and cant-tell are covered")
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

/// Round 1 (DESIGN-SPEC-R1 with revision 1.1): the pipeline graph, the drawn state of every fixture from its verdict
/// facts, the headline and button rules, and the dropdown's words.
@MainActor final class BrainBarNewDesignR1Tests: XCTestCase {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    typealias Fixture = BrainBarDesignFixture

    /// L18: every health check maps to connections, a stage, or the observer; every mapping names a connection that
    /// exists; every connection is guarded by a check or listed as a known gap.
    /// The copy deck names jobs by their launchd raw value (it imports no settings types); every key is a real job.
    func testJobWordsAreKeyedByRealJobs() {
        let jobs = Set(BrainLayerLaunchdJob.allCases.map(\.rawValue))
        XCTAssertTrue(Set(Say.jobNames.keys).isSubset(of: jobs))
        XCTAssertEqual(Set(Say.jobNames.keys), Set(Say.jobHelp.keys))
        XCTAssertEqual(Set(Say.jobNames.keys), jobs.subtracting(["index"]), "every scheduled job has a name and a help line")
    }

    func testEveryHealthCheckMapsInTheGraph() {
        let checks = Set(BrainBarHealthVerdict.Check.labels.keys)
        XCTAssertFalse(checks.isEmpty)
        for name in checks { XCTAssertNotNil(NDPipelineGraph.checks[name], "\(name) maps to nothing") }
        let ids = Set(NDPipelineGraph.connections.map(\.id))
        var guarded: Set<String> = []
        for (name, rule) in NDPipelineGraph.checks {
            if case let .connections(list) = rule {
                for id in list { XCTAssertTrue(ids.contains(id), "\(name) names \(id), which is not a connection") }
                guarded.formUnion(list)
            }
        }
        XCTAssertEqual(ids.subtracting(guarded).subtracting(NDPipelineGraph.factGuards.keys), NDPipelineGraph.unguarded,
                       "a connection with no check is guarded by a named fact or is a listed gap")
        XCTAssertEqual(Set(NDPipelineGraph.connections.flatMap(\.pipes)), Set(NDPipe.allCases), "every drawn pipe belongs to a connection")
        XCTAssertEqual(NDPipelineGraph.checks["monitor"], .observer)
    }

    private func pipeline(_ name: Fixture.Name) -> NDPipelineState { NDPipelineState.make(Fixture.inputs(name)) }

    /// § 3.2's seven states and the others, each drawn from its own facts.
    func testEachFixtureDrawsItsState() throws {
        let calm = pipeline(.calm)
        XCTAssertTrue(calm.breaks.isEmpty)
        XCTAssertTrue(calm.faultStations.isEmpty)
        XCTAssertTrue(calm.quiet.isEmpty)
        XCTAssertFalse(calm.allUnconfirmed)
        XCTAssertEqual(calm.status[.read]?.text, Say.lastRead(NDFormat.ageText(48)))
        XCTAssertEqual(calm.status[.queue]?.text, Say.empty)
        XCTAssertEqual(calm.status[.mac]?.text, Say.lastBackup(NDFormat.ageText(4 * 3600)))
        XCTAssertEqual(calm.status[.drive]?.text, Say.lastBackup(NDFormat.ageText(4 * 3600)), "the older of the two Drive copies")
        XCTAssertEqual(calm.status[.jobs]?.text, Say.lastCheckUp(NDFormat.ageText(5 * 3600)))
        XCTAssertEqual(calm.status[.meaning]?.text, Say.percentReady(99))
        XCTAssertEqual(calm.status[.search]?.text, Say.working)
        XCTAssertTrue(calm.status.values.allSatisfy { $0.severity == nil })

        let agent = pipeline(.agentNotRead)
        XCTAssertEqual(agent.breaks, [.agentCursor: .init(side: .source, severity: .red)])
        XCTAssertEqual(agent.redAgents, [.cursor])
        XCTAssertEqual(agent.phrases.map(\.text), [Say.notBeingRead])

        for name in [Fixture.Name.driveSignedOut, .reconnectWaiting] {
            let drive = pipeline(name)
            XCTAssertEqual(drive.faultStations, [.drive], name.rawValue)
            XCTAssertEqual(drive.stationPipes, [.macDrive, .conversationsCopy], name.rawValue)
            XCTAssertTrue(drive.breaks.isEmpty, "one cause, drawn on the station")
            XCTAssertEqual(drive.status[.drive], .init(text: Say.signedOut, severity: .red))
            XCTAssertEqual(drive.status[.mac]?.severity, nil, "the Mac copy is still fresh")
        }

        let saves = pipeline(.savesStuck)
        XCTAssertEqual(saves.breaks, [.queueMemory: .init(side: .target, severity: .red)])
        XCTAssertEqual(saves.status[.queue], .init(text: Say.waiting(NDFormat.ageText(2 * 3600)), severity: .red))

        let meaning = pipeline(.meaningBehind)
        XCTAssertEqual(meaning.breaks, [.memoryMeaning: .init(side: .target, severity: .orange)])
        XCTAssertEqual(meaning.quiet, [.meaningSearch: .orange], "downstream: delayed")
        XCTAssertEqual(meaning.status[.meaning], .init(text: Say.behind(NDFormat.ageText(2 * 3600)), severity: .orange))
        XCTAssertEqual(meaning.status[.search]?.text, Say.working, "search still works by words")

        let nightly = pipeline(.nightlyFailing)
        XCTAssertEqual(nightly.breaks, [.jobsMemory: .init(side: .source, severity: .red)])
        XCTAssertEqual(nightly.status[.jobs], .init(text: Say.nightlyFailedShort(3), severity: .red), "the status names the job")

        let stale = pipeline(.notConfirmed)
        XCTAssertTrue(stale.allUnconfirmed)
        XCTAssertNotNil(stale.unconfirmedSince)
        XCTAssertTrue(stale.breaks.isEmpty && stale.faultStations.isEmpty, "no red, no marks")
        XCTAssertFalse(stale.status.isEmpty, "status lines keep their last known values (faded)")

        let two = pipeline(.twoFaults)
        XCTAssertEqual(two.faultStations, [.drive])
        XCTAssertEqual(Set(two.breaks.keys), [.jobsMemory])

        XCTAssertEqual(pipeline(.backupsFailedUpload), calm, "a failed upload last week is history, not a break now")
    }

    /// Every check the chart can draw, from the verdict's facts alone: red on a calm verdict draws on its own pipes.
    func testEachRedCheckBreaksItsOwnConnection() {
        let expected: [String: (NDPipelineState) -> Bool] = [
            "ingest": { $0.breaks[.junctionRead]?.side == .target && $0.status[.read]?.severity == .red },
            "stores": { $0.breaks[.inSaves]?.side == .source },
            "embedding": { $0.breaks[.memoryMeaning]?.side == .target && $0.quiet[.meaningSearch] == .red },
            "search": { $0.breaks[.out]?.side == .source },
            "backup_local": { $0.breaks[.memoryMac]?.side == .target },
            "backup_offsite": { $0.breaks[.macDrive]?.side == .target },
            "transcripts": { $0.breaks[.conversationsCopy]?.side == .target },
            "drive_auth": { $0.faultStations.contains(.drive) && $0.stationPipes == [.macDrive, .conversationsCopy] },
            "integrity": { $0.faultStations == [.mac] && $0.breaks.isEmpty },
            "disk": { $0.faultStations == [.mac] && $0.breaks.isEmpty },
            "monitor": { $0.allUnconfirmed && $0.breaks.isEmpty },
        ]
        XCTAssertEqual(Set(expected.keys), Set(NDPipelineGraph.checks.keys))
        for (name, holds) in expected {
            var i = Fixture.inputs(.calm)
            i.capture = Fixture.capture(Fixture.replacing(Fixture.calmChecks(), Fixture.check(name, .red, age: 2 * 3600, limit: 3600)), dot: .red)
            let drawn = NDPipelineState.make(i)
            XCTAssertTrue(holds(drawn), "\(name): \(drawn)")
            // Only its own pipes: nothing else breaks.
            let own = Set(NDPipelineGraph.pipes(for: name))
            XCTAssertTrue(Set(drawn.breaks.keys).isSubset(of: own), "\(name) broke \(drawn.breaks.keys) outside \(own)")
        }
        // A check that cannot say: its pipes are drawn "not confirmed", nothing breaks.
        var unknown = Fixture.inputs(.calm)
        unknown.capture = Fixture.capture(Fixture.replacing(Fixture.calmChecks(), Fixture.check("backup_offsite", .unknown)))
        let drawn = NDPipelineState.make(unknown)
        XCTAssertEqual(drawn.unconfirmed, [.macDrive])
        XCTAssertTrue(drawn.breaks.isEmpty)
        // The word path is never broken and never "flowing": nothing checks it.
        XCTAssertEqual(NDPipelineChart.stroke(for: .words, in: .calm), .unguarded)
    }

    /// Headlines, details and the one filled button (R1.1 change 23: Reconnect only; the report is always bordered).
    func testHeadlinesAndTheOneFilledButton() {
        func r1(_ name: Fixture.Name) -> NDR1 {
            let i = Fixture.inputs(name)
            return NDR1.make(i, NDModel.make(i))
        }
        XCTAssertEqual(r1(.calm).headline, BrainBarMenuPresenter.Say.allGood.text)
        XCTAssertTrue(r1(.calm).items.isEmpty)
        // R1.2 changes 4 to 7 skipped (lead, 09:55): one red item still names itself in the headline.
        XCTAssertEqual(r1(.agentNotRead).headline, Say.agentNotRead("Cursor"))
        XCTAssertEqual(r1(.driveSignedOut).headline, Say.driveSignedOut)
        XCTAssertEqual(r1(.savesStuck).items.first?.detail, Say.savesWaited(38, NDFormat.ageText(2 * 3600)))
        XCTAssertEqual(r1(.meaningBehind).headline, Say.meaningCatchingUp, "D6: amber is catching up, not a fault")
        XCTAssertEqual(r1(.meaningBehind).items.map(\.severity), [.orange], "late, not stopped: one orange item, no presenter echo")
        XCTAssertNil(r1(.meaningBehind).items.first?.action, "nothing to do")
        XCTAssertEqual(r1(.nightlyFailing).items.first?.detail, Say.notCheckedSince("Thursday"))
        XCTAssertEqual(r1(.notConfirmed).headline, Say.notConfirmedSince("06:00"))
        XCTAssertEqual(r1(.notConfirmed).footerLeft, Say.waitingForNextCheck)
        XCTAssertNil(r1(.notConfirmed).footerRight)
        XCTAssertEqual(r1(.twoFaults).headline, Say.thingsNeedYou(2))
        XCTAssertEqual(r1(.twoFaults).items.map(\.id), ["drive", "nightly"])
        for name in Fixture.Name.allCases {
            for item in r1(name).items {
                guard let action = item.action else { continue }
                XCTAssertEqual(NDButtonPolicy.filled(action), [.reconnectDrive, .connectDrive].contains(action.kind), name.rawValue)
            }
        }
        XCTAssertEqual(r1(.twoFaults).items.first?.menuDetail, Say.lastDriveUpload(NDFormat.ageText(27 * 3600)))
        XCTAssertEqual(r1(.twoFaults).items.last?.menuTitle, Say.nightlyFailedShort(3))
    }

    /// § 4.1: the calm dropdown says what it must in 9 words (a number with its unit counts as nothing).
    func testCalmDropdownIsNineWords() throws {
        let i = Fixture.inputs(.calm)
        let r1 = NDR1.make(i, NDModel.make(i))
        let words = ([NDMenuHeadline.text(r1)] + r1.menuRows.flatMap { [$0.label, $0.value] }).joined(separator: " ")
        let stripped = words.replacing(#/\d+(?:[.,]\d+)?\s?(?:%|h|min|s|d)?/#, with: "")
        let count = stripped.split(whereSeparator: { $0 == " " || $0 == "\u{00A0}" }).count
        XCTAssertEqual(count, 9, stripped)
        XCTAssertEqual(r1.menuRows.map(\.value), ["", Say.percentReady(99), Say.ago(NDFormat.ageText(4 * 3600)), Say.fine])
        let two = NDR1.make(Fixture.inputs(.twoFaults), NDModel.make(Fixture.inputs(.twoFaults)))
        XCTAssertEqual(two.menuRows.map(\.value)[2], Say.driveAgo(NDFormat.ageText(27 * 3600)),
                       "D7.3: the stale copy named, one number for one fact (as 'Nothing has uploaded for 27 h')")
        XCTAssertEqual(two.menuRows[2].tone, .fault)
        XCTAssertEqual(two.menuRows[3].value, Say.failed)
    }

    /// L17 and the clip defect: in every fixture, on every page, and in a constructed three-fault state, the laid-out
    /// content ends at least the bottom inset (24 pt) above the footer, and the window is never below the spec's height.
    func testNoPageIsClippedUnderTheFooter() throws {
        var three = Fixture.inputs(.twoFaults)
        three.capture.verdict = BrainBarHealthVerdict(
            dot: .red, checks: Fixture.replacing(three.capture.verdict!.checks, Fixture.check("stores", .red, age: 7200, limit: 1800)),
            notes: [], evaluatedAt: three.capture.verdict!.evaluatedAt)
        three.savesWaiting = 38
        three.oldestSaveQueuedAt = Fixture.fixedNow.addingTimeInterval(-7200)
        var cases: [(String, NDInputs)] = Fixture.Name.allCases.map { ($0.rawValue, Fixture.inputs($0)) }
        cases.append(("three-faults", three))
        var heights: [String] = []
        for (name, inputs) in cases {
            for page in NDPage.allCases {
                let store = NDStore(fixture: inputs, page: page)
                let window = NDPageHeight.window(store, page: page, screen: nil)
                let content = NDPageHeight.content(store, page: page)
                // title bar 52 + top inset 20 + content + bottom inset 20 + footer 28 (R1.2 change 22: no floor)
                XCTAssertLessThanOrEqual(NDGrid.titleBar + NDGrid.pageTop + content + 20 + NDGrid.footer, window + 0.5, "\(name) \(page)")
                if page == .overview { heights.append("\(name) \(Int(window))") }
            }
        }
        let calm = NDPageHeight.window(NDStore(fixture: Fixture.inputs(.calm), page: .overview), page: .overview, screen: nil)
        let twoFaults = NDPageHeight.window(NDStore(fixture: Fixture.inputs(.twoFaults), page: .overview), page: .overview, screen: nil)
        XCTAssertGreaterThan(twoFaults, calm, "two Problems rows are taller, so the window is")
        // L25: capped at the screen's visible height minus 40 pt; past it, the page scrolls and the footer stays.
        XCTAssertEqual(NDPageHeight.window(NDStore(fixture: three, page: .overview), page: .overview, screen: 720), 680)
        print("Overview heights: " + heights.joined(separator: ", "))
    }

    /// Ruling on change 17: more than five agents cannot occur while there are four agent families; this fails loudly the
    /// day a fifth family appears, so the "+N more" list is built then.
    func testAgentFamiliesStayWithinTheChartsList() {
        XCTAssertLessThanOrEqual(AgentFamily.allCases.count, 5, "a fifth agent family: build change 17a (+N more)")
        XCTAssertEqual(NDPipelineLayout.agents.count, AgentFamily.allCases.count)
    }

    /// D2 (real screen, round 1): one state, one truth. For every fixture, no dropdown row shows a fine value for an area
    /// whose stages read late, broken or not confirmed on the Overview (the same `condition` rule).
    func testDropdownRowsNeverClaimFineWhereTheOverviewDoesNot() {
        let stagesOf: [NDR1.MenuRow.Kind: [NDStation]] = [.search: [.meaning, .search], .backups: [.mac, .drive], .jobs: [.jobs]]
        var checkedNotFine = 0
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let r1 = NDR1.make(i, NDModel.make(i))
            for row in r1.menuRows where row.kind != .activity {
                let conditions = (stagesOf[row.kind] ?? []).map(r1.pipeline.condition)
                guard conditions.contains(where: { $0 != .fine }) else { continue }
                checkedNotFine += 1
                XCTAssertTrue(row.tone != .fine || row.value == NDModel.Number.unknown.value,
                              "\(name.rawValue): \(row.label) says \(row.value) while its stage reads \(conditions)")
            }
        }
        XCTAssertGreaterThan(checkedNotFine, 0)
        let behind = NDR1.make(Fixture.inputs(.meaningBehind), NDModel.make(Fixture.inputs(.meaningBehind))).menuRows[1]
        XCTAssertEqual(behind.value, Say.behind(NDFormat.ageText(2 * 3600)), "the Search row leads with what is wrong")
        XCTAssertEqual(behind.tone, .warn)
        let stale = NDR1.make(Fixture.inputs(.notConfirmed), NDModel.make(Fixture.inputs(.notConfirmed))).menuRows
        XCTAssertEqual(stale.dropFirst().map(\.value), Array(repeating: NDModel.Number.unknown.value, count: 3))
    }

    /// D3: nothing confirmed, nothing calm: the icon is comp 01's dimmed glyph whenever every pipe is not confirmed.
    func testIconIsDimmedWhenNothingIsConfirmed() {
        func dimmed(_ name: Fixture.Name) -> Bool {
            let i = Fixture.inputs(name)
            let m = NDModel.make(i)
            return NDIcon.dimmed(m, NDR1.make(i, m))
        }
        XCTAssertTrue(dimmed(.notConfirmed))
        XCTAssertFalse(dimmed(.calm))
        XCTAssertFalse(dimmed(.driveSignedOut))
        var unread = Fixture.inputs(.calm)
        unread.capture.verdict = nil
        let m = NDModel.make(unread)
        XCTAssertTrue(NDIcon.dimmed(m, NDR1.make(unread, m)), "no reading yet")
    }

    /// D4: every popover opens below its station, so none covers the Activity chart above the pipeline; none covers a
    /// station it shares a pipe with, except the hub's over Jobs (directly under it; every edge of the hub has a neighbour).
    func testStationPopoversOpenBelowAndClearTheirNeighbours() {
        let neighbours: [NDStation: Set<NDStation>] = [
            .read: [.memory], .queue: [.memory], .mac: [.memory, .drive], .drive: [.mac],
            .meaning: [.memory, .search], .search: [.meaning], .jobs: [.memory],
        ]
        let i = Fixture.inputs(.calm)
        let m = NDModel.make(i)
        for station in NDStation.allCases {
            XCTAssertEqual(NDStationButton.arrowEdge(station), .bottom, "\(station)")
            let rows = CGFloat(NDStationDetails.lines(station, m, i).count)
            let height = 12 + 22 + rows * 24 + (NDStationDetails.page(station) == nil ? 0 : 32) + 12
            let anchor = station == .memory ? CGPoint(x: NDPipelineLayout.hub.midX, y: NDPipelineLayout.hub.maxY)
                : CGPoint(x: NDPipelineLayout.point(station).x, y: NDPipelineLayout.point(station).y + 13)
            let x = min(max(anchor.x - 138, 12), NDPipelineLayout.size.width - 12 - 276)
            let rect = CGRect(x: x, y: anchor.y + 10, width: 276, height: height)
            XCTAssertGreaterThan(rect.minY, 0, "\(station) never reaches the Activity chart above the pipeline")
            for other in neighbours[station] ?? [] where other != .memory {
                let p = NDPipelineLayout.point(other)
                XCTAssertFalse(rect.intersects(CGRect(x: p.x - 13, y: p.y - 13, width: 26, height: 26)), "\(station) covers \(other)")
            }
        }
    }

    /// D5: the Reconnect → waiting swap holds its layout: the headline and the page measure the same before and during,
    /// and the page's height key (what triggers a refit) does not change, so no refit runs mid-swap.
    func testReconnectSwapHoldsTheLayout() {
        var waiting = Fixture.inputs(.driveSignedOut)
        let before = NDStore(fixture: waiting)
        waiting.driveReconnecting = true
        let during = NDStore(fixture: waiting)
        XCTAssertEqual(NDPageHeight.content(before, page: .overview), NDPageHeight.content(during, page: .overview))
        XCTAssertEqual(NDPageHeight.key(before), NDPageHeight.key(during))
        func headline(_ store: NDStore) -> CGSize {
            NSHostingController(rootView: NDHeadline(store: store, r1: store.r1).frame(width: 852))
                .sizeThatFits(in: CGSize(width: 852, height: 1000))
        }
        XCTAssertEqual(headline(before), headline(during))
    }

    /// A past failed upload under "All good" is said in words: the Backups legend carries "failed" beside the red dot.
    func testAPastFailedUploadIsExplainedOnTheBackupsPage() {
        let store = NDStore(fixture: Fixture.inputs(.backupsFailedUpload), page: .backups)
        let lanes = NDBackupsPage(store: store).lanes(store.model)
        // R1.2 change 13: retried the same night, so a hollow ring and its own legend words, never a filled red dot.
        XCTAssertTrue(lanes.contains { $0.events.contains { $0.kind == .retried } })
        XCTAssertTrue(NDBackupsPage.legend(lanes).contains(Say.failedThenRetried))
        XCTAssertEqual(store.r1.headline, BrainBarMenuPresenter.Say.allGood.text)
        let calm = NDStore(fixture: Fixture.inputs(.calm), page: .backups)
        XCTAssertFalse(NDBackupsPage.legend(NDBackupsPage(store: calm).lanes(calm.model)).contains(Say.failed))
        XCTAssertFalse(NDBackupsPage.legend(NDBackupsPage(store: calm).lanes(calm.model)).contains(Say.failedThenRetried))
    }

    /// D6 (L20 rewritten): red, "needs you", the count, the badges and the fault icon are for red verdicts only. For every
    /// fixture: the fault icon iff at least one station is red; "needs you" iff a red item; badges count red items only.
    func testAmberIsNeverShownAsAFault() {
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let m = NDModel.make(i)
            let r1 = NDR1.make(i, m)
            let p = r1.pipeline
            let redStation = !p.allUnconfirmed && NDStation.allCases.contains { station in
                if case .issue(_, .red) = p.condition(station) { return true }
                return false
            }
            let icon = NDIcon.marked(m, r1)
            XCTAssertEqual(icon.icon == .marked, redStation, "\(name.rawValue): fault icon iff a red station")
            XCTAssertEqual(icon.count, r1.redCount, name.rawValue)
            let needsYou = NDMenuHeadline.text(r1).contains("need")
            XCTAssertEqual(needsYou, r1.redCount > 0, "\(name.rawValue): needs you iff red")
            XCTAssertLessThanOrEqual(r1.backupsBadge + r1.jobsBadge, r1.redCount, name.rawValue)
        }
        let i = Fixture.inputs(.meaningBehind)
        let r1 = NDR1.make(i, NDModel.make(i))
        XCTAssertEqual(r1.redCount, 0)
        XCTAssertEqual(r1.headlineTone, .warn)
        XCTAssertEqual(r1.items.first?.detail, Say.meaningBehindDetail)
        XCTAssertNil(r1.items.first?.action, "nothing to do: no button")
        XCTAssertEqual(NDIcon.marked(NDModel.make(i), r1).icon, .resting, "the resting brain for an amber state")
        XCTAssertEqual(NDMenuHeadline.text(r1), Say.meaningCatchingUp)
        // An amber ingest lag is catching up, never "needs you" (the overnight false alarm).
        var lag = Fixture.inputs(.calm)
        lag.capture = Fixture.capture(Fixture.replacing(Fixture.calmChecks(), Fixture.check("ingest", .amber, age: 1200, limit: 900)), dot: .amber)
        let lr = NDR1.make(lag, NDModel.make(lag))
        XCTAssertEqual(lr.redCount, 0)
        XCTAssertFalse(NDMenuHeadline.text(lr).contains("need"), NDMenuHeadline.text(lr))
        XCTAssertEqual(NDIcon.marked(NDModel.make(lag), lr).icon, .resting)
        XCTAssertTrue(lr.items.allSatisfy { $0.action == nil }, "no report button for a lag")
    }

    /// D7: rows of their own in the two-fault headline; timed statuses; the selected palette redraws; one capture clock.
    func testCouncilRoundOneFixes() throws {
        let two = NDStore(fixture: Fixture.inputs(.twoFaults))
        let size = NSHostingController(rootView: NDHeadline(store: two, r1: two.r1).frame(width: 852))
            .sizeThatFits(in: CGSize(width: 852, height: 1000))
        XCTAssertGreaterThanOrEqual(size.height, 22 + 6 + 2 * 48 + 8, "each fault a 48 pt row, 8 pt apart")
        XCTAssertEqual(two.r1.items.map(\.menuDetail), [Say.lastDriveUpload(NDFormat.ageText(27 * 3600)),
                                                        Say.notCheckedSinceShort("Thursday")])
        XCTAssertNotEqual(NDPalette.menu, NDPalette.menuSelected, "a selected row's chart redraws in the selected colours")
        let pinned = BrainBarDesignFixture.makeNow(capturing: true, wall: Date(), pinned: "2026-11-16T02:24:30+02:00")
        XCTAssertEqual(pinned, ISO8601DateFormatter().date(from: "2026-11-16T02:24:00+02:00"), "BRAINBAR_DESIGN_NOW pins the clock")
        XCTAssertEqual(BrainBarDesignFixture.makeNow(capturing: false, wall: Date(), pinned: "2026-11-16T02:24:30+02:00"),
                       BrainBarDesignFixture.fixedNow, "tests and renders keep the fixed moment")
    }

    /// D8.1: the hover readout is an overlay: the plot's laid-out frame is identical with and without a hovered bucket.
    func testHoverReadoutNeverMovesThePlot() throws {
        let store = NDStore(fixture: Fixture.inputs(.calm))
        func plotFrame(hover: Int?) -> CGRect? {
            var frame: CGRect?
            let host = NSHostingView(rootView: NDActivitySection(store: store, hover: hover) { frame = $0 }.frame(width: 852))
            host.frame = NSRect(x: 0, y: 0, width: 852, height: 220)
            host.layoutSubtreeIfNeeded()
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
            host.layoutSubtreeIfNeeded()
            return frame
        }
        let rest = try XCTUnwrap(plotFrame(hover: nil))
        for bucket in [0, 30, 59] {
            XCTAssertEqual(try XCTUnwrap(plotFrame(hover: bucket)), rest, "bucket \(bucket)")
        }
        XCTAssertEqual(Say.upTo("4", minutes: 1), "up to 4 / min", "R1.2 change 20")
        XCTAssertEqual(Say.upTo("83", minutes: 3), "up to 83 / 3 min")
    }

    /// D8.2 and D8.3: the waiting line starts inside the button's 100 ms fade (no empty frame); the dark read bars are the
    /// lifted #7D8597 (4.64:1 on bg, recorded in the build log).
    func testReconnectSwapHasNoGapAndReadBarsAreLifted() {
        XCTAssertLessThan(NDMotion.waitingDelay, 0.1, "the line arrives before the button has gone")
        // R1.2 change 24: the line fades in from 64 ms, 40 % of the button's 160 ms fade.
        XCTAssertEqual(NDMotion.waitingDelay, 0.064, accuracy: 0.0001)
        let dark = NDColor.read.usingColorSpace(.sRGB).flatMap { _ in
            NSAppearance(named: .darkAqua).map { appearance -> NSColor? in
                var resolved: NSColor?
                appearance.performAsCurrentDrawingAppearance { resolved = NDColor.read.usingColorSpace(.sRGB) }
                return resolved
            } ?? nil
        }
        let c = dark ?? NSColor.black
        // R1.2 change 21: #8189A0 in dark.
        XCTAssertEqual(Int((c.redComponent * 255).rounded()), 0x81)
        XCTAssertEqual(Int((c.greenComponent * 255).rounded()), 0x89)
        XCTAssertEqual(Int((c.blueComponent * 255).rounded()), 0xA0)
    }

    /// L23 over every fixture: the red icon iff something is red; the amber dot iff an amber notice that is not a catching-
    /// up state and nothing red; the "?" badge iff not confirmed (or no reading yet); otherwise the resting brain.
    func testIconPredicateOverEveryFixture() {
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let m = NDModel.make(i)
            let r1 = NDR1.make(i, m)
            let state = NDIcon.state(m, r1)
            if r1.redCount > 0 {
                XCTAssertEqual(state, .red(r1.redCount), name.rawValue)
            } else if r1.pipeline.allUnconfirmed {
                XCTAssertEqual(state, .unconfirmed, name.rawValue)
            } else if r1.items.contains(where: { $0.severity == .orange && !$0.catchingUp }) {
                XCTAssertEqual(state, .amber, name.rawValue)
            } else {
                XCTAssertEqual(state, .rest, name.rawValue)
            }
        }
        func state(_ name: Fixture.Name) -> NDIcon.State {
            let i = Fixture.inputs(name)
            let m = NDModel.make(i)
            return NDIcon.state(m, NDR1.make(i, m))
        }
        XCTAssertEqual(state(.meaningBehind), .rest, "catching up keeps the resting brain")
        XCTAssertEqual(state(.backupLate), .amber, "a late copy: Check soon, the amber dot")
        XCTAssertEqual(state(.notConfirmed), .unconfirmed)
        XCTAssertEqual(state(.twoFaults), .red(2))
    }

    /// R1.2 changes 5, 8 and 9, L24: Reconnect is the only filled button and at most one per page; amber notices never get
    /// a button; one amber notice says itself (catching up by its own title, any other as "Check soon" with its sentence);
    /// the dropdown shows blocks only when something is red or there are two or more notices.
    func testOneFilledButtonAndAmberNeverGetsOne() {
        for name in Fixture.Name.allCases {
            let i = Fixture.inputs(name)
            let r1 = NDR1.make(i, NDModel.make(i))
            let red = r1.items.filter { $0.severity == .red }
            XCTAssertEqual(r1.showsProblemRows, !red.isEmpty || r1.items.count > 1, name.rawValue)
            let filled = r1.items.compactMap(\.action).filter(NDButtonPolicy.filled)
            XCTAssertLessThanOrEqual(filled.count, 1, "\(name.rawValue): one filled button per page")
            XCTAssertTrue(filled.allSatisfy { [.reconnectDrive, .connectDrive].contains($0.kind) }, name.rawValue)
            XCTAssertTrue(r1.items.filter { $0.severity == .orange }.allSatisfy { $0.action == nil }, "\(name.rawValue): amber, no button")
            if red.isEmpty { XCTAssertFalse(r1.headline.contains("need"), "\(name.rawValue): amber is never \"needs you\"") }
        }
        let two = NDR1.make(Fixture.inputs(.twoFaults), NDModel.make(Fixture.inputs(.twoFaults)))
        XCTAssertEqual(two.headline, Say.thingsNeedYou(2))
        let late = NDR1.make(Fixture.inputs(.backupLate), NDModel.make(Fixture.inputs(.backupLate)))
        XCTAssertEqual(late.headline, Say.checkSoon, "L24")
        XCTAssertEqual(late.headlineDetail, late.items.first?.title)
        XCTAssertFalse(late.showsProblemRows)
        let behind = NDR1.make(Fixture.inputs(.meaningBehind), NDModel.make(Fixture.inputs(.meaningBehind)))
        XCTAssertEqual(behind.headline, Say.meaningCatchingUp)
        XCTAssertEqual(behind.headlineDetail, Say.meaningBehindDetail)
    }

    /// Lead, 09:55: the Activity chart carries no "own scale" note; each half's "up to" label says its size.
    func testActivityHasNoScaleNote() throws {
        let sources = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar")
        for file in ["BrainBarMenuSentences.swift", "NewDesign/NDWindow.swift", "NewDesign/NDMenu.swift"] {
            let text = try String(contentsOf: sources.appendingPathComponent(file), encoding: .utf8)
            XCTAssertFalse(text.contains("own scale") || text.contains("ownScale"), file)
        }
    }

    /// R1.2 change 12: "last known" is the last confirmed evaluation's time, the same in the window and the dropdown.
    func testLastKnownTimeIsTheEvaluationsInBothSurfaces() throws {
        let i = Fixture.inputs(.notConfirmed)
        let r1 = NDR1.make(i, NDModel.make(i))
        let at = try XCTUnwrap(i.capture.verdict?.evaluatedAt)
        XCTAssertEqual(r1.pipeline.lastKnownAt, NDR1.hhmm(at, i.capture), "the evaluation's time, never the render time")
        XCTAssertEqual(Say.valuesLastKnownAt(r1.pipeline.lastKnownAt!), "not confirmed: values last known at 06:00")
        XCTAssertEqual(Say.lastKnownAt(r1.pipeline.lastKnownAt!), "Last known at 06:00")
        XCTAssertEqual(NDMenuHeadline.detail(r1, lastVerdictHadRed: false), Say.nothingKnownBrokenShort)
        XCTAssertNil(NDMenuHeadline.detail(r1, lastVerdictHadRed: true), "only when the last confirmed verdict had no red check")
    }

    /// R1.2 change 22 and L25: the window is the page's real height in every fixture, capped at the screen minus 40 pt.
    func testWindowIsThePagesHeightCappedByTheScreen() {
        for name in Fixture.Name.allCases {
            for page in NDPage.allCases {
                let store = NDStore(fixture: Fixture.inputs(name), page: page)
                let needed = ceil(NDPageHeight.content(store, page: page) + NDPageHeight.chrome)
                XCTAssertEqual(NDPageHeight.window(store, page: page, screen: nil), needed, "\(name.rawValue) \(page)")
                XCTAssertEqual(NDPageHeight.window(store, page: page, screen: 600), min(needed, 560), "\(name.rawValue) \(page): L25")
            }
        }
    }

    /// R1.2 change 13: a failed upload retried the same night is one hollow red ring, said in the legend and its help tag.
    func testARetriedFailureIsAHollowRingWithItsWords() throws {
        let store = NDStore(fixture: Fixture.inputs(.backupsFailedUpload), page: .backups)
        let lanes = NDBackupsPage(store: store).lanes(store.model)
        let drive = try XCTUnwrap(lanes.first { $0.id == "backup_offsite" })
        XCTAssertTrue(drive.events.contains { $0.kind == .retried })
        XCTAssertFalse(drive.events.contains { $0.kind == .fail }, "no filled red dot when the retry succeeded")
        XCTAssertNotNil(drive.help)
        XCTAssertTrue(NDBackupsPage.legend(lanes).contains(Say.failedThenRetried))
        XCTAssertEqual(store.r1.headline, BrainBarMenuPresenter.Say.allGood.text)
    }

    /// § 3.1 hover: the pointer's bucket is the pitch that holds it; outside the plot there is none.
    func testActivityHoverPicksTheBucketUnderThePointer() {
        XCTAssertEqual(NDActivitySection.bucket(at: 0, count: 60), 0)
        XCTAssertEqual(NDActivitySection.bucket(at: 13.4, count: 60), 1)
        XCTAssertEqual(NDActivitySection.bucket(at: 767.9, count: 60), 59)
        XCTAssertNil(NDActivitySection.bucket(at: -1, count: 60))
        XCTAssertNil(NDActivitySection.bucket(at: 768, count: 60), "R1.2: a 768 pt plot inside its surface")
        XCTAssertNil(NDActivitySection.bucket(at: 10, count: 0))
    }
}
