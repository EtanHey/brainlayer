#if DEBUG
import AppKit
import SwiftUI

/// Headless renders of the new design at 2x, light and dark, for comparing with the comps while building. A development
/// aid only: stage 2 judges real captures. Nothing here opens a window, a menu or a status item; the hosting views are
/// drawn with `cacheDisplay` off screen. The menu is assembled from its real hosted rows (an `NSMenu` cannot be drawn
/// without being shown), on a flat stand-in for the menu material.
@MainActor enum NDRenderHarness {
    struct Failure: LocalizedError {
        let errorDescription: String?
        init(_ m: String) { errorDescription = m }
    }

    static let appearances: [(String, NSAppearance.Name)] = [("dark", .darkAqua), ("light", .aqua)]
    static let environmentVariable = "BRAINBAR_ND_RENDER"

    /// `BRAINBAR_ND_RENDER=<absolute dir>`: render every surface and exit, before the app starts. No status item, window,
    /// database, socket, collector or timer exists yet, and a prohibited app cannot be activated.
    static func runIfRequested() {
        guard let path = ProcessInfo.processInfo.environment[environmentVariable] else { return }
        guard !path.isEmpty, NSString(string: path).isAbsolutePath else {
            print("[nd-render] \(environmentVariable) must name an absolute directory"); exit(64)
        }
        NSApplication.shared.setActivationPolicy(.prohibited)
        do {
            let written = try renderAll(to: URL(fileURLWithPath: path, isDirectory: true))
            print("[nd-render] wrote \(written.count) files to \(path)")
            exit(0)
        } catch {
            print("[nd-render] failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    @discardableResult
    static func renderAll(to directory: URL) throws -> [String] {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        NDMarkView.captureClock = Date(timeIntervalSinceReferenceDate: 0.3)
        defer { NDMarkView.captureClock = nil }
        var written: [String] = []
        for (suffix, appearance) in appearances {
            for name in BrainBarDesignFixture.Name.allCases {
                let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: name.page)
                written.append(try window(store, name: "window-\(name.rawValue)-\(store.page.rawValue)-\(suffix)", appearance: appearance, in: directory))
                written.append(try menu(store, name: "menu-\(name.rawValue)-\(suffix)", appearance: appearance, in: directory))
            }
            for page in [NDPage.backups, .jobs] {
                for name in [BrainBarDesignFixture.Name.calm, .driveSignedOut, .nightlyFailing, .twoFaults, .notConfirmed] {
                    let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: page)
                    written.append(try window(store, name: "window-\(name.rawValue)-\(page.rawValue)-\(suffix)", appearance: appearance, in: directory))
                }
            }
            // R1.10.1: drive-signed-out right after a press on the row's Reconnect: every surface waits.
            let pressed = NDStore(fixture: BrainBarDesignFixture.inputs(.driveSignedOut), page: .overview)
            pressed.perform(.reconnectDrive)
            written.append(try window(pressed, name: "window-drive-signed-out-after-reconnect-overview-\(suffix)", appearance: appearance,
                                      in: directory))
            written.append(try hover(pressed, station: .drive, name: "hover-drive-drive-signed-out-after-reconnect-\(suffix)",
                                     appearance: appearance, in: directory))
            written.append(try settings(name: "settings-drive-signed-out-after-reconnect-\(suffix)", store: pressed, appearance: appearance,
                                        in: directory))
            // R1.10: Settings calm (no action) and signed out (Reconnect, as in Drive's panel).
            written.append(try settings(name: "settings-calm-\(suffix)", fixture: .calm, appearance: appearance, in: directory))
            written.append(try settings(name: "settings-drive-signed-out-\(suffix)", fixture: .driveSignedOut, appearance: appearance,
                                        in: directory))
            written.append(try icons(name: "icons-\(suffix)", appearance: appearance, in: directory))
            // R1.5 comp 51: the station panels, the Copy report click state.
            written.append(try panelsSheet(name: "panels-\(suffix)", appearance: appearance, in: directory))
            // R1.4 comp 45: where the words went, each in the hover that now carries it.
            written.append(try wordsSheet(name: "hovers-words-\(suffix)", appearance: appearance, in: directory))
            // R1.3 comp 37: the three hover states (Memory's explanation; the Jobs and Drive faults with their actions).
            for (fixture, station) in [(BrainBarDesignFixture.Name.calm, NDStation.memory), (.twoFaults, .jobs), (.twoFaults, .drive),
                                       (.agentNotRead, .agents), (.calm, .search), (.reconnectWaiting, .drive), (.calm, .jobs),
                                       (.calm, .weekly), (.calm, .mac), (.nightlyFailing, .jobs), (.nightlyFailing, .weekly),
                                       (.savesStuck, .jobs), (.savesStuck, .weekly)] {
                let store = NDStore(fixture: BrainBarDesignFixture.inputs(fixture), page: .overview)
                written.append(try hover(store, station: station, name: "hover-\(station.rawValue)-\(fixture.rawValue)-\(suffix)",
                                         appearance: appearance, in: directory))
            }
            // R1.6 item 9: the Activity readout as the window lays it out, at the first, a middle and the last column.
            let calm = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
            for (label, column) in [("first", 0), ("middle", 30), ("last", 59)] {
                written.append(try activity(calm, hover: column, name: "activity-hover-\(label)-calm-\(suffix)", appearance: appearance,
                                            in: directory))
            }
            // R1.6 item 6: not confirmed on 3 h: the first hour dimmed, then the hatch.
            let stale = NDStore(fixture: BrainBarDesignFixture.inputs(.notConfirmed), page: .overview)
            stale.range = .threeHours
            written.append(try activity(stale, hover: nil, name: "activity-not-confirmed-3h-\(suffix)", appearance: appearance, in: directory))
            // R1.6 item 3: the frames of a page switch, as the controller orders them.
            for (from, to) in [(NDPage.overview, NDPage.jobs), (.jobs, .overview)] {
                written.append(try pageSwitch(from: from, to: to, name: "switch-\(from.rawValue)-to-\(to.rawValue)-\(suffix)",
                                              appearance: appearance, in: directory))
            }
        }
        return written
    }

    static func window(_ store: NDStore, name: String, appearance: NSAppearance.Name, size: NSSize? = nil,
                       in directory: URL) throws -> String {
        // The window's fitted height (`NDWindowSizing`: the tallest page's, or L17's per-page height).
        let size = size ?? NSSize(width: NDGrid.windowWidth, height: NDPageHeight.fitted(store, page: store.page, screen: nil))
        // `cacheDisplay` cannot capture Liquid Glass, so the tab pill is drawn as its solid stand-in.
        let view = NDWindowRoot(store: store, drawsTrafficLights: true).environment(\.glassTabSolidPill, true)
            .frame(width: size.width, height: size.height)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    /// The Pipeline surface with one station hovered and its popover drawn where AppKit opens it (popovers are their own
    /// windows, so the harness draws the content at the layout's frame instead).
    static func hover(_ store: NDStore, station: NDStation, name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        let r1 = store.r1
        let chart = NDPipelineChart(state: r1.pipeline, memories: store.model.memories, items: r1.stationItems, interactive: false,
                                    popover: { s in AnyView(NDStationPopover(store: store, station: s)) }, previewHover: station,
                                    backupProgress: store.inputs.backupProgress)
        // 260 pt of room on the left: the agents' panel opens past the window's left edge (L39).
        let margin: CGFloat = 260
        let size = NSSize(width: NDGrid.windowWidth + margin, height: 660)
        let view = ZStack(alignment: .topLeading) {
            ND.bg
            NDCard(padding: EdgeInsets(top: 16, leading: 16, bottom: 12, trailing: 16)) { chart }
                .frame(width: NDGrid.windowWidth - 2 * NDGrid.pageSide)
                .offset(x: margin + NDGrid.pageSide, y: 150)
        }
        .frame(width: size.width, height: size.height)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    /// The Activity block on the page's background, one column hovered (R1.6 item 9).
    static func activity(_ store: NDStore, hover: Int?, name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        let size = NSSize(width: NDGrid.windowWidth, height: 300)
        let view = ZStack(alignment: .topLeading) {
            ND.bg
            NDActivitySection(store: store)
                .environment(\.ndPreviewHover, hover)
                // `cacheDisplay` cannot capture Liquid Glass: the range switch's pill is drawn as its solid stand-in.
                .environment(\.glassTabSolidPill, true)
                .frame(width: NDGrid.windowWidth - 2 * NDGrid.pageSide)
                .offset(x: NDGrid.pageSide, y: 60)
        }
        .frame(width: size.width, height: size.height)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    /// R1.6 item 3 / R1.8 item 1 (per-page heights, `NDWindowSizing.perPageHeights`): five frames of a page switch side
    /// by side (0, 50, 100, 150, 200 ms): the window's height on the controller's curve, its top edge fixed, and the page
    /// it draws (the new one from the first frame). With one fixed height, `pageSlide` instead.
    static func pageSwitch(from: NDPage, to: NDPage, name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        guard NDWindowSizing.perPageHeights else { return try pageSlide(from: from, to: to, name: name, appearance: appearance, in: directory) }
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: from)
        let start = NDPageHeight.window(store, page: from, screen: nil)
        let end = NDPageHeight.window(store, page: to, screen: nil)
        let times: [Double] = [0, 0.25, 0.5, 0.75, 1]
        let gap: CGFloat = 24
        let tall = max(start, end)
        let size = NSSize(width: CGFloat(times.count) * (NDGrid.windowWidth + gap) + gap, height: tall + 2 * gap)
        let frames: [AnyView] = times.map { t in
            let height = start + (end - start) * easeOut(t)
            // R1.8 item 1: the new page from the first frame, in both directions.
            let page = to
            let frameStore = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: page)
            return AnyView(NDWindowRoot(store: frameStore, drawsTrafficLights: true)
                .frame(width: NDGrid.windowWidth, height: NDPageHeight.window(frameStore, page: page, screen: nil), alignment: .top)
                .frame(width: NDGrid.windowWidth, height: height, alignment: .top)
                .background(ND.bg)
                .clipped()
                .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.gray, lineWidth: 1)))
        }
        let view = ZStack(alignment: .topLeading) {
            Color.gray.opacity(0.35)
            HStack(alignment: .top, spacing: gap) { ForEach(0..<frames.count, id: \.self) { frames[$0] } }
                .padding(gap)
        }
        .frame(width: size.width, height: size.height)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    /// The default (one fixed height): five frames of a click's slide side by side (0, 50, 100, 150, 200 ms), the pager's
    /// own spring sampled at 120 Hz, at the window's fitted height; the window never resizes.
    static func pageSlide(from: NDPage, to: NDPage, name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        let pager = GlassTabPager(count: NDPage.allCases.count, selection: NDPage.allCases.firstIndex(of: from) ?? 0)
        pager.isLive = { true }
        pager.select(NDPage.allCases.firstIndex(of: to) ?? 0, at: 0)
        var samples: [Double] = [pager.progress]
        for frame in 1...24 {
            pager.tick(at: Double(frame) / 120)
            if frame % 6 == 0 { samples.append(pager.progress) }
        }
        let gap: CGFloat = 24
        let height = NDPageHeight.fitted(NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: from), page: from, screen: nil)
        let size = NSSize(width: CGFloat(samples.count) * (NDGrid.windowWidth + gap) + gap, height: height + 2 * gap)
        let frames: [AnyView] = samples.map { progress in
            let frameStore = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: from)
            frameStore.tabs.present(progress)
            return AnyView(NDWindowRoot(store: frameStore, drawsTrafficLights: true).environment(\.glassTabSolidPill, true)
                .frame(width: NDGrid.windowWidth, height: height)
                .background(ND.bg)
                .clipped()
                .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.gray, lineWidth: 1)))
        }
        let view = ZStack(alignment: .topLeading) {
            Color.gray.opacity(0.35)
            HStack(alignment: .top, spacing: gap) { ForEach(0..<frames.count, id: \.self) { frames[$0] } }
                .padding(gap)
        }
        .frame(width: size.width, height: size.height)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    /// The page switch's curve, cubic-bezier(0.23, 1, 0.32, 1), at time t.
    static func easeOut(_ t: Double) -> Double {
        let c = NDPageSwitch.curve
        let (x1, y1, x2, y2) = (Double(c.0), Double(c.1), Double(c.2), Double(c.3))
        func x(_ u: Double) -> Double { 3 * (1 - u) * (1 - u) * u * x1 + 3 * (1 - u) * u * u * x2 + u * u * u }
        func y(_ u: Double) -> Double { 3 * (1 - u) * (1 - u) * u * y1 + 3 * (1 - u) * u * u * y2 + u * u * u }
        var lo = 0.0, hi = 1.0
        for _ in 0..<40 { let mid = (lo + hi) / 2; if x(mid) < t { lo = mid } else { hi = mid } }
        return y((lo + hi) / 2)
    }

    /// Comp 51's sheet: Drive with Reconnect; a failing job at rest and "Copied"; a failing agent; Copy on this Mac; the
    /// jobs list; each as its panel card draws it (288 wide, radius 8).
    static func panelsSheet(name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        func panel(_ fixture: BrainBarDesignFixture.Name, _ station: NDStation, copied: Bool = false) -> AnyView {
            let store = NDStore(fixture: BrainBarDesignFixture.inputs(fixture), page: .overview)
            return AnyView(NDStationPopover(store: store, station: station)
                .environment(\.ndPreviewCopied, copied)
                .frame(width: 288)
                .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(ND.pop)
                    .shadow(color: .black.opacity(0.18), radius: 12, y: 8).shadow(color: .black.opacity(0.10), radius: 1.5, y: 1))
                .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(ND.hair2, lineWidth: 0.5)))
        }
        let cards: [AnyView] = [
            panel(.twoFaults, .drive), panel(.twoFaults, .jobs), panel(.twoFaults, .jobs, copied: true),
            panel(.calm, .mac), panel(.calm, .jobs), panel(.calm, .weekly),  // R1.10 comp 59
        ]
        let size = NSSize(width: 980, height: 620)
        let view = ZStack(alignment: .topLeading) {
            ND.bg
            VStack(alignment: .leading, spacing: 28) {
                ForEach(0..<2, id: \.self) { row in
                    HStack(alignment: .top, spacing: 28) {
                        ForEach(0..<3, id: \.self) { col in cards[row * 3 + col] }
                    }
                }
            }
            .padding(28)
        }
        .frame(width: size.width, height: size.height)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    /// Comp 45's sheet: the Cursor icon's name, the rail, Save queue, Search, the return line, the agents' group, Kept safe
    /// and the Activity readout, each as its hover draws it.
    static func wordsSheet(name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm), page: .overview)
        func card<V: View>(_ v: V) -> some View {
            v.background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(ND.raised))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(ND.hair2, lineWidth: 0.5))
        }
        let wave = store.inputs.heroWaves[.hour]
        let peak = wave.map { w in w.saves.indices.max { w.saves[$0] < w.saves[$1] } ?? 0 } ?? 0
        let cards: [AnyView] = [
            AnyView(card(Text(NDProvider.name(.cursor)).font(NDType.bodyStrong).foregroundStyle(ND.t1).padding(12))),
            AnyView(card(NDHoverRegion.card(title: Say.railTitle, line: Say.railLine))),
            AnyView(card(NDStationPopover(store: store, station: .queue))),
            AnyView(card(NDStationPopover(store: store, station: .search))),
            AnyView(card(NDHoverRegion.card(title: Say.answersTitle, line: Say.answersLine))),
            AnyView(card(NDStationPopover(store: store, station: .agents))),
            AnyView(card(NDHoverRegion.card(title: Say.keptSafe, line: Say.keptSafeLine))),
            AnyView(wave.map { NDActivityReadout(wave: $0, index: peak, minutes: 60, now: store.inputs.capture.now, zone: store.inputs.capture.timeZone) }),
        ]
        let size = NSSize(width: 900, height: 760)
        let view = ZStack(alignment: .topLeading) {
            ND.bg
            VStack(alignment: .leading, spacing: 20) {
                ForEach(0..<3, id: \.self) { row in
                    HStack(alignment: .top, spacing: 20) {
                        ForEach(0..<3, id: \.self) { col in
                            if row * 3 + col < cards.count { cards[row * 3 + col] }
                        }
                    }
                }
            }
            .padding(24)
        }
        .frame(width: size.width, height: size.height)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    static func settings(name: String, fixture: BrainBarDesignFixture.Name, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        try settings(name: name, store: NDStore(fixture: BrainBarDesignFixture.inputs(fixture)), appearance: appearance, in: directory)
    }

    static func settings(name: String, store: NDStore, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        store.refreshSettingsFileSummary()
        // Polish 3: the view runs under the transparent 28 pt title bar and draws its hairline; the render adds only the
        // title the system draws there.
        let view = NDSettingsView(store: store)
            .overlay(alignment: .top) {
                Text(BrainBarMenuPresenter.SurfaceSay.settingsTitle).font(NDType.section).foregroundStyle(ND.t2)
                    .frame(height: NDSettingsWindowController.titleBar)
            }
        let height = NDSettingsWindowController.contentHeight
        let size = NSSize(width: NDSettingsWindowController.width, height: height)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    /// The brain in its four states, real size and 4x, on a strip of the menu bar's colour.
    static func icons(name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        // R1.3 change 12 with L47: rest, one red, two red, amber (a late copy), catching up (amber too), not confirmed.
        let images: [NSImage] = [BrainBarDesignFixture.Name.calm, .driveSignedOut, .twoFaults, .backupLate, .meaningBehind, .notConfirmed].map { name in
            let inputs = BrainBarDesignFixture.inputs(name)
            let model = NDModel.make(inputs)
            return NDIcon.image(for: model, NDR1.make(inputs, model))
        }
        let dark = appearance == .darkAqua
        let size = NSSize(width: 6 * 120, height: 140)
        let rep = try bitmap(size)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        (NSAppearance(named: appearance) ?? .currentDrawing()).performAsCurrentDrawingAppearance {
            NDColor.rgb(dark ? 0x1B2440 : 0xD5DEEF).setFill()
            NSRect(origin: .zero, size: size).fill()
            for (index, image) in images.enumerated() {
                let tint: NSColor = dark ? .white : .black
                for (scale, y) in [(CGFloat(1), CGFloat(110)), (CGFloat(4), CGFloat(10))] {
                    let w = image.size.width * scale, h = image.size.height * scale
                    let rect = NSRect(x: CGFloat(index) * 120 + (120 - w) / 2, y: y, width: w, height: h)
                    if image.isTemplate {
                        let tinted = NSImage(size: image.size, flipped: false) { r in
                            image.draw(in: r); tint.set(); r.fill(using: .sourceAtop); return true
                        }
                        tinted.draw(in: rect, from: .zero, operation: .sourceOver, fraction: 1)
                    } else {
                        image.draw(in: rect, from: .zero, operation: .sourceOver, fraction: 1)
                    }
                }
            }
        }
        NSGraphicsContext.restoreGraphicsState()
        try write(rep, name: name, in: directory)
        return name
    }

    /// The dropdown as it opens: each hosted row drawn alone at the menu's width, standard rows drawn at system metrics.
    static func menu(_ store: NDStore, name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        let builder = NDMenuBuilder(store: store, perform: { _ in }, openWindow: { _ in })
        let menu = builder.build()
        let metrics = NDMenuMetrics.measured()
        let width = NDGrid.menuWidth
        let nsAppearance = NSAppearance(named: appearance)
        var rows: [(NSView, CGFloat)] = []
        var y: CGFloat = 5
        for item in menu.items where !item.isHidden && !item.isAlternate {
            let view: NSView = item.isSeparatorItem ? Separator(frame: NSRect(x: 0, y: 0, width: width, height: metrics.separator))
                : item.view ?? Standard(item, frame: NSRect(x: 0, y: 0, width: width, height: 22))
            view.appearance = nsAppearance
            view.layoutSubtreeIfNeeded()
            rows.append((view, y))
            y += view.frame.height
        }
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.3))
        let size = NSSize(width: width, height: y + 5)
        let rep = try bitmap(size)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        (nsAppearance ?? .currentDrawing()).performAsCurrentDrawingAppearance {
            let dark = appearance == .darkAqua
            let shape = NSBezierPath(roundedRect: NSRect(origin: .zero, size: size).insetBy(dx: 0.25, dy: 0.25), xRadius: 10, yRadius: 10)
            NDColor.rgb(dark ? 0x26272E : 0xECEDF0).setFill()
            shape.fill()
            NSColor(white: dark ? 1 : 0, alpha: 0.12).setStroke()
            shape.lineWidth = 0.5
            shape.stroke()
        }
        NSGraphicsContext.restoreGraphicsState()
        for (view, top) in rows {
            let piece = try bitmap(view.bounds.size)
            view.cacheDisplay(in: view.bounds, to: piece)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
            piece.draw(in: NSRect(x: 0, y: size.height - top - view.bounds.height, width: width, height: view.bounds.height),
                       from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: false, hints: nil)
            NSGraphicsContext.restoreGraphicsState()
        }
        try write(rep, name: name, in: directory)
        return name
    }

    static func host<V: View>(_ view: V, size: NSSize, appearance: NSAppearance.Name) throws -> NSBitmapImageRep {
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(origin: .zero, size: size)
        host.appearance = NSAppearance(named: appearance)
        host.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.4))
        host.layoutSubtreeIfNeeded()
        let rep = try bitmap(size)
        host.cacheDisplay(in: host.bounds, to: rep)
        return rep
    }

    static func bitmap(_ size: NSSize) throws -> NSBitmapImageRep {
        guard size.width > 0, size.height > 0, let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: Int(size.width * 2), pixelsHigh: Int(ceil(size.height * 2)), bitsPerSample: 8,
            samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ) else { throw Failure("no bitmap for \(size)") }
        rep.size = size
        return rep
    }

    static func write(_ rep: NSBitmapImageRep, name: String, in directory: URL) throws {
        guard let png = rep.representation(using: .png, properties: [:]), png.count > 2_000 else { throw Failure("\(name): blank render") }
        try png.write(to: directory.appendingPathComponent(name + ".png"), options: .atomic)
    }

    private final class Separator: NSView {
        override var isFlipped: Bool { true }
        override func draw(_ dirtyRect: NSRect) {
            NSColor.separatorColor.setFill()
            NSRect(x: NDGrid.menuInset, y: floor(bounds.midY), width: bounds.width - 2 * NDGrid.menuInset, height: 1).fill()
        }
    }

    private final class Standard: NSView {
        let item: NSMenuItem
        init(_ item: NSMenuItem, frame: NSRect) { self.item = item; super.init(frame: frame) }
        @available(*, unavailable) required init?(coder: NSCoder) { nil }
        override var isFlipped: Bool { true }
        override func draw(_ dirtyRect: NSRect) {
            let font = NSFont.menuFont(ofSize: 0)
            let title = NSAttributedString(string: item.title, attributes: [.font: font, .foregroundColor: NSColor.labelColor])
            let key = NSAttributedString(string: item.keyEquivalent.isEmpty ? "" : "⌘" + item.keyEquivalent.uppercased(),
                                         attributes: [.font: font, .foregroundColor: NSColor.secondaryLabelColor])
            let y = (bounds.height - title.size().height) / 2
            title.draw(at: NSPoint(x: NDGrid.menuInset, y: y))
            key.draw(at: NSPoint(x: bounds.width - NDGrid.menuInset - key.size().width, y: y))
        }
    }
}
#endif
