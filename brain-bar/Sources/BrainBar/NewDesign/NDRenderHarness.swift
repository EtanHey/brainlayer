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
                for name in [BrainBarDesignFixture.Name.calm, .severalFaults] {
                    let store = NDStore(fixture: BrainBarDesignFixture.inputs(name), page: page)
                    written.append(try window(store, name: "window-\(name.rawValue)-\(page.rawValue)-\(suffix)", appearance: appearance, in: directory))
                }
            }
            written.append(try settings(name: "settings-\(suffix)", appearance: appearance, in: directory))
            written.append(try icons(name: "icons-\(suffix)", appearance: appearance, in: directory))
        }
        return written
    }

    static func window(_ store: NDStore, name: String, appearance: NSAppearance.Name, size: NSSize = NDGrid.windowSize,
                       in directory: URL) throws -> String {
        let view = NDWindowRoot(store: store, drawsTrafficLights: true).frame(width: size.width, height: size.height)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    static func settings(name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        let view = VStack(spacing: 0) {
            ZStack {
                Text("BrainBar Settings").font(.system(size: 13, weight: .semibold)).foregroundStyle(ND.ink2)
            }
            .frame(maxWidth: .infinity).frame(height: 28).background(ND.deck)
            .overlay(alignment: .bottom) { ND.rule.frame(height: 1) }
            NDSettingsView(store: store)
        }
        .frame(width: NDSettingsWindowController.width)
        let size = NSSize(width: NDSettingsWindowController.width, height: 28 + 14 + 18 + 3 * 40 + 2)
        try write(host(view, size: size, appearance: appearance), name: name, in: directory)
        return name
    }

    /// The brain in its four states, real size and 4x, on a strip of the menu bar's colour.
    static func icons(name: String, appearance: NSAppearance.Name, in directory: URL) throws -> String {
        let models = [BrainBarDesignFixture.Name.calm, .oneFault, .severalFaults].map { NDModel.make(BrainBarDesignFixture.inputs($0)) }
        var unread = BrainBarDesignFixture.inputs(.calm)
        unread.capture.verdict = nil
        unread.capture.isChecking = true
        let all = models + [NDModel.make(unread)]
        let dark = appearance == .darkAqua
        let size = NSSize(width: 4 * 120, height: 140)
        let rep = try bitmap(size)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        (NSAppearance(named: appearance) ?? .currentDrawing()).performAsCurrentDrawingAppearance {
            NDColor.rgb(dark ? 0x1B2440 : 0xD5DEEF).setFill()
            NSRect(origin: .zero, size: size).fill()
            for (index, model) in all.enumerated() {
                let image = NDIcon.image(for: model)
                let tint: NSColor = dark ? .white : .black
                let alpha: CGFloat = NDIcon.noReadingYet(model) ? 0.45 : 1
                for (scale, y) in [(CGFloat(1), CGFloat(110)), (CGFloat(4), CGFloat(10))] {
                    let w = image.size.width * scale, h = image.size.height * scale
                    let rect = NSRect(x: CGFloat(index) * 120 + (120 - w) / 2, y: y, width: w, height: h)
                    if image.isTemplate {
                        let tinted = NSImage(size: image.size, flipped: false) { r in
                            image.draw(in: r); tint.set(); r.fill(using: .sourceAtop); return true
                        }
                        tinted.draw(in: rect, from: .zero, operation: .sourceOver, fraction: alpha)
                    } else {
                        image.draw(in: rect, from: .zero, operation: .sourceOver, fraction: alpha)
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
        let metrics = BrainBarMenuRenderHarness.measured()
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
            let key = NSAttributedString(string: "⌘" + item.keyEquivalent.uppercased(),
                                         attributes: [.font: font, .foregroundColor: NSColor.secondaryLabelColor])
            let y = (bounds.height - title.size().height) / 2
            title.draw(at: NSPoint(x: NDGrid.menuInset, y: y))
            key.draw(at: NSPoint(x: bounds.width - NDGrid.menuInset - key.size().width, y: y))
        }
    }
}
#endif
