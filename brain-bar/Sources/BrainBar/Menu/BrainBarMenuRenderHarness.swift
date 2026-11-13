#if DEBUG
import AppKit
import SwiftUI

/// `BRAINBAR_RENDER_ONLY=<dir>` renders the M menu off-screen, at 2x, for each of the seventeen contract cases in light
/// and dark (`BRAINBAR_RENDER_SET=menu` renders only these). An `NSMenu` cannot be drawn without being shown, so the
/// picture is assembled from the built menu itself: its hosted rows are the real views, its separators and standard
/// items are drawn at the row heights `NSMenu.size` reports, and the material is the mocks' flat colour.
@MainActor enum BrainBarMenuRenderHarness {
    struct Metrics { var row: CGFloat, separator: CGFloat, padding: CGFloat }

    /// The system's own menu metrics, measured from menus that are never shown.
    static func measured() -> Metrics {
        func size(_ items: [NSMenuItem]) -> CGFloat { let m = NSMenu(); items.forEach(m.addItem); return m.size.height }
        func item() -> NSMenuItem { NSMenuItem(title: "\u{2026}", action: nil, keyEquivalent: "") }
        let one = size([item()]), two = size([item(), item()]), sep = size([item(), .separator(), item()])
        return Metrics(row: two - one, separator: sep - two, padding: (one - (two - one)) / 2)
    }

    static func render(in directory: URL) throws -> [String] {
        let cases = try BrainBarMenuCases.load(BrainBarMenuCases.sourceURL())
        let metrics = measured()
        print("[brainbar-render] menu metrics: row \(metrics.row) separator \(metrics.separator) padding \(metrics.padding)")
        var written: [String] = []
        func shoot(_ id: String, _ suffix: String = "", accessibility: BrainBarMenuAccessibility = .init(),
                   appearance: NSAppearance.Name, optionHeld: Bool = false, focusFirstAction: Bool = false) throws {
            guard let c = cases.first(where: { $0["id"] as? String == id }), let inputs = c["inputs"] as? [String: Any] else {
                throw BrainBarMenuCases.Failure("no case \(id)")
            }
            let builder = BrainBarMenuBuilder(accessibility: accessibility)
            let menu = builder.build(BrainBarMenuPresenter.present(try BrainBarMenuCases.capture(inputs)))
            if focusFirstAction { builder.highlight(menu.items.first { $0.action != nil && $0.view != nil }, byKeyboard: true) }
            let dark = appearance == .darkAqua || appearance == .accessibilityHighContrastDarkAqua
            let name = "menu-\(id)\(suffix)-\(dark ? "dark" : "light")"
            try write(snapshot(menu, metrics: metrics, appearance: appearance, optionHeld: optionHeld), name: name, in: directory)
            written.append(name)
        }
        for c in cases {
            for appearance in [NSAppearance.Name.aqua, .darkAqua] { try shoot(c["id"] as? String ?? "", appearance: appearance) }
        }
        for (plain, contrast) in [(NSAppearance.Name.aqua, NSAppearance.Name.accessibilityHighContrastAqua),
                                  (.darkAqua, .accessibilityHighContrastDarkAqua)] {
            try shoot("S3", "-focus", appearance: plain, focusFirstAction: true)
            try shoot("S1", "-option", appearance: plain, optionHeld: true)
            try shoot("S10", "-reduce-motion", accessibility: .init(reduceMotion: true), appearance: plain)
            try shoot("S12", "-increase-contrast", accessibility: .init(increaseContrast: true), appearance: contrast)
        }
        return written
    }

    /// The menu as it would open, at rest: the Option alternates replace their primaries only when Option is held.
    /// Each row is drawn as its own root view (the only way `cacheDisplay` places a SwiftUI host right), then laid
    /// on the material at its offset.
    static func snapshot(_ menu: NSMenu, metrics: Metrics, appearance name: NSAppearance.Name, optionHeld: Bool) throws -> NSBitmapImageRep {
        let width = BrainBarMenuMetrics.width
        let appearance = NSAppearance(named: name)
        var rows: [(view: NSView, y: CGFloat)] = []
        var y = metrics.padding
        for (index, item) in menu.items.enumerated() where !item.isHidden {
            let replaced = index + 1 < menu.items.count && menu.items[index + 1].isAlternate
            if item.isAlternate != optionHeld && (item.isAlternate || replaced) { continue }
            let view = rowView(item, metrics: metrics)
            view.appearance = appearance
            view.layoutSubtreeIfNeeded()
            rows.append((view, y))
            y += view.frame.height
        }
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.3))
        let size = NSSize(width: width, height: y + metrics.padding)
        let rep = try bitmap(size)
        NSGraphicsContext.saveGraphicsState()
        defer { NSGraphicsContext.restoreGraphicsState() }
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        (appearance ?? NSAppearance.currentDrawing()).performAsCurrentDrawingAppearance {
            let dark = appearance?.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            let shape = NSBezierPath(roundedRect: NSRect(origin: .zero, size: size).insetBy(dx: 0.25, dy: 0.25), xRadius: 13, yRadius: 13)
            NSColor(srgbRed: dark ? 0.122 : 0.925, green: dark ? 0.122 : 0.925, blue: dark ? 0.129 : 0.933, alpha: 1).setFill()
            shape.fill()
            NSColor(white: dark ? 1 : 0, alpha: dark ? 0.14 : 0.16).setStroke()
            shape.lineWidth = 0.5
            shape.stroke()
        }
        for (view, top) in rows {
            let piece = try rowBitmap(view)
            piece.draw(in: NSRect(x: 0, y: size.height - top - view.bounds.height, width: width, height: view.bounds.height),
                       from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: false, hints: nil)
        }
        return rep
    }

    /// What the picture draws for one item: its hosted view, or a separator or standard row at the measured height.
    static func rowView(_ item: NSMenuItem, metrics: Metrics) -> NSView {
        let width = BrainBarMenuMetrics.width
        return item.isSeparatorItem ? SeparatorRow(frame: NSRect(x: 0, y: 0, width: width, height: metrics.separator))
            : item.view ?? StandardRow(item, frame: NSRect(x: 0, y: 0, width: width, height: metrics.row))
    }

    /// One row drawn alone, at 2x (the tests read its pixels and its words from this).
    static func rowBitmap(_ view: NSView) throws -> NSBitmapImageRep {
        let piece = try bitmap(view.bounds.size)
        view.cacheDisplay(in: view.bounds, to: piece)
        return piece
    }

    private static func bitmap(_ size: NSSize) throws -> NSBitmapImageRep {
        guard size.width > 0, size.height > 0, let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: Int(size.width * 2), pixelsHigh: Int(ceil(size.height * 2)), bitsPerSample: 8,
            samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ) else { throw BrainBarMenuCases.Failure("no bitmap for \(size)") }
        rep.size = size
        return rep
    }

    /// A 2x PNG, whatever the screen.
    static func write(_ rep: NSBitmapImageRep, name: String, in directory: URL) throws {
        guard let png = rep.representation(using: .png, properties: [:]), png.count > 5_000 else {
            throw BrainBarMenuCases.Failure("\(name): refusing a blank render")
        }
        let url = directory.appendingPathComponent("\(name).png")
        try png.write(to: url, options: .atomic)
        print("[brainbar-render] \(name) \(Int(rep.size.width))×\(Int(rep.size.height)) pt; wrote \(url.path)")
    }

    private final class SeparatorRow: NSView {
        override var isFlipped: Bool { true }
        override func draw(_ dirtyRect: NSRect) {
            NSColor.separatorColor.setFill()
            NSRect(x: BrainBarMenuMetrics.inset, y: floor(bounds.midY), width: bounds.width - 2 * BrainBarMenuMetrics.inset, height: 1).fill()
        }
    }

    private final class StandardRow: NSView {
        let item: NSMenuItem
        init(_ item: NSMenuItem, frame: NSRect) { self.item = item; super.init(frame: frame) }
        @available(*, unavailable) required init?(coder: NSCoder) { nil }
        override var isFlipped: Bool { true }
        override func draw(_ dirtyRect: NSRect) {
            let font = NSFont.menuFont(ofSize: 0)
            let title = NSAttributedString(string: item.title, attributes: [.font: font, .foregroundColor: NSColor.labelColor])
            let glyphs = [(NSEvent.ModifierFlags.control, "⌃"), (.option, "⌥"), (.shift, "⇧"), (.command, "⌘")]
                .filter { item.keyEquivalentModifierMask.contains($0.0) }.map(\.1).joined() + item.keyEquivalent.uppercased()
            let key = NSAttributedString(string: glyphs, attributes: [.font: font, .foregroundColor: NSColor.secondaryLabelColor])
            let y = (bounds.height - title.size().height) / 2
            title.draw(at: NSPoint(x: BrainBarMenuMetrics.inset, y: y))
            key.draw(at: NSPoint(x: bounds.width - BrainBarMenuMetrics.inset - key.size().width, y: y))
        }
    }
}
#endif
