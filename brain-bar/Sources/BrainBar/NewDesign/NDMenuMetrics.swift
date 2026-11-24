#if DEBUG
import AppKit

/// System menu metrics shared by the offscreen New renderer.
@MainActor enum NDMenuMetrics {
    struct Metrics { var row: CGFloat, separator: CGFloat, padding: CGFloat }

    /// The system's own menu metrics, measured from menus that are never shown.
    static func measured() -> Metrics {
        func size(_ items: [NSMenuItem]) -> CGFloat { let m = NSMenu(); items.forEach(m.addItem); return m.size.height }
        func item() -> NSMenuItem { NSMenuItem(title: "\u{2026}", action: nil, keyEquivalent: "") }
        let one = size([item()]), two = size([item(), item()]), sep = size([item(), .separator(), item()])
        return Metrics(row: two - one, separator: sep - two, padding: (one - (two - one)) / 2)
    }

}
#endif
