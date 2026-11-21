import AppKit

/// The menu-bar brain in four states (L1, comp 01), on #1807's artwork: resting (template), one fault (red with "!"),
/// several (the same, with a tabular count 3 pt beside the glyph, never on it, in the template colour the menu bar gives
/// it at draw time: white on a dark bar, black on a light one; review fix 8), and no reading yet (the resting glyph,
/// dimmed to 55 % for contrast on both bars). No motion in the menu bar.
@MainActor enum NDIcon {
    /// D6 (L20 rewritten): the fault brain and its count are for red items only; an amber (late) state keeps the resting
    /// brain, and the icon agrees with the pipeline (a red station iff the fault icon).
    /// R1.2 change 10; L47 (replacing L23): one predicate for the dot.
    enum State: Equatable {
        case rest
        /// One or more red items (the count beside the brain from two).
        case red(Int)
        /// L47: any check amber (catching up included), and nothing red.
        case amber
        /// No reading yet, or nothing confirmed: the brain at 55 % with a "?" badge.
        case unconfirmed
    }

    static func state(_ model: NDModel, _ r1: NDR1) -> State {
        if r1.redCount > 0 { return .red(r1.redCount) }
        if noReadingYet(model) || r1.pipeline.allUnconfirmed { return .unconfirmed }
        if r1.amberDot { return .amber }
        return .rest
    }

    /// Resolve against the status button, whose appearance can differ from the app over light wallpaper.
    static func menuInk(_ appearance: NSAppearance) -> NSColor {
        appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            ? .white : NSColor(srgbRed: 28 / 255, green: 28 / 255, blue: 30 / 255, alpha: 1)
    }

    static func image(for model: NDModel, _ r1: NDR1, appearance: NSAppearance = .currentDrawing()) -> NSImage {
        let ink = menuInk(appearance)
        switch state(model, r1) {
        case .red: return image(for: marked(model, r1), ink: ink)
        case .rest: return image(for: marked(model, r1), ink: ink)
        case .amber: return badged(resting(model), ink: ink, alpha: 1, accessibility: "BrainBar — " + r1.headline) { rect in
            // A 6 pt amber dot at the top-right, with a 1 pt keyline cut out of the brain (the menu bar shows through).
            let dot = NSRect(x: rect.maxX - 7, y: rect.maxY - 7, width: 6, height: 6)
            NSGraphicsContext.current?.compositingOperation = .clear
            NSBezierPath(ovalIn: dot.insetBy(dx: -1, dy: -1)).fill()
            NSGraphicsContext.current?.compositingOperation = .sourceOver
            NDColor.orange.setFill()
            NSBezierPath(ovalIn: dot).fill()
        }
        case .unconfirmed: return badged(resting(model), ink: ink, alpha: 0.55, accessibility: "BrainBar — " + r1.headline) { rect in
            // A small "?" badge at the bottom-right in the template colour, on a cut-out circle.
            let badge = NSRect(x: rect.maxX - 9, y: rect.minY, width: 9, height: 9)
            NSGraphicsContext.current?.compositingOperation = .clear
            NSBezierPath(ovalIn: badge.insetBy(dx: -1, dy: -1)).fill()
            NSGraphicsContext.current?.compositingOperation = .sourceOver
            ink.setFill()
            NSBezierPath(ovalIn: badge).fill()
            let mark = NSAttributedString(string: "?", attributes: [.font: NSFont.systemFont(ofSize: 7, weight: .bold),
                                                                    .foregroundColor: ink == NSColor.white ? NSColor.black : NSColor.white])
            let size = mark.size()
            mark.draw(at: NSPoint(x: badge.midX - size.width / 2, y: badge.midY - size.height / 2))
        }
        }
    }

    private static func resting(_ model: NDModel) -> NSImage {
        var m = model
        m.icon = .resting
        m.count = 0
        return image(for: m)
    }

    /// The template brain drawn in the menu bar's text colour (resolved at draw time) with a badge over it.
    private static func badged(_ brain: NSImage, ink: NSColor, alpha: CGFloat, accessibility: String, _ badge: @escaping (NSRect) -> Void) -> NSImage {
        let image = NSImage(size: NSSize(width: 22, height: 22), flipped: false) { rect in
            let tinted = NSImage(size: rect.size, flipped: false) { r in
                brain.draw(in: r)
                ink.set()
                r.fill(using: .sourceAtop)
                return true
            }
            tinted.draw(in: rect, from: .zero, operation: .sourceOver, fraction: alpha)
            badge(rect)
            return true
        }
        image.isTemplate = false
        image.accessibilityDescription = accessibility
        return image
    }

    static func marked(_ model: NDModel, _ r1: NDR1) -> NDModel {
        var m = model
        let red = r1.redCount
        m.icon = red > 0 ? .marked : (model.icon == .marked ? .resting : model.icon)
        m.count = red
        return m
    }

    static func image(for model: NDModel, ink: NSColor = menuInk(.currentDrawing())) -> NSImage {
        let presentation = BrainBarMenuPresentation(form: model.icon == .marked ? .needs : .calm, headline: model.headline,
                                                    count: model.count, icon: model.icon, rows: [])
        let brain = BrainBarBrainIcon.image(for: presentation)
        guard model.icon == .marked, model.count > 1 else { return brain }
        let numeral = model.count > 99 ? "99+" : String(model.count)
        let font = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .semibold)
        let width = ceil(NSAttributedString(string: numeral, attributes: [.font: font]).size().width)
        let image = NSImage(size: NSSize(width: 22 + 3 + width, height: 22), flipped: false) { rect in
            brain.draw(in: NSRect(x: 0, y: 0, width: 22, height: 22))
            let text = NSAttributedString(string: numeral, attributes: [.font: font, .foregroundColor: ink])
            let size = text.size()
            text.draw(at: NSPoint(x: 25, y: (rect.height - size.height) / 2))
            return true
        }
        image.isTemplate = false
        image.accessibilityDescription = "BrainBar — " + model.headline
        return image
    }

    /// D3 (real screen, round 1; comp 01's "no reading yet" glyph): the resting brain at 55 % whenever nothing is
    /// confirmed, before the first verdict and whenever the checker is stale (every pipe "not confirmed"), so the icon
    /// is never calmer than the checks (L20). A fault brain stays a fault brain.
    static func dimmed(_ model: NDModel, _ r1: NDR1) -> Bool {
        noReadingYet(model) || (model.icon != .marked && r1.pipeline.allUnconfirmed)
    }

    /// "No reading yet": no verdict has been read since launch. It lasts seconds, before the first verdict.
    static func noReadingYet(_ model: NDModel) -> Bool {
        model.icon == .resting && model.presentation.form == .calm && model.inputsHadNoVerdict
    }
}

extension NDModel {
    var inputsHadNoVerdict: Bool { checked == nil }
}
