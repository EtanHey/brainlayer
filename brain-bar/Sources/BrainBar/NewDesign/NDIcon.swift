import AppKit

/// The menu-bar brain in four states (L1, comp 01), on #1807's artwork: resting (template), one fault (red with "!"),
/// several (the same, with a tabular count 3 pt beside the glyph, never on it, in the template colour the menu bar gives
/// it at draw time: white on a dark bar, black on a light one; review fix 8), and no reading yet (the resting glyph,
/// dimmed by the caller to 45 %). No motion in the menu bar.
@MainActor enum NDIcon {
    static func image(for model: NDModel) -> NSImage {
        let presentation = BrainBarMenuPresentation(form: model.icon == .marked ? .needs : .calm, headline: model.headline,
                                                    count: model.count, icon: model.icon, rows: [])
        let brain = BrainBarBrainIcon.image(for: presentation)
        guard model.icon == .marked, model.count > 1 else { return brain }
        let numeral = model.count > 99 ? "99+" : String(model.count)
        let font = NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .bold)
        let width = ceil(NSAttributedString(string: numeral, attributes: [.font: font]).size().width)
        let image = NSImage(size: NSSize(width: 22 + 3 + width, height: 22), flipped: false) { rect in
            brain.draw(in: NSRect(x: 0, y: 0, width: 22, height: 22))
            let text = NSAttributedString(string: numeral, attributes: [.font: font, .foregroundColor: NSColor.textColor])
            let size = text.size()
            text.draw(at: NSPoint(x: 25, y: (rect.height - size.height) / 2))
            return true
        }
        image.isTemplate = false
        image.accessibilityDescription = "BrainBar — " + model.headline
        return image
    }

    /// "No reading yet": no verdict has been read since launch. It lasts seconds, before the first verdict.
    static func noReadingYet(_ model: NDModel) -> Bool {
        model.icon == .resting && model.presentation.form == .calm && model.inputsHadNoVerdict
    }
}

extension NDModel {
    var inputsHadNoVerdict: Bool { checked == nil }
}
