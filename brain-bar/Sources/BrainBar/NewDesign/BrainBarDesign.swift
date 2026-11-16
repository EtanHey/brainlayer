import AppKit
import SwiftUI

/// L6: one preference decides the whole UI, read live by the status item, the dropdown, the window and Settings.
/// Absent means `classic` until the owner approves the new design; flipping the default is a one-line change here.
enum BrainBarDesign: String, CaseIterable, Sendable {
    case classic, new

    static let key = "BrainBarDesign"
    static let defaultValue: BrainBarDesign = .classic

    /// The design to draw now. A DEBUG capture fixture forces `new`; nothing else does. Reading never writes a value.
    static func current(in defaults: UserDefaults = .standard) -> BrainBarDesign {
#if DEBUG
        if BrainBarDesignFixture.requested != nil { return .new }
#endif
        return defaults.string(forKey: key).flatMap(Self.init(rawValue:)) ?? defaultValue
    }

    var title: String {
        switch self {
        case .classic: "Classic"
        case .new: "New"
        }
    }
}

// MARK: - Tokens (DESIGN-SPEC § 2)

/// The new design's palette, type and grid. Colours are dynamic: they resolve against the view's own appearance, and
/// the structure colours (Rule, Track) rise to 3:1 under Increase Contrast (SPEC-REVIEW, carry note).
enum NDColor {
    static func rgb(_ hex: UInt32, alpha: CGFloat = 1) -> NSColor {
        NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255, green: CGFloat((hex >> 8) & 0xFF) / 255,
                blue: CGFloat(hex & 0xFF) / 255, alpha: alpha)
    }

    static func isDark(_ appearance: NSAppearance) -> Bool {
        appearance.bestMatch(from: [.aqua, .darkAqua, .accessibilityHighContrastAqua, .accessibilityHighContrastDarkAqua])
            .map { $0 == .darkAqua || $0 == .accessibilityHighContrastDarkAqua } ?? false
    }

    static func isHighContrast(_ appearance: NSAppearance) -> Bool {
        appearance.bestMatch(from: [.accessibilityHighContrastAqua, .accessibilityHighContrastDarkAqua, .aqua, .darkAqua])
            .map { $0 == .accessibilityHighContrastAqua || $0 == .accessibilityHighContrastDarkAqua } ?? false
    }

    static func dynamic(_ name: String, dark: NSColor, light: NSColor, contrastDark: NSColor? = nil,
                        contrastLight: NSColor? = nil) -> NSColor {
        NSColor(name: NSColor.Name("brainbar.nd." + name)) { appearance in
            let dark0 = isDark(appearance), contrast = isHighContrast(appearance)
            if contrast, let c = dark0 ? contrastDark : contrastLight { return c }
            return dark0 ? dark : light
        }
    }

    // R1 tokens (DESIGN-SPEC-R1 § 1.1): a neutral cool platform grey; colour only for meaning.
    private static func white(_ a: CGFloat) -> NSColor { NSColor(white: 1, alpha: a) }
    private static func black(_ a: CGFloat) -> NSColor { NSColor(white: 0, alpha: a) }
    static let bg = dynamic("bg", dark: rgb(0x1B1B1E), light: rgb(0xF6F6F7))
    static let raised = dynamic("raised", dark: rgb(0x242428), light: rgb(0xFFFFFF))
    static let pop = dynamic("pop", dark: rgb(0x2C2C31), light: rgb(0xFFFFFF))
    static let bar = dynamic("bar", dark: rgb(0x212124), light: rgb(0xEFEFF1))
    static let node = dynamic("node", dark: rgb(0x2A2A2F), light: rgb(0xFFFFFF))
    static let t1 = dynamic("t1", dark: rgb(0xEDEDF0), light: rgb(0x1D1D1F))
    static let t2 = dynamic("t2", dark: rgb(0xA1A1A8), light: rgb(0x55555D))
    static let t3 = dynamic("t3", dark: rgb(0x8A8A92), light: rgb(0x6E6E76))
    static let hair = dynamic("hair", dark: white(0.08), light: black(0.08))
    static let hair2 = dynamic("hair2", dark: white(0.14), light: black(0.14))
    static let pipe = dynamic("pipe", dark: rgb(0x6E6E77), light: rgb(0x8C8C94))
    static let quiet = dynamic("quiet", dark: rgb(0x45454C), light: rgb(0xCACACF))
    static let accent = dynamic("accent", dark: rgb(0x2E6FE8), light: rgb(0x1F5FD6))
    static let accentHover = dynamic("accent.h", dark: rgb(0x4A86F5), light: rgb(0x3574E6))
    static let accentPressed = dynamic("accent.p", dark: rgb(0x2462D6), light: rgb(0x1A53BD))
    static let focus = dynamic("focus", dark: rgb(0x4D8DFF), light: rgb(0x1F5FD6))
    static let saved = dynamic("saved", dark: rgb(0xA995FF), light: rgb(0x7356DA))
    static let savedText = dynamic("saved.t", dark: rgb(0xB6A6FF), light: rgb(0x6446CC))
    static let read = dynamic("read", dark: rgb(0x6B7385), light: rgb(0x80889B))
    static let readText = dynamic("read.t", dark: rgb(0x9AA1B3), light: rgb(0x5F677A))
    static let red = dynamic("red", dark: rgb(0xFF453A), light: rgb(0xFF3B30))
    static let redText = dynamic("red.t", dark: rgb(0xFF6961), light: rgb(0xD70015))
    static let redTint = dynamic("red.tint", dark: rgb(0xFF453A, alpha: 0.16), light: rgb(0xFF3B30, alpha: 0.12))
    static let orange = dynamic("orange", dark: rgb(0xFF9F0A), light: rgb(0xC97800))
    static let orangeText = dynamic("orange.t", dark: rgb(0xFFB340), light: rgb(0xA85700))
    static let control = dynamic("ctl", dark: white(0.07), light: black(0.06))
    static let controlSelected = dynamic("ctl.sel", dark: white(0.17), light: rgb(0xFFFFFF))
    static let button = dynamic("btn", dark: white(0.10), light: rgb(0xFFFFFF))
    static let buttonHover = dynamic("btn.h", dark: white(0.17), light: rgb(0xF0F0F2))
    static let buttonPressed = dynamic("btn.p", dark: white(0.24), light: rgb(0xE4E4E8))
    /// R1.1 change 19: graphite switches.
    static let switchOn = dynamic("sw.on", dark: rgb(0xA1A1A8), light: rgb(0x6E6E76))

    // The names the earlier surfaces used, kept on the R1 values (one palette).
    static let night = bg
    static let deck = raised
    static let ink = t1
    static let ink2 = t2
    static let lumen = accent
    static let lumenGraphic = saved
    static let tide = read
    static let rule = hair
    static let track = hair2
    static let fault = red
    static let warn = orange
    static let onLumen = NSColor.white
    static let chip = controlSelected
    static let menuSaves = saved
    static let menuReads = read
    static let menuTrack = hair2
}

/// SwiftUI handles of the same colours.
enum ND {
    static let bg = Color(nsColor: NDColor.bg)
    static let raised = Color(nsColor: NDColor.raised)
    static let pop = Color(nsColor: NDColor.pop)
    static let bar = Color(nsColor: NDColor.bar)
    static let node = Color(nsColor: NDColor.node)
    static let t1 = Color(nsColor: NDColor.t1)
    static let t2 = Color(nsColor: NDColor.t2)
    static let t3 = Color(nsColor: NDColor.t3)
    static let hair = Color(nsColor: NDColor.hair)
    static let hair2 = Color(nsColor: NDColor.hair2)
    static let pipe = Color(nsColor: NDColor.pipe)
    static let quiet = Color(nsColor: NDColor.quiet)
    static let accent = Color(nsColor: NDColor.accent)
    static let accentHover = Color(nsColor: NDColor.accentHover)
    static let accentPressed = Color(nsColor: NDColor.accentPressed)
    static let focus = Color(nsColor: NDColor.focus)
    static let saved = Color(nsColor: NDColor.saved)
    static let savedText = Color(nsColor: NDColor.savedText)
    static let read = Color(nsColor: NDColor.read)
    static let readText = Color(nsColor: NDColor.readText)
    static let red = Color(nsColor: NDColor.red)
    static let redText = Color(nsColor: NDColor.redText)
    static let redTint = Color(nsColor: NDColor.redTint)
    static let orange = Color(nsColor: NDColor.orange)
    static let orangeText = Color(nsColor: NDColor.orangeText)
    static let control = Color(nsColor: NDColor.control)
    static let controlSelected = Color(nsColor: NDColor.controlSelected)
    static let button = Color(nsColor: NDColor.button)
    static let buttonHover = Color(nsColor: NDColor.buttonHover)
    static let buttonPressed = Color(nsColor: NDColor.buttonPressed)
    static let switchOn = Color(nsColor: NDColor.switchOn)

    static let night = bg
    static let deck = raised
    static let ink = t1
    static let ink2 = t2
    static let lumen = accent
    static let lumenGraphic = saved
    static let tide = read
    static let rule = hair
    static let track = hair2
    static let fault = red
    static let warn = orange
    static let onLumen = Color.white
    static let chip = controlSelected
    static let menuSaves = saved
    static let menuReads = read
    static let menuTrack = hair2
}

/// R1 type (§ 1.2): one face, three sizes, no condensed width; figures tabular on every number.
enum NDType {
    static let title = Font.system(size: 17, weight: .semibold)
    static let body = Font.system(size: 13)
    static let bodyMedium = Font.system(size: 13, weight: .medium)
    static let bodyStrong = Font.system(size: 13, weight: .semibold)
    static let caption = Font.system(size: 11)
    static let captionMedium = Font.system(size: 11, weight: .medium)
    static let captionStrong = Font.system(size: 11, weight: .semibold)
    static let number = Font.system(size: 13).monospacedDigit()
    static let numberStrong = Font.system(size: 13, weight: .semibold).monospacedDigit()
    static let captionNumber = Font.system(size: 11).monospacedDigit()

    // Earlier names, on the three R1 sizes.
    static let headline = title
    static let pageTitle = title
    static let areaName = bodyStrong
    static let menuAreaName = body
    static let menuAreaNameStrong = bodyStrong
    static let settingsGroup = body
    static let menuHeadline = title
    static let faultTitle = title
    static let menuFaultTitle = bodyStrong
    static let row = body
    static let rowStrong = bodyStrong
    static let fact = body
    static let menuFact = body
    static let axis = caption
}

/// R1 geometry (§ 1.3, § 2.6): a 4 pt grid, 24 pt rows, a 900 pt window that takes each page's height.
enum NDGrid {
    static let windowWidth: CGFloat = 900
    /// L17: each page's own height (the window snaps to it on a tab switch, top edge fixed, no animation).
    static func windowHeight(_ page: NDPage) -> CGFloat {
        switch page {
        case .overview: 716
        case .backups: 480
        case .jobs: 512
        }
    }
    static let windowSize = NSSize(width: 900, height: 716)
    static let windowMinimum = NSSize(width: 900, height: 480)
    static let titleBar: CGFloat = 52
    static let footer: CGFloat = 28
    static let pageSide: CGFloat = 24
    static let pageTop: CGFloat = 20
    static let row: CGFloat = 24
    static let formRow: CGFloat = 36
    static let columnGutter: CGFloat = 24
    static let rowPitch: CGFloat = 24
    // Earlier rail and panel names (Backups and Jobs set their own columns, § 1.3).
    static let rowName: CGFloat = 196
    static let rowGap: CGFloat = 10
    static let rowLane: CGFloat = 24
    static let rowNumber: CGFloat = 84
    static let railName: CGFloat = 220
    static let railNumber: CGFloat = 84
    static let railSwitch: CGFloat = 44
    /// The dropdown: 320 pt, 5 pt menu inset + 14 pt row inset, `16 symbol | 8 | label | flexible | value`.
    static let menuWidth: CGFloat = 320
    static let menuInset: CGFloat = 19
    static let menuMark: CGFloat = 16
    static let menuMarkGap: CGFloat = 8
    static let menuName: CGFloat = 92
    static let menuVisual: CGFloat = 128
    static let menuAreaRow: CGFloat = 24
}

/// R1 motion (§ 3.4): the strong ease-out for what moves because a value changed; nothing continuous.
enum NDMotion {
    static let shift = Animation.timingCurve(0.23, 1, 0.32, 1, duration: 0.24)
    static let needsBand = Animation.easeOut(duration: 0.2)
    static let headlineLeaves = Animation.easeOut(duration: 0.15)
    static let rangeFade = Animation.easeInOut(duration: 0.12)
    static let breakAppears = Animation.easeOut(duration: 0.2)
    static let popoverOpens = Animation.timingCurve(0.23, 1, 0.32, 1, duration: 0.16)
    static let buttonLeaves = Animation.easeOut(duration: 0.1)
    static let waitingEnters = Animation.timingCurve(0.23, 1, 0.32, 1, duration: 0.15).delay(0.1)
    static let runningPeriod: TimeInterval = 1.6
}
