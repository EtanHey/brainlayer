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

    // Window planes and inks.
    static let night = dynamic("night", dark: rgb(0x0E1322), light: rgb(0xF5F6F9))
    static let deck = dynamic("deck", dark: rgb(0x151B2D), light: rgb(0xFFFFFF))
    static let ink = dynamic("ink", dark: rgb(0xE6EAF2), light: rgb(0x131826))
    static let ink2 = dynamic("ink2", dark: rgb(0x9BA5BA), light: rgb(0x566079))
    static let lumen = dynamic("lumen", dark: rgb(0x8EA8FF), light: rgb(0x3559D6))
    static let lumenGraphic = dynamic("lumen.graphic", dark: rgb(0x8EA8FF), light: rgb(0x5873E0))
    static let tide = dynamic("tide", dark: rgb(0x58688E), light: rgb(0x7F8CA8))
    /// Structure only, never meaning alone. Increase Contrast raises both to at least 3:1 on Night.
    static let rule = dynamic("rule", dark: rgb(0x252D44), light: rgb(0xE1E5ED),
                              contrastDark: rgb(0x5A6684), contrastLight: rgb(0x8A93A8))
    static let track = dynamic("track", dark: rgb(0x3A4560), light: rgb(0xC3CAD8),
                               contrastDark: rgb(0x6B7798), contrastLight: rgb(0x7D879C))
    static let fault = dynamic("fault", dark: rgb(0xFF6B5E), light: rgb(0xC8302A))
    static let warn = dynamic("warn", dark: rgb(0xF2B54A), light: rgb(0x9A5C00))
    /// Button text on Lumen.
    static let onLumen = dynamic("on.lumen", dark: rgb(0x0E1322), light: rgb(0xFFFFFF))
    /// The range control's trough and selected chip.
    static let chip = dynamic("chip", dark: rgb(0x0E1322), light: rgb(0xFFFFFF))

    // The dropdown: system label colours for words, these for data (§ 2.1).
    static let menuSaves = dynamic("menu.saves", dark: rgb(0x9DB3FF), light: rgb(0x5873E0))
    static let menuReads = dynamic("menu.reads", dark: rgb(0x7A87A6, alpha: 0.75), light: rgb(0x7C879E))
    static let menuTrack = dynamic("menu.track", dark: NSColor(white: 1, alpha: 0.22), light: NSColor(white: 0, alpha: 0.20),
                                   contrastDark: NSColor(white: 1, alpha: 0.5), contrastLight: NSColor(white: 0, alpha: 0.5))
}

/// SwiftUI handles of the same colours.
enum ND {
    static let night = Color(nsColor: NDColor.night)
    static let deck = Color(nsColor: NDColor.deck)
    static let ink = Color(nsColor: NDColor.ink)
    static let ink2 = Color(nsColor: NDColor.ink2)
    static let lumen = Color(nsColor: NDColor.lumen)
    static let lumenGraphic = Color(nsColor: NDColor.lumenGraphic)
    static let tide = Color(nsColor: NDColor.tide)
    static let rule = Color(nsColor: NDColor.rule)
    static let track = Color(nsColor: NDColor.track)
    static let fault = Color(nsColor: NDColor.fault)
    static let warn = Color(nsColor: NDColor.warn)
    static let onLumen = Color(nsColor: NDColor.onLumen)
    static let chip = Color(nsColor: NDColor.chip)
    static let menuSaves = Color(nsColor: NDColor.menuSaves)
    static let menuReads = Color(nsColor: NDColor.menuReads)
    static let menuTrack = Color(nsColor: NDColor.menuTrack)
}

/// The type scale (§ 2.3). The system face only; area names in the native Condensed width; figures tabular.
enum NDType {
    static let headline = Font.system(size: 26, weight: .semibold)
    static let pageTitle = Font.system(size: 22, weight: .semibold).width(.condensed)
    static let areaName = Font.system(size: 17, weight: .semibold).width(.condensed)
    static let menuAreaName = Font.system(size: 14, weight: .medium).width(.condensed)
    static let menuAreaNameStrong = Font.system(size: 14, weight: .bold).width(.condensed)
    static let settingsGroup = Font.system(size: 15, weight: .semibold).width(.condensed)
    static let menuHeadline = Font.system(size: 15, weight: .semibold)
    static let faultTitle = Font.system(size: 14, weight: .semibold)
    static let menuFaultTitle = Font.system(size: 13, weight: .semibold)
    static let row = Font.system(size: 13)
    static let rowStrong = Font.system(size: 13, weight: .semibold)
    static let fact = Font.system(size: 12.5)
    static let menuFact = Font.system(size: 12)
    static let caption = Font.system(size: 12)
    static let axis = Font.system(size: 11)
    static let number = Font.system(size: 13).monospacedDigit()
}

/// The 8 pt grid and the fixed geometry the comps measure (§ 2.4).
enum NDGrid {
    static let windowSize = NSSize(width: 900, height: 720)
    static let windowMinimum = NSSize(width: 780, height: 560)
    static let titleBar: CGFloat = 52
    static let pageSide: CGFloat = 28
    static let pageTop: CGFloat = 16
    static let columnGutter: CGFloat = 24
    /// Panel rows: `146 name | 10 | gauge (fills) | 24 lane | 56 number`, 24 pt pitch.
    static let rowName: CGFloat = 146
    static let rowGap: CGFloat = 10
    static let rowLane: CGFloat = 24
    static let rowNumber: CGFloat = 56
    static let rowPitch: CGFloat = 24
    /// Rails: `186 name | rail (fills) | 84 number | 40 switch`.
    static let railName: CGFloat = 186
    static let railNumber: CGFloat = 84
    static let railSwitch: CGFloat = 40
    /// The dropdown: 336 pt, 14 pt inset, `12 mark | 8 | 92 name | 120 visual | number`, 26 pt area rows.
    static let menuWidth: CGFloat = 336
    static let menuInset: CGFloat = 14
    static let menuMark: CGFloat = 12
    static let menuMarkGap: CGFloat = 8
    static let menuName: CGFloat = 92
    static let menuVisual: CGFloat = 120
    static let menuAreaRow: CGFloat = 26
}

/// Motion (§ 8): one strong ease-out for things that move because a value changed.
enum NDMotion {
    static let shift = Animation.timingCurve(0.23, 1, 0.32, 1, duration: 0.24)
    static let needsBand = Animation.timingCurve(0.23, 1, 0.32, 1, duration: 0.2)
    static let waitingFade = Animation.easeOut(duration: 0.15)
    /// One turn of the running mark.
    static let runningPeriod: TimeInterval = 1.6
}
