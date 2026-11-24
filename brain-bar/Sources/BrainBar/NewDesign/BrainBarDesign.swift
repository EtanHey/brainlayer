import AppKit
import SwiftUI

/// L6: one preference decides the whole UI, read live by the status item, the dropdown, the window and Settings.
/// An absent preference selects New; an explicit stored choice keeps precedence.
enum BrainBarDesign: String, CaseIterable, Sendable {
    case classic, new

    static let key = "BrainBarDesign"
    static let defaultValue: BrainBarDesign = .new

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
    // R1.5 changes 10 to 21 (L43): light designed on its own ground: a calmer plane, a softer white card, darker lines and
    // pipes, a calmer violet, a deeper red and amber. Dark is unchanged.
    private static func white(_ a: CGFloat) -> NSColor { NSColor(white: 1, alpha: a) }
    private static func black(_ a: CGFloat) -> NSColor { NSColor(white: 0, alpha: a) }
    /// R1.2 change 2: the light plane one step darker, so a white surface reads.
    static let bg = dynamic("bg", dark: rgb(0x1B1B1E), light: rgb(0xE4E4E8))
    /// R1.2 change 2: the one surface treatment for every block (no shadow, 0.5 pt inner line, radius 12).
    static let card = dynamic("card", dark: rgb(0x242428), light: rgb(0xFAFAFB))
    static let cardLine = dynamic("card.line", dark: white(0.06), light: black(0.10))
    static let raised = dynamic("raised", dark: rgb(0x242428), light: rgb(0xFFFFFF))
    static let pop = dynamic("pop", dark: rgb(0x2C2C31), light: rgb(0xFFFFFF))
    static let bar = dynamic("bar", dark: rgb(0x212124), light: rgb(0xEFEFF1))
    static let node = dynamic("node", dark: rgb(0x2A2A2F), light: rgb(0xF0F0F3))
    static let t1 = dynamic("t1", dark: rgb(0xEDEDF0), light: rgb(0x1D1D1F))
    static let t2 = dynamic("t2", dark: rgb(0xA1A1A8), light: rgb(0x55555D))
    static let t3 = dynamic("t3", dark: rgb(0x8A8A92), light: rgb(0x64646C))
    static let hair = dynamic("hair", dark: white(0.08), light: black(0.09))
    static let hair2 = dynamic("hair2", dark: white(0.14), light: black(0.17))
    /// R1.12 (§ D): the tab badge, a neutral count pill: `t1` on white 16 % (dark, 8.19 : 1) / black 10 % (light, 11.37 : 1).
    static let badge = dynamic("badge", dark: white(0.16), light: black(0.10))
    static let pipe = dynamic("pipe", dark: rgb(0x6E6E77), light: rgb(0x6E6E77))
    static let quiet = dynamic("quiet", dark: rgb(0x45454C), light: rgb(0xCACACF))
    static let accent = dynamic("accent", dark: rgb(0x2E6FE8), light: rgb(0x1F5FD6))
    static let accentHover = dynamic("accent.h", dark: rgb(0x4A86F5), light: rgb(0x3574E6))
    static let accentPressed = dynamic("accent.p", dark: rgb(0x2462D6), light: rgb(0x1A53BD))
    static let focus = dynamic("focus", dark: rgb(0x4D8DFF), light: rgb(0x1F5FD6))
    static let saved = dynamic("saved", dark: rgb(0xA995FF), light: rgb(0x7A63D4))
    static let savedText = dynamic("saved.t", dark: rgb(0xB6A6FF), light: rgb(0x5F48C2))
    /// R1.2 change 21: dark read bars #8189A0 (4.69:1 on the card); light unchanged.
    static let read = dynamic("read", dark: rgb(0x8189A0), light: rgb(0x7E8598))
    static let readText = dynamic("read.t", dark: rgb(0x9AA1B3), light: rgb(0x5A6275))
    static let agentRead = dynamic("agent.read", dark: rgb(0x4CB8AC), light: rgb(0x16776F))
    static let red = dynamic("red", dark: rgb(0xFF453A), light: rgb(0xD92D20))
    static let redText = dynamic("red.t", dark: rgb(0xFF6961), light: rgb(0xB8180D))
    static let redTint = dynamic("red.tint", dark: rgb(0xFF453A, alpha: 0.16), light: rgb(0xFF3B30, alpha: 0.12))
    static let orange = dynamic("orange", dark: rgb(0xFF9F0A), light: rgb(0xB86E00))
    static let orangeText = dynamic("orange.t", dark: rgb(0xFFB340), light: rgb(0x995700))
    static let control = dynamic("ctl", dark: white(0.07), light: black(0.06))
    static let controlSelected = dynamic("ctl.sel", dark: white(0.17), light: rgb(0xFFFFFF))
    static let button = dynamic("btn", dark: white(0.10), light: rgb(0xFFFFFF))
    static let buttonHover = dynamic("btn.h", dark: white(0.17), light: rgb(0xF0F0F2))
    static let buttonPressed = dynamic("btn.p", dark: white(0.24), light: rgb(0xE4E4E8))
    /// R1.6 item 12: an on switch shows its state in the system accent; the system's gray is for off only.
    static let switchOn = NSColor.controlAccentColor

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
    /// R1.7 item 1: the dropdown's words sit straight on the system menu material, which composites to grey 128 (dark)
    /// and 135 (light) over a white or black desktop (round-2 captures). Light words are near black, so they hold
    /// 4.5 : 1 everywhere; dark words are white, the most the material allows (3.95 : 1 over a white desktop).
    static let menuText1 = dynamic("menu.t1", dark: rgb(0xFFFFFF), light: rgb(0x000000))
    static let menuText2 = dynamic("menu.t2", dark: rgb(0xE6E6EA), light: rgb(0x1E1E21))
}

/// SwiftUI handles of the same colours.
enum ND {
    static let bg = Color(nsColor: NDColor.bg)
    static let raised = Color(nsColor: NDColor.raised)
    static let card = Color(nsColor: NDColor.card)
    static let cardLine = Color(nsColor: NDColor.cardLine)
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
    static let agentRead = Color(nsColor: NDColor.agentRead)
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
    static let menuText1 = Color(nsColor: NDColor.menuText1)
    static let menuText2 = Color(nsColor: NDColor.menuText2)
}

/// The system menu material as measured (round-2 captures, macOS 27, native 2x): over a white desktop the dark glass
/// composites to grey 128 and the light glass to 250; over a black desktop to 44 and 135. As a tint over the desktop
/// that is dark 65.6 at 0.671 and light 246 at 0.549 (sRGB, 0 to 255). R1.7 item 1: the rows draw no backing on it.
enum NDMenuMaterial {
    static let dark = (tint: CGFloat(65.6), alpha: CGFloat(0.671))
    static let light = (tint: CGFloat(246), alpha: CGFloat(0.549))
}

/// R1.12 (L54, CHANGES-R1.12 § A): the type scale is 24 / 17 / 13 / 11, and every word the new surfaces set takes one
/// of these four sizes through `NDType`.
enum NDScale {
    /// The window's headline and the Backups and Jobs page titles.
    static let headline: CGFloat = 24
    /// The one big number per card (the Activity totals), and the dropdown's headline.
    static let total: CGFloat = 17
    /// Row titles, station names, section labels, body text.
    static let body: CGFloat = 13
    /// Statuses, captions, axis labels, panel text.
    static let caption: CGFloat = 11
    static let all: [CGFloat] = [headline, total, body, caption]
}

/// R1 type (§ 1.2): one face, the four sizes of `NDScale`, no condensed width; figures tabular on every number.
enum NDType {
    /// R1.12 change 1: 24 pt Semibold, a 30 pt line, tracking −0.4, `t1`.
    static let windowHeadline = Font.system(size: NDScale.headline, weight: .semibold)
    static let headlineTracking: CGFloat = -0.4
    static let headlineLine: CGFloat = 30
    /// R1.12 change 2: "Activity" and "Pipeline", 13 Semibold in `t2`, a step below the headline in size and colour.
    static let section = Font.system(size: NDScale.body, weight: .semibold)
    static let title = Font.system(size: NDScale.total, weight: .semibold)
    static let body = Font.system(size: NDScale.body)
    static let bodyMedium = Font.system(size: NDScale.body, weight: .medium)
    static let bodyStrong = Font.system(size: NDScale.body, weight: .semibold)
    /// R1.7 item 10: a hover panel's body and its rows' values. R1.12: on the scale's caption step.
    static let panelBodySize: CGFloat = NDScale.caption
    static let panelBody = Font.system(size: NDScale.caption)
    static let panelValue = Font.system(size: NDScale.caption, weight: .semibold).monospacedDigit()
    static let caption = Font.system(size: NDScale.caption)
    static let captionMedium = Font.system(size: NDScale.caption, weight: .medium)
    static let captionStrong = Font.system(size: NDScale.caption, weight: .semibold)
    static let number = Font.system(size: NDScale.body).monospacedDigit()
    static let numberStrong = Font.system(size: NDScale.body, weight: .semibold).monospacedDigit()
    static let captionNumber = Font.system(size: NDScale.caption).monospacedDigit()
    static let total = Font.system(size: NDScale.total, weight: .semibold).monospacedDigit()

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
    /// R1.12 (changes 4 and B): the Overview's vertical rhythm.
    static let headlineToRows: CGFloat = 14
    static let toActivity: CGFloat = 24
    static let activityToPipeline: CGFloat = 0  // L63b: retain the original gap; fit bubbles within the diagram.
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
    /// R2A item 4: a range switch never draws two charts at once: the old bars leave over 100 ms, then the new arrive
    /// over 150 ms on the strong ease-out.
    static let rangeOutDuration: Double = 0.1
    static let rangeOut = Animation.easeOut(duration: rangeOutDuration)
    static let rangeIn = Animation.timingCurve(0.23, 1, 0.32, 1, duration: 0.15).delay(rangeOutDuration)
    /// R2A item 7, R2 item 3: the window's height change on a page switch: 200 ms ease-out, top edge fixed; none under
    /// Reduce Motion.
    static let pageResizeDuration: Double = 0.2
    static func pageResize(reduceMotion: Bool) -> Double? { reduceMotion ? nil : pageResizeDuration }

    static let breakAppears = Animation.easeOut(duration: 0.2)
    static let popoverOpens = Animation.timingCurve(0.23, 1, 0.32, 1, duration: 0.16)
    /// R1.2 change 24: the pressed button fades over 160 ms ease-out, holding its pressed colour.
    static let buttonLeaves = Animation.easeOut(duration: 0.16)
    /// Council r2a finding 4: no overlap. The waiting line starts when the button has gone (and the button comes back
    /// only after the line has gone), so the two are never visible at once.
    static let buttonLeavesDuration: Double = 0.16
    static let waitingDelay: Double = buttonLeavesDuration
    static let buttonReturns = Animation.easeOut(duration: 0.16).delay(buttonLeavesDuration)
    static let waitingEnters = Animation.timingCurve(0.23, 1, 0.32, 1, duration: 0.16).delay(waitingDelay)
    static let runningPeriod: TimeInterval = 1.6
}

/// R1.8 item 1: a page switch shows the new page first, in both directions, so the tab and the page always agree: the
/// page swaps at once (R1.10.1: no cross-fade, see the controller), and the window's height moves 200 ms on the strong ease-out curve
/// with its top edge fixed (a growing window reveals the new page's foot as it opens). Without animation (Reduce Motion,
/// or no visible window) both change together. 1.6's "frame first" kept the old page up through every growing resize.
enum NDPageSwitch: Equatable {
    case pageThenFrame, together
    static func order(from current: CGFloat, to target: CGFloat, animates: Bool) -> NDPageSwitch {
        guard animates, abs(current - target) > 0.5 else { return .together }
        return .pageThenFrame
    }
    /// cubic-bezier(0.23, 1, 0.32, 1).
    static let curve: (Float, Float, Float, Float) = (0.23, 1, 0.32, 1)
}
