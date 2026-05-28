import AppKit
import SwiftUI

enum BrainBarDesignTokens {
    enum Colors {
        static let backgroundAbyss = NSColor.brainBarHex(0x070B14)
        static let backgroundBase = NSColor.brainBarHex(0x0C1220)
        static let backgroundRaised = NSColor.brainBarHex(0x111A2C)

        static let glassPrimaryBase = NSColor.brainBarRGB(red: 28, green: 40, blue: 66)
        static let glassSecondaryBase = NSColor.brainBarRGB(red: 38, green: 53, blue: 84)
        static let glassTertiaryBase = NSColor.brainBarRGB(red: 52, green: 70, blue: 108)
        static let glassHighlightBase = NSColor.brainBarRGB(red: 120, green: 150, blue: 210)

        static let borderSoft = NSColor.brainBarRGB(red: 150, green: 175, blue: 225, alpha: 0.10)
        static let borderEdge = NSColor.brainBarRGB(red: 180, green: 205, blue: 255, alpha: 0.18)
        static let borderGlow = NSColor.brainBarRGB(red: 110, green: 160, blue: 255, alpha: 0.34)

        static let textPrimary = NSColor.brainBarHex(0xEEF3FB)
        static let textSecondary = NSColor.brainBarHex(0xAEBBD4)
        static let textMuted = NSColor.brainBarHex(0x6B7A98)

        static let accent = NSColor.brainBarHex(0x6EA0FF)
        static let accentBright = NSColor.brainBarHex(0x8FB6FF)
        static let accentDeep = NSColor.brainBarHex(0x3F6FE0)
        static let accentViolet = NSColor.brainBarHex(0xA98BFF)

        static let white = NSColor.white
        static let black = NSColor.black
    }

    enum Glass {
        static let primaryAlpha: CGFloat = 0.34
        static let secondaryAlpha: CGFloat = 0.26
        static let tertiaryAlpha: CGFloat = 0.22
        static let highlightAlpha: CGFloat = 0.08

        static let primary = Colors.glassPrimaryBase.withAlphaComponent(primaryAlpha)
        static let secondary = Colors.glassSecondaryBase.withAlphaComponent(secondaryAlpha)
        static let tertiary = Colors.glassTertiaryBase.withAlphaComponent(tertiaryAlpha)
        static let highlight = Colors.glassHighlightBase.withAlphaComponent(highlightAlpha)

        static let primaryMaterial: Material = .ultraThinMaterial
        static let secondaryMaterial: Material = .thinMaterial
    }

    enum Blur {
        static let lightRadius: CGFloat = 14
        static let faintRadius: CGFloat = 8
        static let lightSaturation: CGFloat = 1.35
        static let faintSaturation: CGFloat = 1.25
    }

    enum TypeScale {
        static let hero: CGFloat = 72
        static let display: CGFloat = 28
        static let title: CGFloat = 20
        static let heading: CGFloat = 15
        static let body: CGFloat = 13
        static let label: CGFloat = 11
        static let mono: CGFloat = 12
    }

    enum Radius {
        static let xl: CGFloat = 26
        static let lg: CGFloat = 18
        static let md: CGFloat = 13
        static let sm: CGFloat = 9
        static let pill: CGFloat = 999
    }

    enum Shadow {
        static let deepColor = NSColor.black.withAlphaComponent(0.55)
        static let softColor = NSColor.black.withAlphaComponent(0.42)
        static let insetTop = NSColor.white.withAlphaComponent(0.06)
    }
}

enum BrainBarStateTheme: CaseIterable, Sendable, Equatable {
    case idle
    case active
    case loading
    case empty
    case degraded
    case error

    var theme: Theme {
        switch self {
        case .idle:
            Theme(color: NSColor.brainBarHex(0x506C8A), glow: NSColor.brainBarRGB(red: 80, green: 108, blue: 138, alpha: 0.28))
        case .active:
            Theme(color: NSColor.brainBarHex(0x30DC97), glow: NSColor.brainBarRGB(red: 48, green: 220, blue: 151, alpha: 0.35))
        case .loading:
            Theme(color: NSColor.brainBarHex(0x6EA0FF), glow: NSColor.brainBarRGB(red: 110, green: 160, blue: 255, alpha: 0.35))
        case .empty:
            Theme(color: NSColor.brainBarHex(0x4A5878), glow: NSColor.brainBarRGB(red: 74, green: 88, blue: 120, alpha: 0.24))
        case .degraded:
            Theme(color: NSColor.brainBarHex(0xF5B34A), glow: NSColor.brainBarRGB(red: 245, green: 179, blue: 74, alpha: 0.32))
        case .error:
            Theme(color: NSColor.brainBarHex(0xFF6B7D), glow: NSColor.brainBarRGB(red: 255, green: 107, blue: 125, alpha: 0.35))
        }
    }

    struct Theme: Sendable, Equatable {
        let color: NSColor
        let glow: NSColor

        var swiftUIColor: Color { Color(nsColor: color) }
        var glowSwiftUIColor: Color { Color(nsColor: glow) }
    }
}

extension NSColor {
    static let brainBarClear = NSColor.clear

    static func brainBarHex(_ hex: UInt32, alpha: CGFloat = 1) -> NSColor {
        let red = CGFloat((hex >> 16) & 0xFF) / 255
        let green = CGFloat((hex >> 8) & 0xFF) / 255
        let blue = CGFloat(hex & 0xFF) / 255
        return NSColor(deviceRed: red, green: green, blue: blue, alpha: alpha)
    }

    static func brainBarRGB(red: CGFloat, green: CGFloat, blue: CGFloat, alpha: CGFloat = 1) -> NSColor {
        NSColor(deviceRed: red / 255, green: green / 255, blue: blue / 255, alpha: alpha)
    }
}

extension Color {
    static func brainBar(nsColor: NSColor) -> Color {
        Color(nsColor: nsColor)
    }

    static func brainBarRGB(red: Double, green: Double, blue: Double, opacity: Double = 1) -> Color {
        Color(red: red, green: green, blue: blue, opacity: opacity)
    }

    static func brainBarHSB(hue: Double, saturation: Double, brightness: Double, opacity: Double = 1) -> Color {
        Color(hue: hue, saturation: saturation, brightness: brightness, opacity: opacity)
    }

    static let brainBarBackgroundAbyss = Color(nsColor: BrainBarDesignTokens.Colors.backgroundAbyss)
    static let brainBarBackgroundBase = Color(nsColor: BrainBarDesignTokens.Colors.backgroundBase)
    static let brainBarBackgroundRaised = Color(nsColor: BrainBarDesignTokens.Colors.backgroundRaised)
    static let brainBarGlassPrimary = Color(nsColor: BrainBarDesignTokens.Glass.primary)
    static let brainBarGlassSecondary = Color(nsColor: BrainBarDesignTokens.Glass.secondary)
    static let brainBarGlassTertiary = Color(nsColor: BrainBarDesignTokens.Glass.tertiary)
    static let brainBarGlassHighlight = Color(nsColor: BrainBarDesignTokens.Glass.highlight)
    static let brainBarBorderSoft = Color(nsColor: BrainBarDesignTokens.Colors.borderSoft)
    static let brainBarBorderEdge = Color(nsColor: BrainBarDesignTokens.Colors.borderEdge)
    static let brainBarBorderGlow = Color(nsColor: BrainBarDesignTokens.Colors.borderGlow)
    static let brainBarTextPrimary = Color(nsColor: BrainBarDesignTokens.Colors.textPrimary)
    static let brainBarTextSecondary = Color(nsColor: BrainBarDesignTokens.Colors.textSecondary)
    static let brainBarTextMuted = Color(nsColor: BrainBarDesignTokens.Colors.textMuted)
    static let brainBarAccent = Color(nsColor: BrainBarDesignTokens.Colors.accent)
    static let brainBarAccentBright = Color(nsColor: BrainBarDesignTokens.Colors.accentBright)
    static let brainBarAccentDeep = Color(nsColor: BrainBarDesignTokens.Colors.accentDeep)
    static let brainBarAccentViolet = Color(nsColor: BrainBarDesignTokens.Colors.accentViolet)
    static let brainBarWhite = Color(nsColor: BrainBarDesignTokens.Colors.white)
    static let brainBarBlack = Color(nsColor: BrainBarDesignTokens.Colors.black)
    static let brainBarClear = Color.clear
    static let brainBarSubtleFill = Color(nsColor: BrainBarDesignTokens.Colors.textPrimary).opacity(0.06)
    static let brainBarMutedFill = Color(nsColor: BrainBarDesignTokens.Colors.textPrimary).opacity(0.08)
}
