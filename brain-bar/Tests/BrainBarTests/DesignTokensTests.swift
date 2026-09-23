import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

final class DesignTokensTests: XCTestCase {
    deinit {}

    func testGroundTruthGlassRedesignTokensMatchMandate() {
        XCTAssertEqual(BrainBarDesignTokens.Colors.backgroundAbyss.hexRGB, "#070B14")
        XCTAssertEqual(BrainBarDesignTokens.Colors.backgroundBase.hexRGB, "#0C1220")
        XCTAssertEqual(BrainBarDesignTokens.Colors.accent.hexRGB, "#6EA0FF")
        XCTAssertEqual(BrainBarDesignTokens.Colors.accentViolet.hexRGB, "#A98BFF")
        XCTAssertEqual(BrainBarDesignTokens.Colors.graphCanvasLightTop.hexRGB, "#F2F2EB")
        XCTAssertEqual(BrainBarDesignTokens.Colors.graphCanvasLightBottom.hexRGB, "#E6EBF0")

        XCTAssertEqual(BrainBarDesignTokens.Glass.primaryAlpha, 0.34, accuracy: 0.001)
        XCTAssertEqual(BrainBarDesignTokens.Glass.secondaryAlpha, 0.26, accuracy: 0.001)
        XCTAssertEqual(BrainBarDesignTokens.Glass.tertiaryAlpha, 0.22, accuracy: 0.001)
        XCTAssertEqual(BrainBarDesignTokens.Blur.lightRadius, 14, accuracy: 0.001)
        XCTAssertEqual(BrainBarDesignTokens.Blur.faintRadius, 8, accuracy: 0.001)
        XCTAssertEqual(BrainBarDesignTokens.TypeScale.hero, 72, accuracy: 0.001)
    }

    @MainActor
    func testSemanticStatusesStaySeparateFromSignalsAndTextHonorsNinePointFloor() {
        XCTAssertEqual(BrainBarDesignTokens.Colors.statusOK.hexRGB, "#32D74B")
        XCTAssertEqual(BrainBarDesignTokens.Colors.statusAttention.hexRGB, "#FFD60A")
        XCTAssertEqual(BrainBarDesignTokens.Colors.statusError.hexRGB, "#FF453A")
        XCTAssertEqual(BrainBarDesignTokens.Colors.statusUnknown.hexRGB, "#8A8A90")
        let chartPaletteColors: [(String, NSColor)] = [
            ("Accent", BrainBarDesignTokens.Colors.accent),
            ("Accent bright", BrainBarDesignTokens.Colors.accentBright),
            ("Accent deep", BrainBarDesignTokens.Colors.accentDeep),
            ("Accent violet", BrainBarDesignTokens.Colors.accentViolet),
            ("Vector", BrainBarDesignTokens.Colors.signalVector),
            ("FTS5", BrainBarDesignTokens.Colors.signalFTS5),
            ("Trigram", BrainBarDesignTokens.Colors.signalTrigram),
            ("Agent", BrainBarDesignTokens.Colors.seriesAgent),
            ("Watcher", BrainBarDesignTokens.Colors.seriesWatcher),
            ("Agent dimmed", BrainBarDesignTokens.Colors.seriesAgentDimmed),
            ("Watcher dimmed", BrainBarDesignTokens.Colors.seriesWatcherDimmed),
        ]
        let statusColors: [(String, NSColor)] = [
            ("OK", BrainBarDesignTokens.Colors.statusOK),
            ("attention", BrainBarDesignTokens.Colors.statusAttention),
            ("error", BrainBarDesignTokens.Colors.statusError),
            ("unknown", BrainBarDesignTokens.Colors.statusUnknown),
        ]
        for (statusName, statusColor) in statusColors {
            for (signalName, signalColor) in chartPaletteColors {
                XCTAssertGreaterThanOrEqual(
                    statusColor.cie76Distance(to: signalColor),
                    25,
                    "\(statusName) status token is too close to \(signalName)"
                )
            }
        }

        XCTAssertEqual(BrainBarDesignTokens.TypeScale.textSize(8), 9, accuracy: 0.001)
        XCTAssertEqual(BrainBarDesignTokens.TypeScale.textSize(11), 11, accuracy: 0.001)
        XCTAssertEqual(DegradationBadge.labelFontSize * DegradationBadge.minimumLabelScaleFactor, 9, accuracy: 0.001)
        XCTAssertEqual(BrainBarFlowStatusPill.fontSize * BrainBarFlowStatusPill.minimumScaleFactor, 9, accuracy: 0.001)
        XCTAssertEqual(BrainBarDesignTokens.TypeScale.textSize(8), 9, accuracy: 0.001)
        XCTAssertEqual(
            BrainBarDesignTokens.Colors.signalCoverageStatus(indexedCount: 1_000, eligibleCount: 1_000, isAvailable: true),
            BrainBarDesignTokens.Colors.statusOK
        )
        XCTAssertEqual(
            BrainBarDesignTokens.Colors.signalCoverageStatus(indexedCount: 0, eligibleCount: 1_000, isAvailable: false),
            BrainBarDesignTokens.Colors.statusUnknown
        )
        XCTAssertEqual(
            BrainBarDesignTokens.Colors.signalCoverageStatus(indexedCount: 999, eligibleCount: 1_000, isAvailable: true),
            BrainBarDesignTokens.Colors.statusUnknown,
            "Incomplete coverage with no measured progress stays neutral"
        )
        XCTAssertEqual(
            BrainBarDesignTokens.Colors.signalCoverageStatus(indexedCount: 999, eligibleCount: 1_000,
                                                             isAvailable: true, lastError: "coverage query failed"),
            BrainBarDesignTokens.Colors.statusAttention,
            "An actual coverage failure needs an attention reason"
        )
        XCTAssertEqual(
            BrainBarDesignTokens.Colors.signalCoverageStatus(indexedCount: 1_000, eligibleCount: 1_000, isAvailable: true),
            BrainBarDesignTokens.Colors.statusOK
        )
        XCTAssertEqual(
            BrainBarDesignTokens.Colors.signalCoverageStatus(indexedCount: 0, eligibleCount: 0, isAvailable: false),
            BrainBarDesignTokens.Colors.statusUnknown,
            "Computing and unavailable coverage must stay unknown"
        )

        let scaledFonts = [
            (DegradationBadge.labelFontSize, DegradationBadge.minimumLabelScaleFactor),
            (BrainBarFlowStatusPill.fontSize, BrainBarFlowStatusPill.minimumScaleFactor),
            (CGFloat(11), BrainBarDesignTokens.TypeScale.minimumScaleFactor(for: 11)),
            (CGFloat(40), BrainBarDesignTokens.TypeScale.minimumScaleFactor(for: 40)),
        ]
        for (fontSize, scaleFactor) in scaledFonts {
            XCTAssertGreaterThanOrEqual(
                fontSize * scaleFactor,
                BrainBarDesignTokens.TypeScale.minimumText,
                "Scaled \(fontSize) pt text can fall below the floor"
            )
        }
    }

    func testStateThemesExposeGroundTruthSemanticColors() {
        XCTAssertEqual(BrainBarStateTheme.idle.theme.color.hexRGB, "#506C8A")
        XCTAssertEqual(BrainBarStateTheme.active.theme.color.hexRGB, "#30DC97")
        XCTAssertEqual(BrainBarStateTheme.loading.theme.color.hexRGB, "#6EA0FF")
        XCTAssertEqual(BrainBarStateTheme.empty.theme.color.hexRGB, "#4A5878")
        XCTAssertEqual(BrainBarStateTheme.degraded.theme.color.hexRGB, "#F5B34A")
        XCTAssertEqual(BrainBarStateTheme.error.theme.color.hexRGB, "#FF6B7D")
    }

    func testPipelineStatesMapToGlassStateThemes() {
        XCTAssertEqual(PipelineState.idle.stateTheme, .idle)
        XCTAssertEqual(PipelineState.indexing.stateTheme, .loading)
        XCTAssertEqual(PipelineState.enriching.stateTheme, .active)
        XCTAssertEqual(PipelineState.degraded.stateTheme, .degraded)

        XCTAssertEqual(PipelineIndicatorStatus.live.stateTheme, .active)
        XCTAssertEqual(PipelineIndicatorStatus.queued.stateTheme, .loading)
        XCTAssertEqual(PipelineIndicatorStatus.idle.stateTheme, .idle)
        XCTAssertEqual(PipelineIndicatorStatus.unavailable.stateTheme, .error)
    }

    func testDashboardLayoutUsesCompactOperatorDensity() {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 1100, height: 760))

        XCTAssertLessThanOrEqual(layout.outerPadding, 24)
        XCTAssertLessThanOrEqual(layout.sectionSpacing, 20)
        XCTAssertLessThanOrEqual(layout.gridSpacing, 16)
        XCTAssertLessThanOrEqual(layout.metricValueFontSize, 40)
        XCTAssertEqual(layout.panelCornerRadius, BrainBarDesignTokens.Radius.xl, accuracy: 0.001)
    }

    func testSwiftUIColorRGBHelperUsesByteComponents() {
        let color = NSColor(Color.brainBarRGB(red: 28, green: 40, blue: 66, opacity: 0.5))

        XCTAssertEqual(color.hexRGB, "#1C2842")
        XCTAssertEqual(color.alphaComponent, 0.5, accuracy: 0.001)
    }
}

private extension NSColor {
    var hexRGB: String {
        guard let color = usingColorSpace(.deviceRGB) else {
            XCTFail("Expected RGB-compatible color")
            return ""
        }
        let r = Int((color.redComponent * 255).rounded())
        let g = Int((color.greenComponent * 255).rounded())
        let b = Int((color.blueComponent * 255).rounded())
        return String(format: "#%02X%02X%02X", r, g, b)
    }

    func cie76Distance(to other: NSColor) -> Double {
        let lhs = labComponents
        let rhs = other.labComponents
        return sqrt(pow(lhs.l - rhs.l, 2) + pow(lhs.a - rhs.a, 2) + pow(lhs.b - rhs.b, 2))
    }

    private var labComponents: (l: Double, a: Double, b: Double) {
        let rgb = usingColorSpace(.deviceRGB)!
        func linear(_ component: Double) -> Double {
            component <= 0.04045 ? component / 12.92 : pow((component + 0.055) / 1.055, 2.4)
        }
        let red = linear(rgb.redComponent)
        let green = linear(rgb.greenComponent)
        let blue = linear(rgb.blueComponent)
        let x = (red * 0.4124564 + green * 0.3575761 + blue * 0.1804375) / 0.95047
        let y = red * 0.2126729 + green * 0.7151522 + blue * 0.0721750
        let z = (red * 0.0193339 + green * 0.1191920 + blue * 0.9503041) / 1.08883
        func labCurve(_ value: Double) -> Double {
            value > 0.008856451679 ? pow(value, 1.0 / 3.0) : 7.787037037 * value + 16.0 / 116.0
        }
        let fx = labCurve(x)
        let fy = labCurve(y)
        let fz = labCurve(z)
        return (116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
    }
}
