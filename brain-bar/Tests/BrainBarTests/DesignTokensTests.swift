import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

final class DesignTokensTests: XCTestCase {
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

    func testDashboardLayoutUsesAiryGlassSpacingAndHeroScale() {
        let layout = BrainBarDashboardLayout(containerSize: CGSize(width: 1100, height: 760))

        XCTAssertGreaterThanOrEqual(layout.outerPadding, 36)
        XCTAssertGreaterThanOrEqual(layout.sectionSpacing, 28)
        XCTAssertGreaterThanOrEqual(layout.gridSpacing, 20)
        XCTAssertEqual(layout.metricValueFontSize, BrainBarDesignTokens.TypeScale.hero, accuracy: 0.001)
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
}
