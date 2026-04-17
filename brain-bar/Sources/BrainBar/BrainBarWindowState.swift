import AppKit
import CoreGraphics
import Foundation

enum BrainBarLaunchMode: Equatable {
    case menuBarWindow
    case legacyStatusItem

    static func resolve(environment: [String: String] = ProcessInfo.processInfo.environment) -> BrainBarLaunchMode {
        let rawValue = environment["BRAINBAR_LEGACY"]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        switch rawValue {
        case "1", "true", "yes", "on":
            return .legacyStatusItem
        default:
            return .menuBarWindow
        }
    }
}

enum BrainBarTab: Int, CaseIterable, Equatable, Identifiable {
    case dashboard = 0
    case injections = 1
    case graph = 2

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .dashboard:
            return "Dashboard"
        case .injections:
            return "Injections"
        case .graph:
            return "Graph"
        }
    }
}

enum BrainBarQuickAction: Equatable {
    case capture
    case search
}

enum BrainBarSparklineStyle: Equatable {
    case active
    case idle
}

struct BrainBarLivePresentation: Equatable {
    let sparklineStyle: BrainBarSparklineStyle
    let badgeText: String
    let statusText: String

    static func derive(stats: BrainDatabase.DashboardStats) -> BrainBarLivePresentation {
        if stats.enrichmentRatePerMinute > 0 {
            return BrainBarLivePresentation(
                sparklineStyle: .active,
                badgeText: DashboardMetricFormatter.liveBadgeString(ratePerMinute: stats.enrichmentRatePerMinute),
                statusText: "Live enrichment stream"
            )
        }

        return BrainBarLivePresentation(
            sparklineStyle: .idle,
            badgeText: "idle",
            statusText: "Idle — no enrichment in last 60s"
        )
    }
}

extension BrainBarLivePresentation {
    var accentColor: NSColor {
        switch sparklineStyle {
        case .active:
            return .systemTeal
        case .idle:
            return .secondaryLabelColor
        }
    }
}

protocol BrainBarKeyValueStoring: AnyObject {
    func string(forKey defaultName: String) -> String?
    func setString(_ value: String?, forKey defaultName: String)
}

extension UserDefaults: BrainBarKeyValueStoring {
    func setString(_ value: String?, forKey defaultName: String) {
        set(value, forKey: defaultName)
    }
}

struct BrainBarWindowFrameStore {
    static let frameDefaultsKey = "brainbar.window.frame"

    let defaults: BrainBarKeyValueStoring
    let key: String

    init(defaults: BrainBarKeyValueStoring = UserDefaults.standard, key: String = Self.frameDefaultsKey) {
        self.defaults = defaults
        self.key = key
    }

    func persistedFrame() -> CGRect? {
        guard let rawValue = defaults.string(forKey: key) else { return nil }
        let components = rawValue.split(separator: ",").compactMap { Double($0) }
        guard components.count == 4 else { return nil }
        return CGRect(
            x: components[0],
            y: components[1],
            width: components[2],
            height: components[3]
        )
    }

    func persist(frame: CGRect) {
        let encoded = "\(frame.origin.x),\(frame.origin.y),\(frame.size.width),\(frame.size.height)"
        defaults.setString(encoded, forKey: key)
    }
}

enum BrainBarWindowPlacement {
    static let defaultSize = CGSize(width: 900, height: 640)
    static let menuBarClearance: CGFloat = 18
    static let menuBarIconGap: CGFloat = 4

    static func isRestorable(frame: CGRect, screenFrames: [CGRect] = NSScreen.screens.map(\.visibleFrame)) -> Bool {
        guard frame.width >= 400, frame.height >= 300 else { return false }
        guard !screenFrames.isEmpty else { return true }
        return screenFrames.contains { screen in
            screen.intersects(frame)
        }
    }

    static func centeredDefaultFrame(screenFrames: [CGRect] = NSScreen.screens.map(\.visibleFrame)) -> CGRect? {
        guard let screenFrame = screenFrames.first else { return nil }
        return CGRect(
            x: screenFrame.midX - defaultSize.width / 2,
            y: screenFrame.midY - defaultSize.height / 2,
            width: defaultSize.width,
            height: defaultSize.height
        )
    }

    static func resolvedFrame(
        persistedFrame: CGRect?,
        screenFrames: [CGRect] = NSScreen.screens.map(\.visibleFrame)
    ) -> CGRect? {
        if let persistedFrame,
           isRestorable(frame: persistedFrame, screenFrames: screenFrames) {
            return clearingMenuBar(
                frame: persistedFrame,
                screenFrames: screenFrames
            )
        }

        guard let defaultFrame = centeredDefaultFrame(screenFrames: screenFrames) else {
            return nil
        }
        return clearingMenuBar(
            frame: defaultFrame,
            screenFrames: screenFrames
        )
    }

    static func accessibilityReferenceTop(screenFrames: [CGRect] = NSScreen.screens.map(\.frame)) -> CGFloat? {
        screenFrames.first?.maxY
    }

    static func accessibilityFrame(
        fromAppKit frame: CGRect,
        screenFrames: [CGRect] = NSScreen.screens.map(\.frame)
    ) -> CGRect? {
        guard let referenceTop = accessibilityReferenceTop(screenFrames: screenFrames) else { return nil }
        return CGRect(
            x: frame.minX,
            y: referenceTop - frame.maxY,
            width: frame.width,
            height: frame.height
        )
    }

    static func appKitFrame(
        fromAccessibility frame: CGRect,
        screenFrames: [CGRect] = NSScreen.screens.map(\.frame)
    ) -> CGRect? {
        guard let referenceTop = accessibilityReferenceTop(screenFrames: screenFrames) else { return nil }
        return CGRect(
            x: frame.minX,
            y: referenceTop - frame.minY - frame.height,
            width: frame.width,
            height: frame.height
        )
    }

    static func clearingMenuBar(
        frame: CGRect,
        screenFrames: [CGRect] = NSScreen.screens.map(\.visibleFrame),
        topGap: CGFloat = menuBarClearance
    ) -> CGRect {
        guard let screenFrame = screenFrames.first(where: { $0.intersects(frame) }) ?? screenFrames.first else {
            return frame
        }

        let maxOriginY = screenFrame.maxY - topGap - frame.height
        guard frame.origin.y > maxOriginY else { return frame }

        return CGRect(
            x: frame.origin.x,
            y: maxOriginY,
            width: frame.width,
            height: frame.height
        )
    }

    static func anchoredFrameBelowMenuBarItem(
        currentAccessibilityFrame: CGRect,
        menuBarItemAccessibilityFrame: CGRect,
        screenFrames: [CGRect] = NSScreen.screens.map(\.frame),
        visibleScreenFrames: [CGRect] = NSScreen.screens.map(\.visibleFrame),
        gap: CGFloat = menuBarIconGap
    ) -> CGRect? {
        guard let currentFrame = appKitFrame(
                fromAccessibility: currentAccessibilityFrame,
                screenFrames: screenFrames
              ),
              let menuBarItemFrame = appKitFrame(
                fromAccessibility: menuBarItemAccessibilityFrame,
                screenFrames: screenFrames
              )
        else {
            return nil
        }

        let targetOriginY = menuBarItemFrame.minY - gap - currentFrame.height
        let targetFrame = CGRect(
            x: currentFrame.minX,
            y: targetOriginY,
            width: currentFrame.width,
            height: currentFrame.height
        )

        return clamp(
            frame: targetFrame,
            nearMenuBarItemFrame: menuBarItemFrame,
            screenFrames: screenFrames,
            visibleScreenFrames: visibleScreenFrames,
            topGap: gap
        )
    }

    static func anchoredFrameBelowMenuBarItem(
        currentFrame: CGRect,
        menuBarItemAccessibilityFrame: CGRect,
        screenFrames: [CGRect] = NSScreen.screens.map(\.frame),
        visibleScreenFrames: [CGRect] = NSScreen.screens.map(\.visibleFrame),
        gap: CGFloat = menuBarIconGap
    ) -> CGRect? {
        guard let menuBarItemFrame = appKitFrame(
            fromAccessibility: menuBarItemAccessibilityFrame,
            screenFrames: screenFrames
        ) else {
            return nil
        }

        let targetOriginY = menuBarItemFrame.minY - gap - currentFrame.height
        let targetFrame = CGRect(
            x: currentFrame.minX,
            y: targetOriginY,
            width: currentFrame.width,
            height: currentFrame.height
        )

        return clamp(
            frame: targetFrame,
            nearMenuBarItemFrame: menuBarItemFrame,
            screenFrames: screenFrames,
            visibleScreenFrames: visibleScreenFrames,
            topGap: gap
        )
    }

    static func preferredMenuBarItemFrame(
        candidates: [CGRect],
        mouseLocation: CGPoint
    ) -> CGRect? {
        guard !candidates.isEmpty else { return nil }
        if let containingCandidate = candidates.first(where: { $0.contains(mouseLocation) }) {
            return containingCandidate
        }

        return candidates.min { lhs, rhs in
            distanceSquared(from: mouseLocation, to: lhs) < distanceSquared(from: mouseLocation, to: rhs)
        }
    }

    private static func clamp(
        frame: CGRect,
        nearMenuBarItemFrame menuBarItemFrame: CGRect,
        screenFrames: [CGRect],
        visibleScreenFrames: [CGRect],
        topGap: CGFloat
    ) -> CGRect {
        guard let visibleScreenFrame = visibleScreenFrame(
            nearMenuBarItemFrame: menuBarItemFrame,
            screenFrames: screenFrames,
            visibleScreenFrames: visibleScreenFrames
        ) else {
            return frame
        }

        let maxOriginX = max(visibleScreenFrame.minX, visibleScreenFrame.maxX - frame.width)
        let maxOriginY = visibleScreenFrame.maxY - topGap - frame.height

        return CGRect(
            x: min(max(frame.origin.x, visibleScreenFrame.minX), maxOriginX),
            y: min(max(frame.origin.y, visibleScreenFrame.minY), maxOriginY),
            width: frame.width,
            height: frame.height
        )
    }

    private static func visibleScreenFrame(
        nearMenuBarItemFrame menuBarItemFrame: CGRect,
        screenFrames: [CGRect],
        visibleScreenFrames: [CGRect]
    ) -> CGRect? {
        let menuBarItemMidpoint = CGPoint(x: menuBarItemFrame.midX, y: menuBarItemFrame.midY)

        if screenFrames.count == visibleScreenFrames.count,
           let index = screenFrames.firstIndex(where: { $0.contains(menuBarItemMidpoint) }) {
            return visibleScreenFrames[index]
        }

        return visibleScreenFrames.first
    }

    private static func distanceSquared(from point: CGPoint, to rect: CGRect) -> CGFloat {
        let deltaX: CGFloat
        if point.x < rect.minX {
            deltaX = rect.minX - point.x
        } else if point.x > rect.maxX {
            deltaX = point.x - rect.maxX
        } else {
            deltaX = 0
        }

        let deltaY: CGFloat
        if point.y < rect.minY {
            deltaY = rect.minY - point.y
        } else if point.y > rect.maxY {
            deltaY = point.y - rect.maxY
        } else {
            deltaY = 0
        }

        return (deltaX * deltaX) + (deltaY * deltaY)
    }
}

@MainActor
protocol BrainBarWindowHandling: AnyObject {
    var frame: CGRect { get }
    var isVisible: Bool { get }
    func apply(frame: CGRect)
    func show()
    func hide()
}

extension NSWindow: BrainBarWindowHandling {
    func apply(frame: CGRect) {
        setFrame(frame, display: true)
    }

    func show() {
        NSApp.activate(ignoringOtherApps: true)
        makeKeyAndOrderFront(nil)
        orderFrontRegardless()
    }

    func hide() {
        orderOut(nil)
    }
}

@MainActor
final class BrainBarWindowCoordinator {
    private let frameStore: BrainBarWindowFrameStore
    private let screenFramesProvider: () -> [CGRect]
    private weak var window: BrainBarWindowHandling?

    init(
        frameStore: BrainBarWindowFrameStore = BrainBarWindowFrameStore(),
        screenFramesProvider: @escaping () -> [CGRect] = { NSScreen.screens.map(\.visibleFrame) }
    ) {
        self.frameStore = frameStore
        self.screenFramesProvider = screenFramesProvider
    }

    func attach(window: BrainBarWindowHandling) {
        self.window = window
        if let targetFrame = BrainBarWindowPlacement.resolvedFrame(
            persistedFrame: frameStore.persistedFrame(),
            screenFrames: screenFramesProvider()
        ) {
            window.apply(frame: targetFrame)
        }
    }

    func captureCurrentFrame() {
        guard let window else { return }
        frameStore.persist(frame: window.frame)
    }

    @discardableResult
    func toggleVisibility() -> Bool {
        guard let window else { return false }
        if window.isVisible {
            window.hide()
        } else {
            window.show()
        }
        return true
    }
}
