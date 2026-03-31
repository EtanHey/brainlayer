import Combine
import Foundation

@MainActor
final class HotkeyRouteStatus: ObservableObject {
    @Published private(set) var statusLine = "Hotkey route: brainbar:// URLs"
    @Published var useCGEventTapFallback = false {
        didSet {
            guard oldValue != useCGEventTapFallback else { return }
            onFallbackChange?()
        }
    }

    var onFallbackChange: (() -> Void)?

    func refreshStatusLine(eventTapActive: Bool) {
        if useCGEventTapFallback {
            statusLine = eventTapActive
                ? "Hotkey route: CGEventTap fallback active"
                : "Hotkey route: CGEventTap fallback unavailable"
        } else {
            statusLine = "Hotkey route: brainbar:// URLs"
        }
    }
}
