import Foundation

enum BrainBarURLAction: Equatable {
    case toggle
    case search

    static func parse(url: URL) -> BrainBarURLAction? {
        guard url.scheme?.lowercased() == "brainbar" else { return nil }

        let host = url.host?.lowercased()
        let path = url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/")).lowercased()
        let route = host?.isEmpty == false ? host! : path

        switch route {
        case "toggle":
            return .toggle
        case "search":
            return .search
        default:
            return nil
        }
    }
}

final class HotkeyRouteStatus: ObservableObject {
    @Published private(set) var statusLine: String = "Karabiner URL route active"
    @Published private(set) var useCGEventTapFallback: Bool = false

    var onFallbackChange: (() -> Void)?

    func setUseCGEventTapFallback(_ enabled: Bool) {
        guard useCGEventTapFallback != enabled else { return }
        useCGEventTapFallback = enabled
        refreshStatusLine(eventTapActive: false)
        onFallbackChange?()
    }

    func refreshStatusLine(eventTapActive: Bool) {
        if useCGEventTapFallback {
            statusLine = eventTapActive ? "CGEventTap fallback active" : "CGEventTap fallback idle"
        } else {
            statusLine = "Karabiner URL route active"
        }
    }
}
