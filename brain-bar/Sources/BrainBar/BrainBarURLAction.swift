import Foundation

enum BrainBarURLAction: Equatable {
    case toggle
#if BRAINBAR_UI
    case dashboard
    case settings(BrainBarSettingsSection)
#endif

    static func parse(url: URL) -> BrainBarURLAction? {
        guard url.scheme?.lowercased() == "brainbar" else { return nil }

        let path = url.path.split(separator: "/").map { $0.lowercased() }
        let host = url.host?.lowercased()
        let hasHost = host?.isEmpty == false
        let target = hasHost ? host! : (path.first ?? "")
        let remainder = hasHost ? path : Array(path.dropFirst())
        switch target {
        case "toggle":
            return .toggle
#if BRAINBAR_UI
        case "dashboard" where remainder.isEmpty:
            return .dashboard
        case "settings" where remainder.count <= 1:
            // General was absorbed into Dashboard (#963); an old link to it opens Dashboard.
            if remainder.first == "general" { return .dashboard }
            return .settings(remainder.first.flatMap { BrainBarSettingsSection(rawValue: $0) } ?? .jobs)
#endif
        default:
            return nil
        }
    }
}
