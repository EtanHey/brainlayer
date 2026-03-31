import Foundation

enum BrainBarURLAction: Equatable {
    case toggle
    case search

    static func parse(url: URL) -> BrainBarURLAction? {
        guard url.scheme?.lowercased() == "brainbar" else { return nil }

        let target = url.host?.lowercased() ?? url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/")).lowercased()
        switch target {
        case "toggle":
            return .toggle
        case "search":
            return .search
        default:
            return nil
        }
    }
}
