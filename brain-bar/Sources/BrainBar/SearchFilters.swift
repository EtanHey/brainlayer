import Foundation

enum SearchFilterChip: CaseIterable, Equatable, Sendable {
    case all
    case important
    case unread

    var title: String {
        switch self {
        case .all:
            return "All"
        case .important:
            return "Important"
        case .unread:
            return "Unread"
        }
    }
}

struct SearchFilters: Equatable, Sendable {
    var primaryChip: SearchFilterChip = .all

    var importanceMin: Double? {
        primaryChip == .important ? 7 : nil
    }

    var unreadOnly: Bool {
        primaryChip == .unread
    }

    func selecting(_ chip: SearchFilterChip) -> SearchFilters {
        var copy = self
        copy.primaryChip = chip
        return copy
    }
}
