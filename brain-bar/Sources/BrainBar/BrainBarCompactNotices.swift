import Foundation

/// The compact status list the menu and the Dashboard both render (Etan, 2026-11-08: "the menu-bar
/// dropdown keeps growing and is ugly. Make it compact"). Presentation only: it orders and caps
/// rows, and never changes a row's text, state or count. Every row stays reachable.
///
/// Rows are never merged. Grouping needs a cause two rows share, and no producer attaches one:
/// every `BrainBarStatusIssue.id` is unique per row (`Builder.add` folds equal ids), and
/// `component` names an area, not a cause. Matching words in the text merged unrelated faults
/// (Codex #1698 r1), so each row keeps its own line.
struct BrainBarCompactNotices: Equatable {
    typealias Row = BrainBarStatusPopoverController.VerdictRow

    /// At most this many rows show before "+N more".
    static let visibleLimit = 3

    /// Every distinct row, most severe first.
    let entries: [Row]

    var visible: [Row] { Array(entries.prefix(Self.visibleLimit)) }
    var overflow: [Row] { Array(entries.dropFirst(Self.visibleLimit)) }
    var overflowTitle: String? { overflow.isEmpty ? nil : "+\(overflow.count) more" }

    init(rows: [Row]) {
        var seen = Set<String>()
        let unique = rows.filter { seen.insert($0.text).inserted }
        // Severity first; the producer's order breaks ties (sorting alone is not stable).
        entries = unique.enumerated().sorted {
            let lhs = Self.rank($0.element), rhs = Self.rank($1.element)
            return lhs != rhs ? lhs < rhs : $0.offset < $1.offset
        }.map(\.element)
    }

    /// red, then amber, then other current rows, then earlier (historical) rows.
    static func rank(_ row: Row) -> Int {
        if row.isOld { return 3 }
        return switch row.state {
        case .red?: 0
        case .amber?: 1
        default: 2
        }
    }

    /// The Dashboard's attention items with the severity the menu gives the same rows: the shared
    /// `problemRows` states (red, amber, unknown), so both surfaces rank one list one way. An item
    /// the verdict does not carry (the standalone fallback) is red only when it is a dot issue.
    static func dashboardRows(_ items: [String], verdict: BrainBarStatusVerdict?,
                              health: BrainBarHealthVerdict?) -> [Row] {
        let shared = verdict.map { BrainBarStatusPopoverController.problemRows($0, health: health) } ?? []
        let byText = Dictionary(shared.map { ($0.text, $0) }, uniquingKeysWith: { first, _ in first })
        return items.map { item in
            byText[item] ?? Row(text: item, state: verdict?.attention.contains(item) == true ? .red : nil)
        }
    }
}
