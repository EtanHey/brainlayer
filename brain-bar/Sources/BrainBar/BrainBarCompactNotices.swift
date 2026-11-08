import Foundation

/// The compact status list the menu and the Dashboard both render (Etan, 2026-11-08: "the menu-bar
/// dropdown keeps growing and is ugly. Make it compact"). Presentation only: it orders, groups and
/// caps rows, and never changes a row's text, state or count. Every row stays reachable.
struct BrainBarCompactNotices: Equatable {
    typealias Row = BrainBarStatusPopoverController.VerdictRow

    struct Entry: Equatable {
        /// A row's own text, or the cause line when several rows share one cause.
        let title: String
        let rows: [Row]
        var isGroup: Bool { rows.count > 1 }
        /// The most severe member state.
        var state: BrainBarHealthVerdict.State? {
            rows.min { BrainBarCompactNotices.rank($0) < BrainBarCompactNotices.rank($1) }?.state
        }
        var isOld: Bool { rows.allSatisfy(\.isOld) }
        /// Every member's full text, for tooltips.
        var fullText: String { rows.map(\.text).joined(separator: "\n") }
    }

    /// At most this many entries show before "+N more".
    static let visibleLimit = 3

    let entries: [Entry]

    var visible: [Entry] { Array(entries.prefix(Self.visibleLimit)) }
    var overflow: [Entry] { Array(entries.dropFirst(Self.visibleLimit)) }
    /// Every underlying row, grouped or not; grouping never lowers a count.
    var rowCount: Int { entries.reduce(0) { $0 + $1.rows.count } }
    var overflowRowCount: Int { overflow.reduce(0) { $0 + $1.rows.count } }
    var overflowTitle: String? { overflow.isEmpty ? nil : "+\(overflowRowCount) more" }

    init(rows: [Row]) {
        var seen = Set<String>()
        let unique = rows.filter { seen.insert($0.text).inserted }
        var entries: [(entry: Entry, order: Int)] = []
        let supervisor = unique.filter(Self.isSupervisorStatusGap)
        var groupPlaced = false
        for (order, row) in unique.enumerated() {
            if supervisor.count > 1, Self.isSupervisorStatusGap(row) {
                guard !groupPlaced else { continue }
                groupPlaced = true
                entries.append((Entry(title: Self.supervisorTitle(count: supervisor.count), rows: supervisor), order))
            } else {
                entries.append((Entry(title: row.text, rows: [row]), order))
            }
        }
        // Severity first; the producer's order breaks ties (sorting alone is not stable).
        self.entries = entries.sorted {
            let lhs = Self.rank($0.entry), rhs = Self.rank($1.entry)
            return lhs != rhs ? lhs < rhs : $0.order < $1.order
        }.map(\.entry)
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

    static func rank(_ entry: Entry) -> Int { entry.rows.map(rank).min() ?? 2 }

    static func supervisorTitle(count: Int) -> String {
        "launchd status unavailable — \(count) checks can't be read"
    }

    /// A current, non-red row that only says the launchd supervisor could not report a status.
    /// One unreachable supervisor produces several of these; they collapse into one cause line.
    static func isSupervisorStatusGap(_ row: Row) -> Bool {
        guard !row.isOld, row.state != .red else { return false }
        let text = row.text.lowercased()
        let names = ["launchd", "launchctl", "supervisor", "kill check"].contains { text.contains($0) }
        let gap = ["unavailable", "unknown", "unreachable"].contains { text.contains($0) }
        return names && gap
    }
}
