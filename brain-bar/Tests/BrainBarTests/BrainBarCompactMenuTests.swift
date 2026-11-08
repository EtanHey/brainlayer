import AppKit
import XCTest
@testable import BrainBar

/// The compact status list (Etan, 2026-11-08: "the menu-bar dropdown keeps growing and is ugly").
@MainActor
final class BrainBarCompactMenuTests: XCTestCase {
    /// The M1's real list on 2026-11-08, as the shared snapshot carries it.
    static let m1Checks: [BrainBarHealthVerdict.Check] = [
        .init(name: "backup_local", state: .amber, note: "newest verified snapshot is 32.9 h old"),
        .init(name: "backup_offsite", state: .amber, note: "newest Drive copy is 32.9 h old"),
        .init(name: "embedding", state: .amber, note: "backlog is 5.8 h behind"),
        .init(name: "disk", state: .unknown, note: "backup disk space unmeasured"),
        .init(name: "monitor", state: .unknown, note: "launchd status unavailable"),
    ]
    static let m1Notices = [
        "Supervisor status unavailable for watch",
        "Kill check unavailable: Watcher supervisor status unavailable",
        "Scheduled maintenance runner failed at 03:15",
        "Watcher status unknown",
    ]

    static func m1Snapshot(now: Date = Date()) -> BrainBarStatusSnapshot {
        let health = BrainBarHealthVerdict(dot: .amber, checks: m1Checks, notes: [], evaluatedAt: now)
        let issues = health.checks.map {
            BrainBarStatusIssue(id: "verdict.\($0.name)", component: .health, severity: .notice, text: $0.text)
        } + m1Notices.enumerated().map {
            BrainBarStatusIssue(id: "notice.\($0.offset)", component: .watcher, severity: .notice, text: $0.element)
        }
        return BrainBarStatusSnapshot(issues: issues, isChecking: false, healthVerdict: health)
    }

    typealias Row = BrainBarStatusPopoverController.VerdictRow

    private func m1Compact() -> BrainBarCompactNotices {
        let snapshot = Self.m1Snapshot()
        return BrainBarCompactNotices(rows: BrainBarStatusPopoverController.problemRows(
            snapshot.verdict, health: snapshot.healthVerdict))
    }

    func testRanksRedThenAmberThenOtherThenEarlierKeepingProducerOrder() {
        let compact = BrainBarCompactNotices(rows: [
            Row(text: "old", isOld: true), Row(text: "neutral A"), Row(text: "amber A", state: .amber),
            Row(text: "red", state: .red), Row(text: "neutral B"), Row(text: "amber B", state: .amber),
        ])
        XCTAssertEqual(compact.entries.map(\.text), ["red", "amber A", "amber B", "neutral A", "neutral B", "old"])
    }

    func testCapsAtThreeEntriesThenCountsEveryRemainingRow() {
        let rows = (1...5).map { Row(text: "Problem \($0)") }
        let compact = BrainBarCompactNotices(rows: rows)
        XCTAssertEqual(compact.visible.map(\.text), ["Problem 1", "Problem 2", "Problem 3"])
        XCTAssertEqual(compact.overflow.map(\.text), ["Problem 4", "Problem 5"])
        XCTAssertEqual(compact.overflowTitle, "+2 more")
        XCTAssertNil(BrainBarCompactNotices(rows: Array(rows.prefix(3))).overflowTitle)
        XCTAssertTrue(BrainBarCompactNotices(rows: []).entries.isEmpty)
    }

    /// Codex #1698 r1: "supervisor unreachable" and an independent launchctl parse failure were
    /// merged into one "launchd status unavailable" line by keyword. No producer attaches a cause
    /// two rows share, so no two rows ever merge.
    func testDifferentCausesAreNeverMerged() {
        let rows = [
            Row(text: "Supervisor status unreachable"),
            Row(text: "Kill check unavailable: launchctl print output could not be parsed"),
        ]
        let compact = BrainBarCompactNotices(rows: rows)
        XCTAssertEqual(compact.entries.map(\.text), rows.map(\.text))
        XCTAssertEqual(compact.visible.map(\.text), rows.map(\.text))
        // The M1's real list: every row keeps its own line, launchd-flavoured or not.
        let m1Rows = BrainBarStatusPopoverController.problemRows(Self.m1Snapshot().verdict,
                                                                  health: Self.m1Snapshot().healthVerdict)
        XCTAssertEqual(m1Compact().entries.count, m1Rows.count)
    }

    /// Codex #1698 r1: with three or fewer lines a group's members were reachable only through a
    /// tooltip. Without overflow, every row is its own visible line.
    func testThreeOrFewerRowsAreEachAVisibleLineWithoutOverflow() {
        let rows = [
            Row(text: "Health monitor: launchd status unavailable"),
            Row(text: "Supervisor status unavailable for watch"),
            Row(text: "Watcher status unknown"),
        ]
        let compact = BrainBarCompactNotices(rows: rows)
        XCTAssertNil(compact.overflowTitle)
        XCTAssertEqual(compact.visible.map(\.text), rows.map(\.text))
    }

    /// Codex #1698 r1: the Dashboard ranked only red vs. nothing, so unknown checks ahead of an
    /// amber one pushed it below the cap. It now ranks with the menu's states.
    func testDashboardRanksAmberAheadOfUnknownLikeTheMenu() throws {
        let checks: [BrainBarHealthVerdict.Check] = [
            .init(name: "monitor", state: .unknown, note: "launchd status unavailable"),
            .init(name: "disk", state: .unknown, note: "backup disk space unmeasured"),
            .init(name: "stores", state: .unknown, note: "store count unmeasured"),
            .init(name: "backup_local", state: .amber, note: "newest verified snapshot is 32.9 h old"),
        ]
        let health = BrainBarHealthVerdict(dot: .amber, checks: checks, notes: [], evaluatedAt: Date())
        let snapshot = BrainBarStatusSnapshot(issues: checks.map {
            BrainBarStatusIssue(id: "verdict.\($0.name)", component: .health, severity: .notice, text: $0.text)
        }, isChecking: false, healthVerdict: health)
        let verdict = snapshot.verdict
        // The Dashboard's list under a verdict dot (`derive(statusVerdict:)`): attention + notices.
        let items = verdict.attention + verdict.notices
        let dashboard = BrainBarCompactNotices(rows: BrainBarCompactNotices.dashboardRows(
            items, verdict: verdict, health: snapshot.healthVerdict))
        let amber = "Database snapshot (verified): newest verified snapshot is 32.9 h old"
        XCTAssertEqual(dashboard.visible.first?.text, amber)
        XCTAssertEqual(dashboard.visible.first?.state, .amber)
        let menu = BrainBarCompactNotices(rows: BrainBarStatusPopoverController.problemRows(
            verdict, health: snapshot.healthVerdict))
        XCTAssertEqual(dashboard.entries.map(\.text), menu.entries.map(\.text), "one list, one order")
        // Without a verdict (standalone fallback) every item keeps its own line, none red.
        XCTAssertEqual(BrainBarCompactNotices.dashboardRows(["a", "b"], verdict: nil, health: nil).map(\.state),
                       [nil, nil])
    }

    func testM1ListRendersAtMostThreeEntriesAndKeepsEveryRowReachable() {
        let snapshot = Self.m1Snapshot()
        let rows = BrainBarStatusPopoverController.problemRows(snapshot.verdict, health: snapshot.healthVerdict)
        let compact = BrainBarCompactNotices(rows: rows)
        XCTAssertEqual(rows.count, 9)
        XCTAssertLessThanOrEqual(compact.visible.count, BrainBarCompactNotices.visibleLimit)
        XCTAssertEqual(compact.visible.map(\.text), [
            "Database snapshot (verified): newest verified snapshot is 32.9 h old",
            "Offsite (Drive) database backup: newest Drive copy is 32.9 h old",
            "Embedding: backlog is 5.8 h behind",
        ])
        XCTAssertEqual(compact.overflowTitle, "+6 more")
        // Nothing hidden: every row is one line, visible or under "+N more", and the counts stay true.
        XCTAssertEqual(compact.entries.map(\.text).sorted(), rows.map(\.text).sorted())
        XCTAssertEqual(compact.visible.count + compact.overflow.count, rows.count)
        XCTAssertEqual(BrainBarStatusPopoverController.statusLineTitle(for: snapshot.verdict), "9 things need you")
        XCTAssertEqual(snapshot.verdict.dashboardStatus.headline, "9 things need you")
    }

    func testLongListStaysBoundedAtAnyLength() {
        for count in [4, 12, 40] {
            let rows = (1...count).map { Row(text: String(repeating: "very long notice \($0) ", count: 12)) }
            let compact = BrainBarCompactNotices(rows: rows)
            XCTAssertEqual(compact.visible.count, 3)
            XCTAssertEqual(compact.overflowTitle, "+\(count - 3) more")
        }
    }

    func testRealMenuShowsCappedRowsAndOverflowSubmenuWithEveryRow() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let runtime = BrainBarRuntime()
        runtime.install(collector: BrainBarDashboardFixture.makeCollector(), database: nil,
            databasePath: directory.appendingPathComponent("fixture.db").path)
        let panel = BrainBarDashboardPanelController(workspace: BrainBarRecordingWorkspace(), runtime: runtime)
        let controller = BrainBarStatusPopoverController(workspace: BrainBarRecordingWorkspace(), runtime: runtime,
            dashboardPanelController: panel, statusBar: TestStatusItemHost())
        defer { controller.stop() }
        var badge = BadgeStatePresentation(badgeOn: false, reason: "", activeCodes: [])
        badge.healthVerdict = .init(dot: .amber, checks: Self.m1Checks, notes: Self.m1Notices.enumerated().map {
            .init(code: "fixture_\($0.offset)", message: $0.element)
        }, evaluatedAt: Date())
        controller.setBadgeForTesting(badge, alertsURL: directory.appendingPathComponent("job-alerts.json"))
        let menu = controller.contextMenuForTesting
        controller.menuNeedsUpdate(menu)
        menu.update()

        XCTAssertEqual(menu.items.first?.title, "9 things need you")
        XCTAssertNotNil(menu.items.first?.image, "the headline carries the status dot")
        let overflow = try XCTUnwrap(menu.items.first { $0.title == "+6 more" })
        XCTAssertTrue(overflow.isEnabled)
        let submenu = try XCTUnwrap(overflow.submenu)
        for text in ["Watcher status unknown", "Supervisor status unavailable for watch",
                     "Kill check unavailable: Watcher supervisor status unavailable",
                     "Scheduled maintenance runner failed at 03:15"] {
            XCTAssertTrue(submenu.items.contains { $0.title == text }, "\(text) reachable under +6 more")
        }
        let open = try XCTUnwrap(submenu.items.last)
        XCTAssertEqual(open.title, "Open Dashboard")
        XCTAssertTrue(open.target === controller)
        XCTAssertNotNil(open.action)
        // The top-level menu holds at most three entries plus "+N more" between the headline and backups.
        let statusRows = menu.items.prefix { $0.title != "" && !$0.isSeparatorItem }.dropFirst()
            .filter { !$0.isHidden && !$0.title.hasPrefix("Show log") }
        XCTAssertLessThanOrEqual(statusRows.count, BrainBarCompactNotices.visibleLimit + 1, "\(statusRows.map(\.title))")
        // Reopening rebuilds, never stacks rows.
        controller.menuNeedsUpdate(menu)
        XCTAssertEqual(menu.items.filter { $0.title == "+6 more" }.count, 1)
    }

    /// Before/after evidence: `BRAINBAR_COMPACT_RENDER_DIR=<dir> swift test --filter BrainBarCompactMenuTests`.
    func testRenderM1MenuAndDashboard() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_COMPACT_RENDER_DIR"] else {
            throw XCTSkip("Set BRAINBAR_COMPACT_RENDER_DIR for isolated render evidence")
        }
        let directory = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let snapshot = Self.m1Snapshot()
        let backups = [
            ObservabilityStatusLine(text: "Database snapshot · 32.9 h ago", tone: .amber),
            ObservabilityStatusLine(text: "Transcripts · Drive · 1 h ago", tone: .green),
        ]
        try BrainBarRenderHarness.renderHealthVerdictPopover(snapshot, backupLines: backups,
                                                            directory: directory, name: "m1-menu")
        try BrainBarRenderHarness.renderHealthVerdictPopover(snapshot, backupLines: backups,
            directory: directory, name: "m1-menu-more-open", showsOverflow: true)
        try BrainBarRenderHarness.renderMenuCoherenceDashboard(snapshot,
            result: BrainBarDashboardFixture.healthyObservabilityResult, now: BrainBarDashboardFixture.fetchedAt,
            directory: directory, name: "m1-dashboard", width: 960, calendar: .current)
        try BrainBarRenderHarness.renderMenuCoherenceDashboard(snapshot,
            result: BrainBarDashboardFixture.healthyObservabilityResult, now: BrainBarDashboardFixture.fetchedAt,
            directory: directory, name: "m1-dashboard-more-open", width: 960, calendar: .current,
            showsAllAttention: true)
    }
}
