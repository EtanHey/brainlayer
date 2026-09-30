import Darwin
import Foundation
import XCTest

@testable import BrainBar

/// #972: the Runtime rows read "Daemon: Unavailable / Last seen: Unavailable" after
/// any daemon restart, because the monitor watched a PID captured once at app launch.
///
/// The daemon is identified as the PRODUCTION service only: the process listening on
/// the BrainBar socket whose executable is inside the running UI's own `BrainBar.app` (#980),
/// else the installed `/Applications/BrainBar.app` copy. A basename match, a DEV/scratch build,
/// another bundle's daemon, or a reused PID is never proof. Every
/// process fact and the clock are injected, so no test spawns or signals a process.
final class DaemonHealthMonitorRestartTests: XCTestCase {
    private static let socket = "/tmp/brainbar.sock"
    private static let installed = "/Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon"
    fileprivate static let installedPath = installed
    private static let devBuild = "/Users/dev/Gits/brainlayer/brain-bar/.build/debug/BrainBarDaemon"
    private static let scratchBuild = "/tmp/build/BrainBarDaemon"
    private static let selfBuilt = "/Users/u/Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon"
    private static let now = Date(timeIntervalSince1970: 1_000_000)

    private func monitor(_ table: FakeProcessTable) -> DaemonHealthMonitor {
        DaemonHealthMonitor(inspector: table, socketPath: Self.socket, now: { Self.now })
    }

    func test_restart_PID_A_dies_then_PID_B_keeps_the_daemon_populated() throws {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now.addingTimeInterval(-3_900)),
        ])
        let monitor = monitor(table)

        let first = try XCTUnwrap(monitor.read().snapshot)
        XCTAssertEqual(first.pid, 101)
        XCTAssertEqual(first.uptime, 3_900)

        table.remove(101)
        table.add(202, .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now.addingTimeInterval(-60)))

        let reading = monitor.read()
        let second = try XCTUnwrap(reading.snapshot, "a restart must not blank the row: \(reading)")
        XCTAssertEqual(second.pid, 202)
        XCTAssertEqual(second.uptime, 60)
        XCTAssertEqual(second.startedAt, Self.now.addingTimeInterval(-60))
        XCTAssertNil(reading.unavailability)
    }

    func test_a_live_verified_PID_is_not_rescanned_every_sample() {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now),
        ])
        let monitor = monitor(table)

        _ = monitor.read()
        _ = monitor.read()
        _ = monitor.read()

        XCTAssertEqual(table.scanCount, 1, "a verified cached PID must not trigger a process-table scan")
    }

    func test_a_reused_PID_is_re_verified_and_not_trusted() throws {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now),
        ])
        let monitor = monitor(table)
        XCTAssertEqual(monitor.read().snapshot?.pid, 101)

        // PID 101 is reused by an unrelated process; the real daemon came back as 303.
        table.remove(101)
        table.add(101, .other(shortName: "node", path: "/usr/local/bin/node"))
        table.add(303, .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now))

        XCTAssertEqual(try XCTUnwrap(monitor.read().snapshot).pid, 303)
    }

    func test_an_installed_copy_that_is_not_serving_the_socket_is_not_the_daemon() throws {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: nil, startedAt: Self.now),
        ])

        let reading = monitor(table).read()

        XCTAssertNil(reading.snapshot)
        XCTAssertEqual(reading.unavailability, .down("BrainBarDaemon is running but not serving /tmp/brainbar.sock"))
    }

    func test_a_dev_build_serving_the_socket_is_named_not_shown_as_the_daemon() throws {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: nil, startedAt: Self.now),
            404: .daemon(path: Self.devBuild, listening: Self.socket, startedAt: Self.now),
        ])

        let reading = monitor(table).read()

        XCTAssertNil(reading.snapshot, "a DEV build must never populate the installed Runtime row")
        XCTAssertEqual(
            reading.unavailability,
            .down("/tmp/brainbar.sock is served by a non-installed BrainBarDaemon (\(Self.devBuild))")
        )
    }

    func test_scratch_and_dev_copies_by_basename_alone_are_ignored() throws {
        let table = FakeProcessTable([
            501: .daemon(path: Self.scratchBuild, listening: nil, startedAt: Self.now),
            502: .daemon(path: Self.devBuild, listening: "/tmp/brainbar-dev.sock", startedAt: Self.now),
        ])

        let reading = monitor(table).read()

        XCTAssertNil(reading.snapshot)
        XCTAssertEqual(reading.unavailability, .down("BrainBarDaemon not running"))
    }

    func test_daemon_down_after_exit_is_a_reasoned_attention_state() throws {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now),
        ])
        let monitor = monitor(table)
        XCTAssertNotNil(monitor.read().snapshot)

        table.remove(101)

        let reading = monitor.read()
        XCTAssertNil(reading.snapshot)
        let unavailability = try XCTUnwrap(reading.unavailability)
        XCTAssertEqual(unavailability, .down("BrainBarDaemon not running (last PID 101 exited)"))

        let row = DaemonRuntimeRows.daemonText(daemon: nil, unavailability: unavailability)
        XCTAssertEqual(row, "Down — BrainBarDaemon not running (last PID 101 exited)")
        XCTAssertTrue(DaemonRuntimeRows.isAttention(row), "a real outage keeps the attention colour")
    }

    /// #976 R2 B3: when the monitor cannot measure (process table or the daemon's
    /// sockets unreadable) there is no confirmed outage, so the visible row is a
    /// neutral "Unknown — …", never "Down — …".
    @MainActor
    func test_an_unreadable_process_table_shows_a_neutral_unknown_row() async throws {
        let table = FakeProcessTable([:])
        table.isReadable = false
        XCTAssertEqual(monitor(table).read().unavailability, .unknown("process table unreadable"))

        let row = try await visibleDaemonRow(table)

        XCTAssertEqual(row, "Unknown — process table unreadable")
        XCTAssertFalse(DaemonRuntimeRows.isAttention(row), "an unmeasured state must not wear the outage colour")
    }

    @MainActor
    func test_an_unreadable_socket_table_on_the_installed_daemon_is_unknown_not_down() async throws {
        let table = FakeProcessTable([
            101: .daemonSocketsUnreadable(path: Self.installed, startedAt: Self.now),
        ])
        XCTAssertEqual(monitor(table).read().unavailability, .unknown("sockets of BrainBarDaemon PID 101 unreadable"))

        let row = try await visibleDaemonRow(table)

        XCTAssertEqual(row, "Unknown — sockets of BrainBarDaemon PID 101 unreadable")
        XCTAssertFalse(DaemonRuntimeRows.isAttention(row))
    }

    func test_a_verified_daemon_whose_sockets_become_unreadable_is_unknown_not_exited() {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now),
        ])
        let monitor = monitor(table)
        XCTAssertEqual(monitor.read().snapshot?.pid, 101)

        table.add(101, .daemonSocketsUnreadable(path: Self.installed, startedAt: Self.now))

        XCTAssertEqual(monitor.read().unavailability, .unknown("sockets of BrainBarDaemon PID 101 unreadable"))
    }

    /// #978: every inspection failure is "Unknown"; "Down" needs a fully measured table.
    @MainActor
    func test_an_unreadable_executable_path_on_a_daemon_candidate_is_unknown() async throws {
        let table = FakeProcessTable([
            101: .daemonPathUnreadable(listening: Self.socket, startedAt: Self.now),
        ])

        let row = try await visibleDaemonRow(table)

        XCTAssertEqual(row, "Unknown — executable path of BrainBarDaemon PID 101 unreadable")
        XCTAssertFalse(DaemonRuntimeRows.isAttention(row))
    }

    @MainActor
    func test_an_unreadable_process_with_no_daemon_found_is_unknown_not_down() async throws {
        let table = FakeProcessTable([
            77: .unreadable,
            78: .other(shortName: "node", path: "/usr/local/bin/node"),
        ])

        let row = try await visibleDaemonRow(table)

        XCTAssertEqual(row, "Unknown — process PID 77 unreadable")
        XCTAssertFalse(DaemonRuntimeRows.isAttention(row))
    }

    @MainActor
    func test_a_zombie_cannot_be_the_daemon_so_a_fully_measured_table_is_down() async throws {
        let table = FakeProcessTable([
            77: .zombie,
            78: .other(shortName: "node", path: "/usr/local/bin/node"),
        ])

        let row = try await visibleDaemonRow(table)

        XCTAssertEqual(row, "Down — BrainBarDaemon not running")
        XCTAssertTrue(DaemonRuntimeRows.isAttention(row))
    }

    func test_the_production_daemon_wins_over_an_unrelated_unreadable_process() {
        let table = FakeProcessTable([
            77: .unreadable,
            101: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now),
        ])

        XCTAssertEqual(monitor(table).read().snapshot?.pid, 101)
    }

    /// The live inspector's per-descriptor verdict: a positive match is proof; with no
    /// match, any unreadable descriptor leaves socket ownership unknown (nil).
    func test_a_failed_descriptor_read_leaves_socket_ownership_unknown() {
        typealias Read = LiveDaemonProcessInspector.SocketRead
        let socket = Self.socket
        XCTAssertNil(LiveDaemonProcessInspector.listeningVerdict([.unreadable, .other], socketPath: socket))
        XCTAssertEqual(
            LiveDaemonProcessInspector.listeningVerdict([.unreadable, .listening("/private/tmp/brainbar.sock")], socketPath: socket),
            true
        )
        XCTAssertEqual(LiveDaemonProcessInspector.listeningVerdict([.other, .listening("/tmp/other.sock")], socketPath: socket), false)
        XCTAssertEqual(LiveDaemonProcessInspector.listeningVerdict([Read](), socketPath: socket), false)
    }

    /// #976 Macroscope (b): a daemon verified as serving whose metrics cannot be read
    /// is an inspection failure, so the row is neutral "Unknown", not "Unavailable".
    @MainActor
    func test_a_serving_daemon_with_unreadable_process_info_is_unknown() async throws {
        let table = FakeProcessTable([
            101: .daemonInfoUnreadable(listening: Self.socket),
        ])

        let row = try await visibleDaemonRow(table)

        XCTAssertEqual(row, "Unknown — process info of BrainBarDaemon PID 101 unreadable")
        XCTAssertFalse(DaemonRuntimeRows.isAttention(row))
    }

    /// #976 Macroscope (c): "(last PID n exited)" only when that PID is really gone.
    func test_a_cached_daemon_that_stops_serving_but_is_alive_is_not_called_exited() throws {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now),
        ])
        let monitor = monitor(table)
        XCTAssertEqual(monitor.read().snapshot?.pid, 101)

        table.add(101, .daemon(path: Self.installed, listening: nil, startedAt: Self.now))

        XCTAssertEqual(
            monitor.read().unavailability,
            .down("BrainBarDaemon is running but not serving /tmp/brainbar.sock")
        )
    }

    /// #979 R1 B1: one sample reads the daemon's metrics once; a read that fails
    /// between two calls must not surface as an "Unavailable" snapshot.
    func test_one_sample_reads_process_info_once_and_a_later_failure_is_unknown() {
        let table = FakeProcessTable([
            101: .daemonInfoFlaky(listening: Self.socket, startedAt: Self.now.addingTimeInterval(-60)),
        ])
        let monitor = monitor(table)

        let first = monitor.read()
        XCTAssertEqual(first.snapshot?.pid, 101)
        XCTAssertEqual(first.snapshot?.uptime, 60)
        XCTAssertEqual(table.processInfoReads, 1, "one sample, one metrics read")

        let second = monitor.read()
        XCTAssertNil(second.snapshot)
        XCTAssertEqual(second.unavailability, .unknown("process info of BrainBarDaemon PID 101 unreadable"))
        XCTAssertEqual(
            DaemonRuntimeRows.daemonText(daemon: second.snapshot, unavailability: second.unavailability),
            "Unknown — process info of BrainBarDaemon PID 101 unreadable"
        )
    }

    /// #979 R1 B2: a non-installed listener proves nothing while a production
    /// candidate is unmeasured; the row stays neutral "Unknown".
    @MainActor
    func test_a_scratch_listener_does_not_preempt_an_unmeasured_production_candidate() async throws {
        let cases: [(FakeProcessTable.Entry, pid_t, String)] = [
            (.daemonSocketsUnreadable(path: Self.installed, startedAt: Self.now), 101,
             "Unknown — sockets of BrainBarDaemon PID 101 unreadable"),
            (.daemonPathUnreadable(listening: nil, startedAt: Self.now), 101,
             "Unknown — executable path of BrainBarDaemon PID 101 unreadable"),
            (.unreadable, 77, "Unknown — process PID 77 unreadable"),
        ]
        for (entry, pid, expected) in cases {
            let table = FakeProcessTable([
                pid: entry,
                501: .daemon(path: Self.scratchBuild, listening: Self.socket, startedAt: Self.now),
            ])

            let row = try await visibleDaemonRow(table)

            XCTAssertEqual(row, expected)
            XCTAssertFalse(DaemonRuntimeRows.isAttention(row))
        }
    }

    /// #979 R1 B3: a descriptor table that cannot be counted is not "0 sockets".
    func test_an_uncountable_socket_table_is_unmeasured_not_zero() {
        let start = Self.now
        XCTAssertNil(LiveDaemonProcessInspector.processInfo(rssBytes: 1, startedAt: start, openSockets: nil))
        XCTAssertEqual(
            LiveDaemonProcessInspector.processInfo(rssBytes: 1, startedAt: start, openSockets: 0),
            DaemonProcessInfo(rssBytes: 1, startedAt: start, openSockets: 0)
        )
    }

    /// The Daemon row exactly as the Runtime card renders it for this collector.
    @MainActor
    private func visibleDaemonRow(_ table: FakeProcessTable) async throws -> String {
        let collector = StatsCollector(
            dbPath: "/nonexistent/brainbar-976-r3.db",
            daemonMonitor: monitor(table),
            autoRefreshInterval: 3600,
            nowProvider: { Self.now },
            dashboardStatsProvider: { throw StubStatsUnavailable() }
        )
        defer { collector.stop() }
        collector.refresh(force: true)
        try await waitUntil { collector.daemonUnavailability != nil }
        return DaemonRuntimeRows.daemonText(collector: collector)
    }

    func test_installed_executable_check_is_the_app_bundle_not_the_basename() {
        let identity = DaemonIdentity(socketPath: Self.socket, expectedExecutablePath: Self.installed)
        XCTAssertTrue(identity.isExpectedExecutable(Self.installed))
        XCTAssertFalse(identity.isExpectedExecutable(Self.devBuild))
        XCTAssertFalse(identity.isExpectedExecutable(Self.scratchBuild))
        XCTAssertFalse(identity.isExpectedExecutable(Self.selfBuilt))
        XCTAssertFalse(identity.isExpectedExecutable("/Applications/BrainBar.app/Contents/MacOS/BrainBar"))
        XCTAssertFalse(identity.isExpectedExecutable("/Applications/BrainBar Dev.app/Contents/MacOS/BrainBarDaemon"))
    }

    // MARK: #980: the daemon inside the running UI's own bundle

    /// `build-app.sh` installs a self-built app at ~/Applications/BrainBar.app. That UI's own
    /// daemon is production; the one in /Applications is then foreign. DEV bundles
    /// (`BrainBar-DEV-<branch>.app`) and non-bundle runs still accept only /Applications.
    func test_the_expected_daemon_is_the_one_in_the_running_uis_own_bundle() {
        func expected(_ bundle: String) -> String {
            DaemonIdentity.expectedExecutablePath(forAppAt: URL(fileURLWithPath: bundle))
        }
        XCTAssertEqual(expected("/Applications/BrainBar.app"), Self.installed)
        XCTAssertEqual(expected("/Users/u/Applications/BrainBar.app"), Self.selfBuilt)
        XCTAssertEqual(expected("/Users/u/Applications/BrainBar-DEV-wt-foo.app"), Self.installed, "a DEV bundle is never production")
        XCTAssertEqual(expected("/Users/u/Gits/brainlayer/brain-bar/.build/debug"), Self.installed, "a non-bundle run")
        XCTAssertEqual(expected("/Applications/Xcode.app/Contents/Developer/usr/bin"), Self.installed)
    }

    func test_a_symlinked_ui_bundle_resolves_to_its_real_daemon_path() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let real = root.appendingPathComponent("real/BrainBar.app")
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let link = root.appendingPathComponent("link.app")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        // proc_pidpath reports the real path, so the expectation must be the real path too.
        XCTAssertEqual(
            DaemonIdentity.expectedExecutablePath(forAppAt: link),
            real.resolvingSymlinksInPath().appendingPathComponent("Contents/MacOS/BrainBarDaemon").path
        )
    }

    func test_a_self_built_ui_accepts_its_own_daemon_and_rejects_the_applications_one() throws {
        let own = FakeProcessTable([
            101: .daemon(path: Self.selfBuilt, listening: Self.socket, startedAt: Self.now.addingTimeInterval(-60)),
        ])
        let selfBuiltUI = DaemonHealthMonitor(inspector: own, socketPath: Self.socket, expectedDaemonPath: Self.selfBuilt, now: { Self.now })
        XCTAssertEqual(try XCTUnwrap(selfBuiltUI.read().snapshot).pid, 101)

        let foreign = FakeProcessTable([
            202: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now.addingTimeInterval(-60)),
        ])
        let reading = DaemonHealthMonitor(inspector: foreign, socketPath: Self.socket, expectedDaemonPath: Self.selfBuilt, now: { Self.now }).read()
        XCTAssertNil(reading.snapshot, "another bundle's daemon is not this UI's production daemon")
        XCTAssertEqual(reading.unavailability, .down("\(Self.socket) is served by a non-installed BrainBarDaemon (\(Self.installed))"))
    }

    func test_socket_paths_match_through_the_tmp_symlink() {
        XCTAssertTrue(LiveDaemonProcessInspector.socketPathsMatch("/private/tmp/brainbar.sock", "/tmp/brainbar.sock"))
        XCTAssertTrue(LiveDaemonProcessInspector.socketPathsMatch("/tmp/brainbar.sock", "/tmp/brainbar.sock"))
        XCTAssertFalse(LiveDaemonProcessInspector.socketPathsMatch("/tmp/brainbar-dev.sock", "/tmp/brainbar.sock"))
    }

    func test_daemon_row_names_PID_uptime_and_sockets() {
        let snapshot = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 0,
            uptime: 2 * 3600 + 5 * 60,
            openConnections: 3,
            lastSeenAt: Self.now,
            startedAt: Self.now.addingTimeInterval(-(2 * 3600 + 5 * 60))
        )

        let row = DaemonRuntimeRows.daemonText(daemon: snapshot, unavailability: nil)

        XCTAssertEqual(row, "PID 4242 · up 2h 5m · 3 sockets")
        XCTAssertFalse(DaemonRuntimeRows.isAttention(row))
    }

    func test_last_seen_is_the_last_socket_answer_not_the_sample_time() {
        XCTAssertEqual(
            DaemonRuntimeRows.lastSeenText(lastAnswerAt: Self.now.addingTimeInterval(-5 * 60), now: Self.now),
            "5m ago"
        )
        XCTAssertEqual(DaemonRuntimeRows.lastSeenText(lastAnswerAt: nil, now: Self.now), "No socket answer yet")
    }

    func test_before_the_first_sample_the_daemon_row_is_checking_not_down() {
        let row = DaemonRuntimeRows.daemonText(daemon: nil, unavailability: nil)
        XCTAssertEqual(row, "Checking…")
        XCTAssertFalse(DaemonRuntimeRows.isAttention(row))
    }

    @MainActor
    func test_collector_survives_a_restart_and_stamps_last_seen_from_the_brain_bus() async throws {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now),
        ])
        let bus = ScriptedBrainBus()
        let collector = StatsCollector(
            dbPath: "/nonexistent/brainbar-972.db",
            daemonMonitor: monitor(table),
            autoRefreshInterval: 3600,
            nowProvider: { Self.now },
            brainBusEvents: bus,
            dashboardStatsProvider: { throw StubStatsUnavailable() }
        )
        defer { collector.stop() }

        collector.start()
        try await waitUntil { collector.daemon?.pid == 101 }

        table.remove(101)
        table.add(202, .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now))
        let tickAt = Self.now.addingTimeInterval(-3)
        bus.publish(BrainBusEvent.healthTick(openConnections: 2).withSequence(7, generatedAt: tickAt))

        try await waitUntil { collector.daemon?.pid == 202 }
        let snapshot = try XCTUnwrap(collector.daemon)
        XCTAssertEqual(snapshot.lastSeenAt, tickAt, "Last seen is the socket answer's time")
        XCTAssertEqual(collector.lastDaemonAnswerAt, tickAt)
        XCTAssertNil(collector.daemonUnavailability)
        XCTAssertFalse(table.scannedOnMainThread, "daemon resolution must stay off the main thread")
    }

    @MainActor
    func testMakeUIStatsCollectorResolvesTheDaemonOnEverySampleNotOnceAtLaunch() async throws {
        let table = FakeProcessTable([
            101: .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now),
        ])
        let collector = BrainBarAppSupport.makeUIStatsCollector(
            dbPath: "/nonexistent/brainbar-972-ui.db",
            brainBusEvents: nil,
            daemonMonitor: monitor(table)
        )
        defer { collector.stop() }

        collector.refresh(force: true)
        try await waitUntil { collector.daemon?.pid == 101 }

        table.remove(101)
        table.add(202, .daemon(path: Self.installed, listening: Self.socket, startedAt: Self.now))
        collector.refresh(force: true)

        try await waitUntil { collector.daemon?.pid == 202 }
    }

    @MainActor
    private func waitUntil(
        timeout: TimeInterval = 3.0,
        _ predicate: () -> Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !predicate(), Date() < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        XCTAssertTrue(predicate(), "condition not met within \(timeout)s", file: file, line: line)
    }
}

/// A scripted process table: every fact the monitor reads comes from here.
private final class FakeProcessTable: DaemonProcessInspecting, @unchecked Sendable {
    enum Entry {
        case daemon(path: String, listening: String?, startedAt: Date)
        /// A daemon whose file-descriptor table cannot be read.
        case daemonSocketsUnreadable(path: String, startedAt: Date)
        /// A BrainBarDaemon whose executable path cannot be read.
        case daemonPathUnreadable(listening: String?, startedAt: Date)
        /// A serving installed daemon whose task/BSD info cannot be read.
        case daemonInfoUnreadable(listening: String?)
        /// A serving installed daemon whose metrics read once, then fail.
        case daemonInfoFlaky(listening: String?, startedAt: Date)
        /// A process whose name and state cannot be read at all.
        case unreadable
        /// An exited, unreaped process: it holds no sockets and is never the daemon.
        case zombie
        case other(shortName: String, path: String)
    }

    private let lock = NSLock()
    private var entries: [pid_t: Entry]
    private var _isReadable = true
    private var _scanCount = 0
    private var _scannedOnMainThread = false
    private var _processInfoReads = 0

    init(_ entries: [pid_t: Entry]) {
        self.entries = entries
    }

    var isReadable: Bool {
        get { lock.withLock { _isReadable } }
        set { lock.withLock { _isReadable = newValue } }
    }

    var scanCount: Int { lock.withLock { _scanCount } }
    var scannedOnMainThread: Bool { lock.withLock { _scannedOnMainThread } }
    var processInfoReads: Int { lock.withLock { _processInfoReads } }

    func add(_ pid: pid_t, _ entry: Entry) { lock.withLock { entries[pid] = entry } }
    func remove(_ pid: pid_t) { lock.withLock { _ = entries.removeValue(forKey: pid) } }

    func processIDs() -> [pid_t]? {
        lock.withLock {
            _scanCount += 1
            if Thread.isMainThread { _scannedOnMainThread = true }
            return _isReadable ? entries.keys.sorted() : nil
        }
    }

    func shortName(of pid: pid_t) -> ProcessNameRead {
        lock.withLock {
            switch entries[pid] {
            case .daemon?, .daemonSocketsUnreadable?, .daemonPathUnreadable?, .daemonInfoUnreadable?, .daemonInfoFlaky?:
                return .name("BrainBarDaemon")
            case .other(let name, _)?: return .name(name)
            case .zombie?: return .zombie
            case .unreadable?, nil: return .unreadable
            }
        }
    }

    func executablePath(of pid: pid_t) -> String? {
        lock.withLock {
            switch entries[pid] {
            case .daemon(let path, _, _)?: return path
            case .daemonSocketsUnreadable(let path, _)?: return path
            case .daemonInfoUnreadable?, .daemonInfoFlaky?: return DaemonHealthMonitorRestartTests.installedPath
            case .other(_, let path)?: return path
            case .daemonPathUnreadable?, .unreadable?, .zombie?, nil: return nil
            }
        }
    }

    func isListening(_ pid: pid_t, onUnixSocket path: String) -> Bool? {
        lock.withLock {
            switch entries[pid] {
            case .daemon(_, let listening, _)?: return listening == path
            case .daemonPathUnreadable(let listening, _)?, .daemonInfoUnreadable(let listening)?,
                 .daemonInfoFlaky(let listening, _)?:
                return listening == path
            case .daemonSocketsUnreadable?, .unreadable?: return nil
            case .other?, .zombie?, nil: return false
            }
        }
    }

    func processInfo(of pid: pid_t) -> DaemonProcessInfo? {
        lock.withLock {
            _processInfoReads += 1
            let startedAt: Date
            switch entries[pid] {
            case .daemonInfoFlaky(_, let started)? where _processInfoReads == 1:
                startedAt = started
            case .daemon(_, _, let started)?, .daemonSocketsUnreadable(_, let started)?,
                 .daemonPathUnreadable(_, let started)?:
                startedAt = started
            case .daemonInfoUnreadable?, .daemonInfoFlaky?, .other?, .unreadable?, .zombie?, nil: return nil
            }
            return DaemonProcessInfo(rssBytes: 1_024, startedAt: startedAt, openSockets: 3)
        }
    }
}

private struct StubStatsUnavailable: Error {}

private final class ScriptedBrainBus: BrainBusEventSource, @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: AsyncStream<BrainBusEvent>.Continuation?

    func events() -> AsyncStream<BrainBusEvent> {
        AsyncStream { continuation in
            lock.withLock { self.continuation = continuation }
        }
    }

    func publish(_ event: BrainBusEvent) {
        lock.withLock { continuation }?.yield(event)
    }
}
