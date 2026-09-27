import Darwin
import Foundation
import XCTest

@testable import BrainBar

/// #972: the Runtime rows read "Daemon: Unavailable / Last seen: Unavailable" after
/// any daemon restart, because the monitor watched a PID captured once at app launch.
/// These tests pin the replacement contract: the PID is resolved live on every sample,
/// off the main thread; "Last seen" is the last socket answer, not the sample time;
/// and a daemon that is really down still reads as a reasoned attention state.
final class DaemonHealthMonitorRestartTests: XCTestCase {
    private var children: [Process] = []

    override func tearDown() {
        for child in children where child.isRunning {
            child.terminate()
            child.waitUntilExit()
        }
        children.removeAll()
        super.tearDown()
    }

    func test_restart_PID_A_dies_then_PID_B_keeps_the_daemon_populated() throws {
        let daemonA = try spawnStandInDaemon()
        let daemonB = try spawnStandInDaemon()
        let resolver = ScriptedDaemonPIDResolver(current: daemonA.processIdentifier)
        let monitor = DaemonHealthMonitor(pidResolver: resolver)

        let first = try XCTUnwrap(monitor.read().snapshot, "PID A is running")
        XCTAssertEqual(first.pid, daemonA.processIdentifier)
        XCTAssertTrue(first.isResponsive)

        daemonA.terminate()
        daemonA.waitUntilExit()
        resolver.current = daemonB.processIdentifier

        let reading = monitor.read()
        let second = try XCTUnwrap(reading.snapshot, "a restart must not blank the row: \(reading)")
        XCTAssertEqual(second.pid, daemonB.processIdentifier)
        XCTAssertTrue(second.isResponsive)
        XCTAssertNotNil(second.startedAt, "the row shows the daemon's real start time")
        XCTAssertNil(reading.downReason)
    }

    func test_monitor_keeps_the_cached_PID_without_a_fresh_lookup_while_it_is_alive() throws {
        let daemon = try spawnStandInDaemon()
        let resolver = ScriptedDaemonPIDResolver(current: daemon.processIdentifier)
        let monitor = DaemonHealthMonitor(pidResolver: resolver)

        _ = monitor.read()
        _ = monitor.read()
        _ = monitor.read()

        XCTAssertEqual(resolver.resolveCount, 1, "a live cached PID must not trigger a process scan every sample")
    }

    func test_daemon_down_is_a_reasoned_attention_state_not_a_bare_unavailable() throws {
        let daemon = try spawnStandInDaemon()
        let resolver = ScriptedDaemonPIDResolver(current: daemon.processIdentifier)
        let monitor = DaemonHealthMonitor(pidResolver: resolver)
        XCTAssertNotNil(monitor.read().snapshot)

        daemon.terminate()
        daemon.waitUntilExit()
        resolver.current = nil

        let reading = monitor.read()
        XCTAssertNil(reading.snapshot)
        let reason = try XCTUnwrap(reading.downReason)
        XCTAssertTrue(reason.contains("not running"), reason)

        let row = DaemonRuntimeRows.daemonText(daemon: nil, downReason: reason)
        XCTAssertTrue(row.localizedCaseInsensitiveContains("down"), row)
        XCTAssertTrue(row.contains("not running"), row)
        XCTAssertTrue(DaemonRuntimeRows.isAttention(row), "a real outage keeps the attention colour")
    }

    func test_daemon_row_names_PID_uptime_and_sockets() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let snapshot = DaemonHealthSnapshot(
            pid: 4242,
            isResponsive: true,
            rssBytes: 0,
            uptime: 2 * 3600 + 5 * 60,
            openConnections: 3,
            lastSeenAt: now,
            startedAt: now.addingTimeInterval(-(2 * 3600 + 5 * 60))
        )

        let row = DaemonRuntimeRows.daemonText(daemon: snapshot, downReason: nil)

        XCTAssertEqual(row, "PID 4242 · up 2h 5m · 3 sockets")
        XCTAssertFalse(DaemonRuntimeRows.isAttention(row))
    }

    func test_last_seen_is_the_last_socket_answer_not_the_sample_time() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let answered = DaemonHealthSnapshot(
            pid: 1,
            isResponsive: true,
            rssBytes: 0,
            uptime: 600,
            openConnections: 1,
            lastSeenAt: now.addingTimeInterval(-5 * 60),
            startedAt: nil
        )
        XCTAssertEqual(DaemonRuntimeRows.lastSeenText(daemon: answered, now: now), "5m ago")

        let neverAnswered = DaemonHealthSnapshot(
            pid: 1,
            isResponsive: true,
            rssBytes: 0,
            uptime: 600,
            openConnections: 1,
            lastSeenAt: nil,
            startedAt: nil
        )
        XCTAssertEqual(DaemonRuntimeRows.lastSeenText(daemon: neverAnswered, now: now), "No socket answer yet")
    }

    @MainActor
    func test_collector_survives_a_restart_and_stamps_last_seen_from_the_brain_bus() async throws {
        let daemonA = try spawnStandInDaemon()
        let daemonB = try spawnStandInDaemon()
        let resolver = ScriptedDaemonPIDResolver(current: daemonA.processIdentifier)
        let bus = ScriptedBrainBus()
        let collector = StatsCollector(
            dbPath: "/nonexistent/brainbar-972.db",
            daemonMonitor: DaemonHealthMonitor(pidResolver: resolver),
            autoRefreshInterval: 3600,
            brainBusEvents: bus
        )
        defer { collector.stop() }

        collector.start()
        try await waitUntil { collector.daemon?.pid == daemonA.processIdentifier }

        daemonA.terminate()
        daemonA.waitUntilExit()
        resolver.current = daemonB.processIdentifier
        let tick = BrainBusEvent.healthTick(openConnections: 2)
            .withSequence(7, generatedAt: Date().addingTimeInterval(-3))
        bus.publish(tick)

        try await waitUntil { collector.daemon?.pid == daemonB.processIdentifier }
        let snapshot = try XCTUnwrap(collector.daemon)
        XCTAssertEqual(snapshot.lastSeenAt, tick.generatedAt, "Last seen is the socket answer's time")
        XCTAssertNil(collector.daemonDownReason)
        XCTAssertFalse(resolver.resolvedOnMainThread, "PID resolution must stay off the main thread")
    }

    @MainActor
    func testMakeUIStatsCollectorResolvesTheDaemonPIDOnEverySampleNotOnceAtLaunch() async throws {
        let daemonA = try spawnStandInDaemon()
        let daemonB = try spawnStandInDaemon()
        let resolver = ScriptedDaemonPIDResolver(current: daemonA.processIdentifier)
        let collector = BrainBarAppSupport.makeUIStatsCollector(
            dbPath: "/nonexistent/brainbar-972-ui.db",
            brainBusEvents: nil,
            daemonPIDResolver: resolver
        )
        defer { collector.stop() }

        collector.refresh(force: true)
        try await waitUntil { collector.daemon?.pid == daemonA.processIdentifier }

        daemonA.terminate()
        daemonA.waitUntilExit()
        resolver.current = daemonB.processIdentifier
        collector.refresh(force: true)

        try await waitUntil { collector.daemon?.pid == daemonB.processIdentifier }
        XCTAssertEqual(collector.daemon?.pid, daemonB.processIdentifier)
    }

    // MARK: - helpers

    /// A real, killable process standing in for BrainBarDaemon, so the monitor's
    /// `kill(pid, 0)` / `proc_pidinfo` path runs against a genuine PID.
    private func spawnStandInDaemon() throws -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sleep")
        process.arguments = ["120"]
        try process.run()
        children.append(process)
        return process
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

private final class ScriptedDaemonPIDResolver: DaemonPIDResolving, @unchecked Sendable {
    private let lock = NSLock()
    private var _current: pid_t?
    private var _resolveCount = 0
    private var _resolvedOnMainThread = false

    init(current: pid_t?) {
        _current = current
    }

    var current: pid_t? {
        get { lock.withLock { _current } }
        set { lock.withLock { _current = newValue } }
    }

    var resolveCount: Int { lock.withLock { _resolveCount } }
    var resolvedOnMainThread: Bool { lock.withLock { _resolvedOnMainThread } }

    func isDaemon(_ pid: pid_t) -> Bool {
        lock.withLock { pid == _current }
    }

    func resolveDaemonPID() -> pid_t? {
        lock.withLock {
            _resolveCount += 1
            if Thread.isMainThread { _resolvedOnMainThread = true }
            return _current
        }
    }
}

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
