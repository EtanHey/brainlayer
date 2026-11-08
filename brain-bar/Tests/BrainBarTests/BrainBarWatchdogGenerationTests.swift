import Darwin
import Foundation
import XCTest
@testable import BrainBarLifecycle

final class BrainBarWatchdogGenerationTests: XCTestCase {
    final class State: @unchecked Sendable {
        let lock = NSLock()
        var delayed: [@Sendable () -> Void] = []
        var signals: [Int32] = []
        var launches = 0
        var uptime: UInt64 = 100_000_000_000
        func schedule(_ action: @escaping @Sendable () -> Void) { lock.lock(); defer { lock.unlock() }; delayed.append(action) }
        func fire() { lock.lock(); let work = delayed; delayed = []; lock.unlock(); work.forEach { $0() } }
    }

    func testStopFencesAlreadyScheduledKillAndRelaunch() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build.noindex/watchdog-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let heartbeat = root.appendingPathComponent("heartbeat").path
        let oldClock = BrainBarLifecycleWatchdog.HeartbeatClock(wallNow: { Date(timeIntervalSince1970: 100) }, uptimeNanos: { 1_000_000_000 }, bootSession: { "test" })
        BrainBarLifecycleWatchdog.writeHeartbeat(to: heartbeat, clock: oldClock)
        let state = State()
        let watchdog = BrainBarLifecycleWatchdog(
            configuration: .init(watchedName: "Test", heartbeatPath: heartbeat, staleTimeout: 1,
                relaunchCommand: .launchctlKickstart(label: "never-run")),
            processProvider: { [123] },
            terminateProcess: { _, signal in state.signals.append(signal) },
            relaunch: { _ in state.launches += 1 },
            delayedScheduler: { _, action in state.schedule(action) },
            clock: .init(wallNow: { Date(timeIntervalSince1970: 200) }, uptimeNanos: { 100_000_000_000 }, bootSession: { "test" }),
            processIdentity: { _ in .init(uid: getuid(), realUID: getuid(), startedSeconds: 1, startedMicroseconds: 0, executablePath: "/test/Test") })
        watchdog.checkNow()
        XCTAssertEqual(state.signals, [SIGTERM])
        watchdog.stop()
        state.fire()
        watchdog.stop() // Flush the serial queue after delivering the captured delayed block.
        XCTAssertEqual(state.signals, [SIGTERM])
        XCTAssertEqual(state.launches, 0)
    }
    func testRepeatedRestartsBackOffAndAlert() throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build.noindex/backoff-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let heartbeat = root.appendingPathComponent("heartbeat").path
        BrainBarLifecycleWatchdog.writeHeartbeat(to: heartbeat, clock: .init(
            wallNow: { Date(timeIntervalSince1970: 100) }, uptimeNanos: { 1_000_000_000 }, bootSession: { "test" }))
        let state = State()
        let log = root.appendingPathComponent("events").path
        let watchdog = BrainBarLifecycleWatchdog(configuration: .init(watchedName: "Test", heartbeatPath: heartbeat,
            staleTimeout: 1, relaunchCommand: .launchctlKickstart(label: "never-run"), eventLogPath: log),
            processProvider: { [] }, relaunch: { _ in state.launches += 1 },
            delayedScheduler: { _, action in state.schedule(action) },
            clock: .init(wallNow: { Date(timeIntervalSince1970: 200) }, uptimeNanos: { state.uptime }, bootSession: { "test" }))
        watchdog.checkNow()
        XCTAssertEqual(state.launches, 1)
        state.fire(); watchdog.stop()
        watchdog.checkNow()
        XCTAssertEqual(state.launches, 1, "No launch within the initial five-second backoff")
        state.uptime += 6_000_000_000
        watchdog.checkNow()
        XCTAssertEqual(state.launches, 2)
        state.fire(); watchdog.stop()
        watchdog.checkNow()
        XCTAssertEqual(state.launches, 2, "Second backoff grows to ten seconds")
        state.uptime += 11_000_000_000
        watchdog.checkNow()
        XCTAssertEqual(state.launches, 3)
        XCTAssertTrue(try String(contentsOfFile: log, encoding: .utf8).contains("ALERT:"))
        watchdog.stop()
    }

}
