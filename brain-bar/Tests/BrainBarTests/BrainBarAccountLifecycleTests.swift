@testable import BrainBarLifecycle
import Darwin
import Foundation
import XCTest

final class BrainBarAccountLifecycleTests: XCTestCase {
    private typealias Identity = BrainBarLifecycleWatchdog.ProcessIdentity
    private final class Processes: @unchecked Sendable {
        let lock = NSLock()
        var identities: [pid_t: Identity] = [:]
        var signals: [(pid_t, Int32)] = []
        var launches = 0
        var reads = 0
        var replacementBeforeTERM: Identity?
        var replacementAfterTERM: Identity?
        var unreadableAfterTERM = false
        func read(_ pid: pid_t) -> Identity? {
            lock.lock(); defer { lock.unlock() }
            reads += 1
            if reads == 2, let replacementBeforeTERM { identities[pid] = replacementBeforeTERM }
            return identities[pid]
        }

        func signal(_ pid: pid_t, _ signal: Int32) {
            lock.lock(); defer { lock.unlock() }
            signals.append((pid, signal))
            if signal == SIGTERM, let replacementAfterTERM { identities[pid] = replacementAfterTERM }
            if signal == SIGTERM, unreadableAfterTERM { identities[pid] = nil }
        }

        func launch() {
            lock.lock(); defer { lock.unlock() }; launches += 1
        }

        func result() -> ([Int32], Int) {
            lock.lock(); defer { lock.unlock() }; return (signals.map(\.1), launches)
        }
    }

    func testDefaultHeartbeatNamespaceBelongsToCurrentAccount() {
        let home = NSHomeDirectory() + "/"
        XCTAssertTrue(BrainBarLifecycleWatchdog.uiHeartbeatPath.hasPrefix(home))
        XCTAssertTrue(BrainBarLifecycleWatchdog.daemonHeartbeatPath.hasPrefix(home))
        XCTAssertNotEqual(BrainBarLifecycleWatchdog.uiHeartbeatPath, BrainBarLifecycleWatchdog.daemonHeartbeatPath)
    }

    func testWriterRefusesSymlinkAndPreservesSentinel() throws {
        let directory = try scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sentinel = directory.appendingPathComponent("sentinel")
        let heartbeat = directory.appendingPathComponent("heartbeat")
        try Data("private sentinel".utf8).write(to: sentinel)
        try FileManager.default.createSymbolicLink(at: heartbeat, withDestinationURL: sentinel)
        BrainBarLifecycleWatchdog.writeHeartbeat(to: heartbeat.path)
        XCTAssertEqual(try Data(contentsOf: sentinel), Data("private sentinel".utf8))
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: heartbeat.path), sentinel.path)
    }

    func testWriterRefusesHardlinkWithoutReplacingEitherName() throws {
        let directory = try scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let sentinel = directory.appendingPathComponent("sentinel")
        let heartbeat = directory.appendingPathComponent("heartbeat")
        try Data("private sentinel".utf8).write(to: sentinel)
        XCTAssertEqual(link(sentinel.path, heartbeat.path), 0)
        BrainBarLifecycleWatchdog.writeHeartbeat(to: heartbeat.path)
        XCTAssertEqual(try Data(contentsOf: sentinel), Data("private sentinel".utf8))
        XCTAssertEqual(try Data(contentsOf: heartbeat), Data("private sentinel".utf8))
    }

    func testDistinctAccountHomesNeverAdoptGlobalHeartbeat() {
        let first = BrainBarLifecycleWatchdog.heartbeatPath("ui", home: "/Users/account501")
        let second = BrainBarLifecycleWatchdog.heartbeatPath("ui", home: "/Users/account502")
        XCTAssertNotEqual(first, second)
        XCTAssertFalse(first.hasPrefix("/tmp/"))
        XCTAssertTrue(second.hasPrefix("/Users/account502/"))
    }

    func testExpectedPeerUsesThePerAccountInstalledBundle() {
        let app = URL(fileURLWithPath: "/Users/account502/Applications/BrainBar.app")
        XCTAssertEqual(BrainBarLifecycleWatchdog.expectedExecutablePath("BrainBarDaemon", bundleURL: app),
                       app.appendingPathComponent("Contents/MacOS/BrainBarDaemon").path)
        XCTAssertEqual(BrainBarLifecycleWatchdog.expectedExecutablePath("BrainBar", bundleURL: URL(fileURLWithPath: "/scratch")),
                       "/Applications/BrainBar.app/Contents/MacOS/BrainBar")
    }

    func testWriterRefusesWritableParentWithoutCreatingHeartbeat() throws {
        let directory = try scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.setAttributes([.posixPermissions: 0o777], ofItemAtPath: directory.path)
        let path = directory.appendingPathComponent("heartbeat").path
        XCTAssertFalse(BrainBarLifecycleWatchdog.writeHeartbeat(to: path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: path))
    }

    func testOwnedWriterIsPrivateAndOtherUIDCannotUseItsFreshHeartbeat() throws {
        let directory = try scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("heartbeat").path
        XCTAssertTrue(BrainBarLifecycleWatchdog.writeHeartbeat(to: path))
        let attributes = try FileManager.default.attributesOfItem(atPath: path)
        XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        let foreignUID: uid_t = getuid() == 502 ? 501 : 502
        switch BrainBarLifecycleWatchdog.observeHeartbeat(atPath: path, clock: .system, ownerUID: foreignUID) {
        case .unmeasured: break
        case .measured: XCTFail("A different UID must not adopt a fresh heartbeat")
        }
    }

    func testWritableHeartbeatIsNotTrustedOrModified() throws {
        let directory = try scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("heartbeat").path
        XCTAssertTrue(BrainBarLifecycleWatchdog.writeHeartbeat(to: path))
        let before = try Data(contentsOf: URL(fileURLWithPath: path))
        try FileManager.default.setAttributes([.posixPermissions: 0o666], ofItemAtPath: path)
        switch BrainBarLifecycleWatchdog.observeHeartbeat(atPath: path, clock: .system) {
        case .unmeasured: break
        case .measured: XCTFail("A writable heartbeat is not owned liveness")
        }
        XCTAssertFalse(BrainBarLifecycleWatchdog.writeHeartbeat(to: path))
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: path)), before)
    }

    func testForeignRealUIDWrongBundleAndUnreadableProcessesAreNeverSignalled() throws {
        for identity in [Identity(uid: 501, realUID: 501, startedSeconds: 1, startedMicroseconds: 0, executablePath: "/test/BrainBar"),
                         Identity(uid: 502, realUID: 501, startedSeconds: 1, startedMicroseconds: 0, executablePath: "/test/BrainBar"),
                         Identity(uid: 502, realUID: 502, startedSeconds: 1, startedMicroseconds: 0, executablePath: "/other/BrainBar"), nil]
        {
            let processes = Processes()
            processes.identities[111] = identity
            try run(processes)
            XCTAssertTrue(processes.result().0.isEmpty)
            if identity == nil { XCTAssertEqual(processes.result().1, 0) }
        }
    }

    func testOwnedHungPeerStillReceivesBothSignalsAndRelaunch() throws {
        let processes = Processes()
        processes.identities[111] = ownedIdentity()
        try run(processes)
        XCTAssertEqual(processes.result().0, [SIGTERM, SIGKILL])
        XCTAssertEqual(processes.result().1, 1)
    }

    func testPIDReusedAfterTERMIsNotKilledOrKickstarted() throws {
        let processes = Processes()
        processes.identities[111] = ownedIdentity()
        processes.replacementAfterTERM = ownedIdentity(start: 2)
        try run(processes)
        XCTAssertEqual(processes.result().0, [SIGTERM])
        XCTAssertEqual(processes.result().1, 0, "kickstart -k must not kill the replacement either")
    }

    func testUIDChangedAfterTERMNeverReceivesKILL() throws {
        let processes = Processes()
        processes.identities[111] = ownedIdentity()
        processes.replacementAfterTERM = .init(uid: 501, realUID: 501, startedSeconds: 2, startedMicroseconds: 0, executablePath: "/test/BrainBar")
        try run(processes)
        XCTAssertEqual(processes.result().0, [SIGTERM])
    }

    func testIdentityChangesBeforeTERMNeverReceiveEitherSignal() throws {
        for replacement in [ownedIdentity(start: 2), Identity(uid: 501, realUID: 501, startedSeconds: 2, startedMicroseconds: 0, executablePath: "/test/BrainBar")] {
            let processes = Processes()
            processes.identities[111] = ownedIdentity()
            processes.replacementBeforeTERM = replacement
            try run(processes)
            XCTAssertTrue(processes.result().0.isEmpty)
        }
    }

    func testUnreadableReplacementAfterTERMIsNeitherKilledNorKickstarted() throws {
        let processes = Processes()
        processes.identities[111] = ownedIdentity()
        processes.unreadableAfterTERM = true
        try run(processes)
        XCTAssertEqual(processes.result().0, [SIGTERM])
        XCTAssertEqual(processes.result().1, 0)
    }

    private func ownedIdentity(start: UInt64 = 1) -> Identity {
        .init(uid: 502, realUID: 502, startedSeconds: start, startedMicroseconds: 0, executablePath: "/test/BrainBar")
    }

    private func run(_ processes: Processes) throws {
        let directory = try scratch()
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("heartbeat").path
        let clock = BrainBarLifecycleWatchdog.HeartbeatClock(wallNow: { Date() }, uptimeNanos: { 1 }, bootSession: { "test" })
        XCTAssertTrue(BrainBarLifecycleWatchdog.writeHeartbeat(to: path, clock: clock))
        let watchdog = BrainBarLifecycleWatchdog(
            configuration: .init(watchedName: "BrainBar", heartbeatPath: path, terminateGraceInterval: 0.01,
                                 relaunchCommand: .launchctlKickstart(label: "never-run"), expectedExecutablePath: "/test/BrainBar"),
            processProvider: { [111] }, terminateProcess: { processes.signal($0, $1) }, relaunch: { _ in processes.launch() },
            clock: .init(wallNow: { Date() }, uptimeNanos: { 70_000_000_000 }, bootSession: { "test" }),
            ownerUID: 502, processIdentity: { processes.read($0) }
        )
        watchdog.checkNow()
        let settled = expectation(description: "watchdog callbacks settle")
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.2) { settled.fulfill() }
        wait(for: [settled], timeout: 2)
        withExtendedLifetime(watchdog) {}
    }

    private func scratch() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("brainbar-account-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        return directory
    }
}
