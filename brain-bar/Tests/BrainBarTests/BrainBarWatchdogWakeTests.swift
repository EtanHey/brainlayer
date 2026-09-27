import Darwin
import Foundation
import XCTest
@testable import BrainBarLifecycle

/// The lifecycle watchdog killed a healthy BrainBarDaemon on every macOS wake
/// (9 restarts on 2026-09-27, each ~3 s after a DarkWake/Wake): the heartbeat
/// file cannot advance while the Mac sleeps, so its wall-clock mtime looked
/// stale the moment the watchdog's timer fired after wake. These tests pin the
/// fix: staleness is measured in awake-seconds (CLOCK_UPTIME_RAW), and a daemon
/// whose socket answers is never killed on a stale heartbeat alone.
final class BrainBarWatchdogWakeTests: XCTestCase {
    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var _signals: [(pid_t, Int32)] = []
        private var _relaunches = 0
        private var _probes = 0
        var probeAnswers: Bool
        var pids: [pid_t]

        init(probeAnswers: Bool, pids: [pid_t] = [111]) {
            self.probeAnswers = probeAnswers
            self.pids = pids
        }

        func processProvider() -> [pid_t] {
            lock.lock(); defer { lock.unlock() }
            return pids
        }

        func terminate(_ pid: pid_t, _ signal: Int32) {
            lock.lock(); defer { lock.unlock() }
            _signals.append((pid, signal))
        }

        func relaunch() {
            lock.lock(); defer { lock.unlock() }
            _relaunches += 1
        }

        func probe() -> Bool {
            lock.lock(); defer { lock.unlock() }
            _probes += 1
            return probeAnswers
        }

        var signals: [(pid_t, Int32)] { lock.lock(); defer { lock.unlock() }; return _signals }
        var relaunches: Int { lock.lock(); defer { lock.unlock() }; return _relaunches }
        var probes: Int { lock.lock(); defer { lock.unlock() }; return _probes }
    }

    private final class MutableClock: @unchecked Sendable {
        private let lock = NSLock()
        private var wall: Date
        private var uptime: UInt64
        private var boot: String

        init(wall: Date, uptimeNanos: UInt64, bootSession: String = "BOOT-A") {
            self.wall = wall
            self.uptime = uptimeNanos
            self.boot = bootSession
        }

        /// Sleep advances the wall clock only; CLOCK_UPTIME_RAW pauses.
        func sleep(seconds: TimeInterval) {
            lock.lock(); defer { lock.unlock() }
            wall = wall.addingTimeInterval(seconds)
        }

        /// Awake time advances both clocks.
        func awake(seconds: TimeInterval) {
            lock.lock(); defer { lock.unlock() }
            wall = wall.addingTimeInterval(seconds)
            uptime += UInt64(seconds * 1_000_000_000)
        }

        func reboot(bootSession: String, uptimeNanos: UInt64) {
            lock.lock(); defer { lock.unlock() }
            boot = bootSession
            uptime = uptimeNanos
        }

        var heartbeatClock: BrainBarLifecycleWatchdog.HeartbeatClock {
            BrainBarLifecycleWatchdog.HeartbeatClock(
                wallNow: { [self] in lock.lock(); defer { lock.unlock() }; return wall },
                uptimeNanos: { [self] in lock.lock(); defer { lock.unlock() }; return uptime },
                bootSession: { [self] in lock.lock(); defer { lock.unlock() }; return boot }
            )
        }
    }

    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-watchdog-wake-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private var heartbeatPath: String { directory.appendingPathComponent("daemon.heartbeat").path }
    private var eventLogPath: String { directory.appendingPathComponent("daemon-debug.log").path }

    private func eventLog() -> String {
        (try? String(contentsOfFile: eventLogPath, encoding: .utf8)) ?? ""
    }

    private func makeWatchdog(
        clock: MutableClock,
        recorder: Recorder,
        withProbe: Bool = true,
        relaunched: XCTestExpectation? = nil
    ) -> BrainBarLifecycleWatchdog {
        var probe: (@Sendable () -> Bool)?
        if withProbe {
            probe = { @Sendable in recorder.probe() }
        }
        return BrainBarLifecycleWatchdog(
            configuration: .init(
                watchedName: "TestBrainBarDaemon",
                heartbeatPath: heartbeatPath,
                staleTimeout: 45,
                checkInterval: 10,
                terminateGraceInterval: 0.02,
                relaunchCommand: .launchctlKickstart(label: "com.brainlayer.test-never-run"),
                eventLogPath: eventLogPath
            ),
            processProvider: { recorder.processProvider() },
            terminateProcess: { pid, signal in recorder.terminate(pid, signal) },
            relaunch: { _ in
                recorder.relaunch()
                relaunched?.fulfill()
            },
            clock: clock.heartbeatClock,
            livenessProbe: probe
        )
    }

    private func writeHeartbeat(_ clock: MutableClock) {
        BrainBarLifecycleWatchdog.writeHeartbeat(to: heartbeatPath, clock: clock.heartbeatClock)
    }

    // MARK: - (a) sleep does not count

    func testLongSleepThenWakeWithHealthyDaemonDoesNotKillOrEvenProbe() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        writeHeartbeat(clock)
        // Reality after an 8 h sleep: the file's mtime is 8 h old too.
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: 1_790_000_000)],
            ofItemAtPath: heartbeatPath
        )
        clock.awake(seconds: 2)
        clock.sleep(seconds: 8 * 3600)
        clock.awake(seconds: 3) // the watchdog timer fires ~3 s after DarkWake

        let recorder = Recorder(probeAnswers: true)
        let watchdog = makeWatchdog(clock: clock, recorder: recorder)
        watchdog.checkNow()

        XCTAssertEqual(
            try XCTUnwrap(BrainBarLifecycleWatchdog.heartbeatUptimeAge(atPath: heartbeatPath, clock: clock.heartbeatClock)),
            5,
            accuracy: 0.001,
            "Sleep must not count toward heartbeat age."
        )
        XCTAssertTrue(recorder.signals.isEmpty, "A healthy daemon must survive wake.")
        XCTAssertEqual(recorder.relaunches, 0)
        XCTAssertEqual(recorder.probes, 0, "A heartbeat that is fresh in awake-time needs no probe.")
    }

    // MARK: - (b) a real hang is still recovered

    func testAwakeStaleHeartbeatAndFailedProbeKillsAndKickstartsWithReason() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        writeHeartbeat(clock)
        clock.awake(seconds: 60)

        let recorder = Recorder(probeAnswers: false)
        let relaunched = expectation(description: "wedged daemon is relaunched")
        let watchdog = makeWatchdog(clock: clock, recorder: recorder, relaunched: relaunched)
        watchdog.checkNow()
        wait(for: [relaunched], timeout: 2)

        XCTAssertEqual(recorder.probes, 1)
        XCTAssertEqual(recorder.signals.map(\.0), [111, 111])
        XCTAssertEqual(recorder.signals.map(\.1), [SIGTERM, SIGKILL])
        XCTAssertEqual(recorder.relaunches, 1)
        let log = eventLog()
        XCTAssertTrue(log.contains("TestBrainBarDaemon"), log)
        XCTAssertTrue(log.contains("terminating PIDs 111"), log)
        XCTAssertTrue(log.contains("60s awake"), "Kill reason must name the awake-time age: \(log)")
        XCTAssertTrue(log.contains("liveness probe failed"), "Kill reason must name the probe failure: \(log)")
    }

    // MARK: - (c) a stale heartbeat alone never kills a daemon that answers

    func testAwakeStaleHeartbeatButAnsweringDaemonIsNotKilledAndIsLoggedOnce() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        writeHeartbeat(clock)
        clock.awake(seconds: 60)

        let recorder = Recorder(probeAnswers: true)
        let watchdog = makeWatchdog(clock: clock, recorder: recorder)
        watchdog.checkNow()
        clock.awake(seconds: 10)
        watchdog.checkNow()

        XCTAssertEqual(recorder.probes, 2, "Each stale check re-probes before deciding.")
        XCTAssertTrue(recorder.signals.isEmpty)
        XCTAssertEqual(recorder.relaunches, 0)
        let lines = eventLog().split(separator: "\n").filter { $0.contains("answered") }
        XCTAssertEqual(lines.count, 1, "Stale-but-answering is logged once per stale episode: \(eventLog())")
    }

    func testStaleHeartbeatWithNoPIDFoundButAnsweringSocketDoesNotKickstart() throws {
        // `launchctl kickstart -k` kills the running job, so a PID-lookup miss
        // must not turn into a restart of a daemon that answers.
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        writeHeartbeat(clock)
        clock.awake(seconds: 60)

        let recorder = Recorder(probeAnswers: true, pids: [])
        let watchdog = makeWatchdog(clock: clock, recorder: recorder)
        watchdog.checkNow()

        XCTAssertEqual(recorder.relaunches, 0)
        XCTAssertTrue(recorder.signals.isEmpty)
    }

    func testHeartbeatFromAPreviousBootIsNotMeasurableOnTheUptimeClock() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        writeHeartbeat(clock)
        clock.reboot(bootSession: "BOOT-B", uptimeNanos: 510_000_000_000)

        XCTAssertNil(BrainBarLifecycleWatchdog.heartbeatUptimeAge(atPath: heartbeatPath, clock: clock.heartbeatClock))
    }

    func testMissingOrLegacyHeartbeatIsNotMeasurableOnTheUptimeClock() throws {
        let clock = MutableClock(wall: Date(), uptimeNanos: 1)
        XCTAssertNil(BrainBarLifecycleWatchdog.heartbeatUptimeAge(atPath: heartbeatPath, clock: clock.heartbeatClock))
        try "1790000000.0\n".write(toFile: heartbeatPath, atomically: true, encoding: .utf8)
        XCTAssertNil(BrainBarLifecycleWatchdog.heartbeatUptimeAge(atPath: heartbeatPath, clock: clock.heartbeatClock))
    }

    // MARK: - mixed-version upgrade: a legacy (wall-clock only) heartbeat

    /// #970 review B2: a new daemon watching a still-running old UI sees only
    /// wall-clock heartbeats, and the UI watchdog has no probe. Its age must be
    /// counted in awake-seconds this watchdog has seen it unchanged.
    private func writeLegacyHeartbeat(epoch: TimeInterval) throws {
        try "\(epoch)\n".write(toFile: heartbeatPath, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.modificationDate: Date(timeIntervalSince1970: epoch)],
            ofItemAtPath: heartbeatPath
        )
    }

    func testLegacyUIHeartbeatSurvivesSleepDuringMixedVersionUpgradeButAnAwakeHangIsRecovered() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        try writeLegacyHeartbeat(epoch: 1_790_000_000)
        let recorder = Recorder(probeAnswers: false, pids: [333])
        let relaunched = expectation(description: "hung legacy UI is relaunched")
        let watchdog = makeWatchdog(clock: clock, recorder: recorder, withProbe: false, relaunched: relaunched)

        clock.awake(seconds: 2)
        watchdog.checkNow()
        clock.sleep(seconds: 8 * 3600)
        clock.awake(seconds: 3)
        watchdog.checkNow()
        XCTAssertTrue(recorder.signals.isEmpty, "A legacy heartbeat must not kill a healthy UI on wake: \(eventLog())")

        clock.awake(seconds: 60)
        watchdog.checkNow()
        wait(for: [relaunched], timeout: 2)
        XCTAssertEqual(recorder.signals.map(\.1), [SIGTERM, SIGKILL])
        XCTAssertTrue(eventLog().contains("unchanged for 63s awake"), eventLog())
    }

    func testLegacyHeartbeatFirstSeenRightAfterWakeGetsAnAwakeGrace() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        try writeLegacyHeartbeat(epoch: 1_790_000_000)
        clock.sleep(seconds: 8 * 3600)
        clock.awake(seconds: 3)

        let recorder = Recorder(probeAnswers: false, pids: [333])
        let watchdog = makeWatchdog(clock: clock, recorder: recorder, withProbe: false)
        watchdog.checkNow()
        clock.awake(seconds: 30)
        watchdog.checkNow()

        XCTAssertTrue(recorder.signals.isEmpty, eventLog())
        XCTAssertEqual(recorder.relaunches, 0)
    }

    func testAdvancingLegacyHeartbeatResetsTheAwakeGrace() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        try writeLegacyHeartbeat(epoch: 1_790_000_000)
        let recorder = Recorder(probeAnswers: false, pids: [333])
        let watchdog = makeWatchdog(clock: clock, recorder: recorder, withProbe: false)

        watchdog.checkNow()
        clock.awake(seconds: 40)
        try writeLegacyHeartbeat(epoch: 1_790_000_040)
        watchdog.checkNow()
        clock.awake(seconds: 40)
        watchdog.checkNow()

        XCTAssertTrue(recorder.signals.isEmpty, "A legacy heartbeat that keeps advancing is alive: \(eventLog())")
    }

    func testMissingUIHeartbeatGetsAnAwakeGraceThenIsRecovered() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        let recorder = Recorder(probeAnswers: false, pids: [333])
        let relaunched = expectation(description: "UI with no heartbeat is relaunched")
        let watchdog = makeWatchdog(clock: clock, recorder: recorder, withProbe: false, relaunched: relaunched)

        watchdog.checkNow()
        XCTAssertTrue(recorder.signals.isEmpty, "The first sighting of a missing heartbeat starts the grace.")
        clock.awake(seconds: 46)
        watchdog.checkNow()
        wait(for: [relaunched], timeout: 2)
        XCTAssertEqual(recorder.signals.map(\.1), [SIGTERM, SIGKILL])
        XCTAssertTrue(eventLog().contains("heartbeat missing"), eventLog())
    }

    func testHeartbeatPayloadCarriesWallUptimeAndBoot() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 42_000_000_000, bootSession: "BOOT-A")
        writeHeartbeat(clock)
        let payload = try String(contentsOfFile: heartbeatPath, encoding: .utf8)
        let lines = payload.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.first, "1790000000.0", "Line 1 stays the wall-clock epoch for any legacy reader.")
        XCTAssertTrue(lines.contains("uptime_ns=42000000000"), payload)
        XCTAssertTrue(lines.contains("boot=BOOT-A"), payload)
    }

    func testSystemClockUptimeIsTheSleepPausingClock() {
        let clock = BrainBarLifecycleWatchdog.HeartbeatClock.system
        let expected = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let observed = clock.uptimeNanos()
        XCTAssertLessThan(observed - expected, 1_000_000_000)
        XCTAssertFalse(clock.bootSession().isEmpty, "kern.bootsessionuuid must resolve")
    }

    // MARK: - (d) the reverse watchdog (daemon watches the UI app)

    func testUIWatchdogWithoutProbeIgnoresSleepButStillRestartsAnAwakeHang() throws {
        let clock = MutableClock(wall: Date(timeIntervalSince1970: 1_790_000_000), uptimeNanos: 500_000_000_000)
        writeHeartbeat(clock)
        clock.sleep(seconds: 8 * 3600)
        clock.awake(seconds: 3)

        let recorder = Recorder(probeAnswers: false, pids: [333])
        let relaunched = expectation(description: "hung UI is relaunched")
        let watchdog = makeWatchdog(clock: clock, recorder: recorder, withProbe: false, relaunched: relaunched)
        watchdog.checkNow()
        XCTAssertTrue(recorder.signals.isEmpty, "Sleep must not kill the UI app either.")

        clock.awake(seconds: 60)
        watchdog.checkNow()
        wait(for: [relaunched], timeout: 2)
        XCTAssertEqual(recorder.signals.map(\.1), [SIGTERM, SIGKILL])
        XCTAssertTrue(eventLog().contains("no liveness probe"), eventLog())
    }

    func testFactoriesProbeTheDaemonSocketAndLogToTheDaemonLog() {
        let daemon = BrainBarLifecycleWatchdog.makeDaemonWatchdog()
        XCTAssertTrue(daemon.hasLivenessProbe, "The daemon watchdog must probe the socket before killing.")
        XCTAssertEqual(daemon.eventLogPath, BrainBarLifecycleWatchdog.daemonDebugLogPath)
        let ui = BrainBarLifecycleWatchdog.makeUIWatchdog(bundlePath: "/nonexistent/BrainBar.app")
        XCTAssertFalse(ui.hasLivenessProbe)
        XCTAssertEqual(ui.eventLogPath, BrainBarLifecycleWatchdog.daemonDebugLogPath)
        XCTAssertEqual(BrainBarLifecycleWatchdog.daemonDebugLogPath, "/tmp/brainbar-debug.log")
    }

    // MARK: - the real socket probe, against a scratch socket only

    func testSocketProbeAnswersOnlyWhenAFramedPingGetsAReply() throws {
        let answering = try ScratchSocketServer(replyToPing: true)
        defer { answering.stop() }
        XCTAssertTrue(BrainBarLifecycleWatchdog.socketAnswersPing(path: answering.path, timeout: 2))

        let wedged = try ScratchSocketServer(replyToPing: false)
        defer { wedged.stop() }
        let started = Date()
        XCTAssertFalse(
            BrainBarLifecycleWatchdog.socketAnswersPing(path: wedged.path, timeout: 0.3),
            "Accepting a connection is not an answer: the kernel accepts into the backlog even when the daemon is wedged."
        )
        XCTAssertLessThan(Date().timeIntervalSince(started), 2, "The probe must honour its timeout.")

        let missing = directory.appendingPathComponent("absent.sock").path
        XCTAssertFalse(BrainBarLifecycleWatchdog.socketAnswersPing(path: missing, timeout: 0.3))
    }
}

/// A minimal AF_UNIX server on a scratch path. It never touches /tmp/brainbar.sock.
private final class ScratchSocketServer: @unchecked Sendable {
    let path: String
    private let fd: Int32
    private let replyToPing: Bool
    private let queue = DispatchQueue(label: "scratch-socket-server")
    private var accepted: [Int32] = []
    private let lock = NSLock()

    init(replyToPing: Bool) throws {
        self.replyToPing = replyToPing
        path = NSTemporaryDirectory() + "bbwd-\(UUID().uuidString.prefix(8)).sock"
        unlink(path)
        fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw POSIXError(.EIO) }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(path.utf8)
        guard bytes.count < MemoryLayout.size(ofValue: addr.sun_path) else { throw POSIXError(.ENAMETOOLONG) }
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            raw.copyBytes(from: bytes)
            raw[bytes.count] = 0
        }
        let bound = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bound == 0, listen(fd, 4) == 0 else { throw POSIXError(.EADDRINUSE) }
        queue.async { [self] in serve() }
    }

    /// Serves exactly one connection, so no thread stays parked in accept().
    private func serve() {
        let client = accept(fd, nil, nil)
        guard client >= 0 else { return }
        lock.lock(); accepted.append(client); lock.unlock()
        guard replyToPing else { return }
        var buffer = [UInt8](repeating: 0, count: 4096)
        var received = Data()
        while true {
            let n = read(client, &buffer, buffer.count)
            guard n > 0 else { return }
            received.append(contentsOf: buffer[0..<n])
            guard let text = String(data: received, encoding: .utf8),
                  let split = text.range(of: "\r\n\r\n"),
                  text[split.upperBound...].contains("}")
            else { continue }
            let request = String(text[split.upperBound...])
            guard let object = try? JSONSerialization.jsonObject(with: Data(request.utf8)) as? [String: Any],
                  object["method"] as? String == "ping"
            else { return }
            let idJSON = (object["id"] as? String).map { "\"\($0)\"" } ?? "0"
            let body = "{\"jsonrpc\":\"2.0\",\"id\":\(idJSON),\"result\":{}}"
            let reply = "Content-Length: \(body.utf8.count)\r\n\r\n\(body)"
            _ = reply.withCString { write(client, $0, strlen($0)) }
            return
        }
    }

    func stop() {
        close(fd)
        lock.lock(); accepted.forEach { close($0) }; lock.unlock()
        unlink(path)
    }
}
