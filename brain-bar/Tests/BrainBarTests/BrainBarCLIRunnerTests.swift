import XCTest
@testable import BrainBar

/// #1028 review B1: the live `brainlayer` runner returns at its timeout no matter what the child
/// does. It reads stdout without blocking on it, stops the child's whole process group (SIGTERM,
/// a grace, then SIGKILL), says the run timed out, and leaks no process or file descriptor.
/// Every test runs a real, short-lived `/bin/sh` child; nothing touches the brainlayer CLI.
final class BrainBarCLIRunnerTests: XCTestCase {
    private func sh(_ script: String, timeout: TimeInterval, grace: TimeInterval = 0.3) -> (result: BrainLayerCLIResult?, elapsed: TimeInterval) {
        let start = Date()
        let result = ProcessBrainLayerCLIRunner.spawn(
            executable: "/bin/sh", arguments: ["-c", script], environment: ["PATH": "/usr/bin:/bin"],
            timeout: timeout, grace: grace
        )
        return (result, Date().timeIntervalSince(start))
    }

    /// The pids a script printed as `pids <pid> <pid> …`.
    private func pids(in output: String?) -> [pid_t] {
        guard let line = output?.split(separator: "\n").first(where: { $0.hasPrefix("pids ") }) else { return [] }
        return line.split(separator: " ").dropFirst().compactMap { pid_t($0) }
    }

    /// Gone means no process at all (a zombie would still answer kill 0), polled briefly because an
    /// orphaned grandchild is reaped by launchd, not by us.
    private func isGone(_ pid: pid_t, within limit: TimeInterval = 3) -> Bool {
        let deadline = Date().addingTimeInterval(limit)
        repeat {
            if kill(pid, 0) == -1, errno == ESRCH { return true }
            usleep(20_000)
        } while Date() < deadline
        return false
    }

    private func openDescriptorCount() -> Int {
        (try? FileManager.default.contentsOfDirectory(atPath: "/dev/fd").count) ?? -1
    }

    func test_normal_completion_returns_stdout_and_the_exit_status() {
        let (result, elapsed) = sh(#"echo '{"state":"valid"}'; exit 3"#, timeout: 10)
        XCTAssertEqual(result, BrainLayerCLIResult(terminationStatus: 3, stdout: "{\"state\":\"valid\"}\n", timedOut: false))
        XCTAssertLessThan(elapsed, 2)
    }

    func test_a_child_that_ignores_sigterm_is_killed_with_its_group_at_timeout_plus_grace() {
        // Ignored signals survive exec, so the backgrounded sleep ignores SIGTERM too.
        let (result, elapsed) = sh(#"trap "" TERM; sleep 5 & echo "pids $$ $!"; wait"#, timeout: 0.3, grace: 0.3)
        XCTAssertEqual(result?.timedOut, true)
        XCTAssertLessThan(elapsed, 0.3 + 0.3 + 1.5, "bounded by timeout + grace, not by the child")
        let children = pids(in: result?.stdout)
        XCTAssertEqual(children.count, 2, "output written before the timeout is kept: \(result?.stdout ?? "nil")")
        for pid in children { XCTAssertTrue(isGone(pid), "pid \(pid) outlived the runner") }
    }

    func test_a_child_that_honours_sigterm_stops_without_waiting_for_the_grace() {
        let (result, elapsed) = sh(#"echo started; sleep 5"#, timeout: 0.3, grace: 5)
        XCTAssertEqual(result?.timedOut, true)
        XCTAssertEqual(result?.stdout, "started\n")
        XCTAssertLessThan(elapsed, 0.3 + 1.5)
    }

    func test_a_grandchild_holding_stdout_open_cannot_hold_the_runner() {
        let (result, elapsed) = sh(#"sleep 5 & echo "pids $!"; exit 0"#, timeout: 10)
        XCTAssertEqual(result?.terminationStatus, 0)
        XCTAssertEqual(result?.timedOut, false)
        XCTAssertLessThan(elapsed, 2.5)
        for pid in pids(in: result?.stdout) { XCTAssertTrue(isGone(pid), "pid \(pid) outlived the runner") }
    }

    func test_no_file_descriptor_leaks() {
        let before = openDescriptorCount()
        _ = sh("echo ok", timeout: 5)
        _ = sh(#"trap "" TERM; sleep 5"#, timeout: 0.2, grace: 0.2)
        _ = sh(#"sleep 5 & echo bg"#, timeout: 5)
        XCTAssertEqual(openDescriptorCount(), before)
    }

    func test_a_missing_executable_is_nil() {
        XCTAssertNil(ProcessBrainLayerCLIRunner.spawn(executable: "/nonexistent/brainlayer", arguments: [], environment: [:], timeout: 1))
    }

    // MARK: what a timed-out run means to the model

    func test_a_timed_out_run_is_a_timeout_not_an_exit_code() {
        let killed = BrainLayerCLIResult(terminationStatus: 137, stdout: "", timedOut: true)
        XCTAssertEqual(DriveAuthOutcome.parse(killed), .timeout("brainlayer backup auth did not finish in time"))
        XCTAssertEqual(DriveAuthStatus.parse(killed), .unknown("brainlayer backup auth --status did not finish in time"))
        // A report the CLI printed before it was stopped still wins.
        let reported = BrainLayerCLIResult(terminationStatus: 143, stdout: #"{"status":"ok","reason":""}"# + "\n", timedOut: true)
        XCTAssertEqual(DriveAuthOutcome.parse(reported), .ok)
    }
}
