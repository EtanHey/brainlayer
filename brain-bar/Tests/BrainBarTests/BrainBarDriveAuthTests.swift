import XCTest
@testable import BrainBar

/// "Reconnect Google Drive" (Etan: "click once and never again"). BrainBar reads and renews Drive
/// access only through `brainlayer backup auth` (the backend PR's CLI contract), off the main
/// thread, and never shows a token. Every test injects the command runner.
final class BrainBarDriveAuthTests: XCTestCase {
    private let now = ISO8601DateFormatter().date(from: "2026-09-30T12:00:00Z")!
    private func date(_ iso: String) -> Date { ISO8601DateFormatter().date(from: iso)! }
    private func result(_ json: String, exit: Int32 = 0) -> BrainLayerCLIResult { .init(terminationStatus: exit, stdout: json) }
    private func present(_ status: DriveAuthStatus, reconnecting: Bool = false, outcome: DriveAuthOutcome? = nil) -> DriveAuthPresentation {
        DriveAuthPresentation.derive(status: status, isReconnecting: reconnecting, lastOutcome: outcome, now: now,
                                     formatDate: { ISO8601DateFormatter().string(from: $0) })
    }

    // MARK: status contract (`--status --json`)

    func test_status_states_parse_from_the_cli_contract() {
        let valid = DriveAuthStatus.parse(result(#"{"state":"valid","reason":"","expires_at":"2026-10-05T09:00:00Z","days_left":4.9}"#))
        XCTAssertEqual(valid, .init(state: .valid, reason: nil, expiresAt: date("2026-10-05T09:00:00Z")))
        XCTAssertEqual(DriveAuthStatus.parse(result(#"{"state":"expiring","expires_at":"2026-09-30T17:30:00Z"}"#)).state, .expiring)
        XCTAssertEqual(DriveAuthStatus.parse(result(#"{"state":"missing","reason":"no token file"}"#)),
                       .init(state: .missing, reason: "no token file", expiresAt: nil))
        XCTAssertEqual(DriveAuthStatus.parse(result(#"{"state":"invalid","reason":"invalid_grant"}"#)).state, .invalid)
        // Anything outside the contract is an honest unknown with a reason.
        XCTAssertEqual(DriveAuthStatus.parse(nil), .init(state: .unknown, reason: "brainlayer CLI not found", expiresAt: nil))
        XCTAssertEqual(DriveAuthStatus.parse(result("No such command 'auth'.", exit: 2)),
                       .init(state: .unknown, reason: "brainlayer backup auth --status failed (exit 2)", expiresAt: nil))
        XCTAssertEqual(DriveAuthStatus.parse(result(#"{"state":"sparkly"}"#)).reason, "unrecognised Drive state \"sparkly\"")
    }

    func test_each_status_state_presents_what_to_do() {
        let valid = present(.init(state: .valid, reason: nil, expiresAt: date("2026-10-05T09:00:00Z")))
        XCTAssertEqual(valid.line, "Connected: renews by 2026-10-05T09:00:00Z")
        XCTAssertEqual(valid.tone, .connected)
        XCTAssertFalse(valid.showsReconnect)
        XCTAssertFalse(valid.needsAttention)

        let expiring = present(.init(state: .expiring, reason: nil, expiresAt: date("2026-09-30T17:30:00Z")))
        XCTAssertEqual(expiring.line, "Drive access expires in 6 h. Reconnect")
        XCTAssertEqual(expiring.tone, .expiring)
        XCTAssertTrue(expiring.showsReconnect && expiring.reconnectEnabled && expiring.needsAttention)

        let missing = present(.init(state: .missing, reason: "no token file", expiresAt: nil))
        XCTAssertEqual(missing.line, "Google Drive is not connected. Backups can't upload.")
        XCTAssertEqual(missing.detail, "no token file")
        XCTAssertEqual(missing.tone, .attention)
        XCTAssertTrue(missing.showsReconnect && missing.needsAttention)

        let invalid = present(.init(state: .invalid, reason: "invalid_grant", expiresAt: nil))
        XCTAssertEqual(invalid.line, "Drive access expired or was revoked. Backups can't upload.")
        XCTAssertTrue(invalid.showsReconnect && invalid.needsAttention)

        let unknown = present(.init(state: .unknown, reason: "brainlayer CLI not found", expiresAt: nil))
        XCTAssertEqual(unknown.line, "Drive status unknown — brainlayer CLI not found")
        XCTAssertFalse(unknown.showsReconnect, "no button that cannot work")
        XCTAssertFalse(unknown.needsAttention)
    }

    // MARK: reconnect contract (`--json`): the four outcomes

    func test_reconnect_outcomes_parse_from_the_cli_contract() {
        XCTAssertEqual(DriveAuthOutcome.parse(result(#"{"status":"ok","reason":""}"#)), .ok)
        XCTAssertEqual(DriveAuthOutcome.parse(result(#"{"status":"cancelled","reason":"consent denied"}"#, exit: 1)), .cancelled("consent denied"))
        XCTAssertEqual(DriveAuthOutcome.parse(result(#"{"status":"timeout","reason":"no redirect in 300 s"}"#, exit: 1)), .timeout("no redirect in 300 s"))
        XCTAssertEqual(DriveAuthOutcome.parse(result(#"{"status":"error","reason":"client file missing"}"#, exit: 1)), .error("client file missing"))
        XCTAssertEqual(DriveAuthOutcome.parse(nil), .error("brainlayer CLI not found"))
        XCTAssertEqual(DriveAuthOutcome.parse(result("Traceback …", exit: 1)), .error("brainlayer backup auth failed (exit 1)"))
    }

    func test_a_failed_reconnect_keeps_the_button_and_says_why() {
        let missing = DriveAuthStatus(state: .missing, reason: nil, expiresAt: nil)
        for (outcome, text) in [
            (DriveAuthOutcome.cancelled("consent denied"), "Reconnect was cancelled: consent denied"),
            (.timeout("no redirect in 300 s"), "Reconnect timed out: no redirect in 300 s"),
            (.error("client file missing"), "Reconnect failed: client file missing"),
        ] {
            let presentation = present(missing, outcome: outcome)
            XCTAssertEqual(presentation.detail, text)
            XCTAssertTrue(presentation.showsReconnect && presentation.reconnectEnabled)
        }
        let waiting = present(missing, reconnecting: true)
        XCTAssertEqual(waiting.detail, "Waiting for Google consent in your browser…")
        XCTAssertTrue(waiting.showsReconnect)
        XCTAssertFalse(waiting.reconnectEnabled, "one consent at a time")
    }

    // MARK: the model, with an injected runner

    private final class ScriptedRunner: BrainLayerCLIRunning, @unchecked Sendable {
        private let lock = NSLock()
        private var statuses: [BrainLayerCLIResult?]
        private let reconnectResult: BrainLayerCLIResult?
        private(set) var calls: [[String]] = []
        private(set) var ranOnMainThread = false
        init(statuses: [BrainLayerCLIResult?], reconnect: BrainLayerCLIResult?) {
            self.statuses = statuses
            reconnectResult = reconnect
        }
        func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult? {
            lock.withLock {
                calls.append(arguments)
                if Thread.isMainThread { ranOnMainThread = true }
                if arguments.contains("--status") { return statuses.isEmpty ? nil : statuses.removeFirst() }
                return reconnectResult
            }
        }
    }

    @MainActor
    func test_ok_reconnect_refreshes_to_connected_off_the_main_thread() async {
        let runner = ScriptedRunner(
            statuses: [result(#"{"state":"missing"}"#), result(#"{"state":"valid","expires_at":"2026-10-07T12:00:00Z"}"#)],
            reconnect: result(#"{"status":"ok"}"#)
        )
        let model = BrainBarDriveAuthModel(runner: runner, now: { [now] in now })
        await model.refreshStatus()
        XCTAssertEqual(model.status.state, .missing)
        await model.reconnect()
        XCTAssertEqual(model.lastOutcome, .ok)
        XCTAssertEqual(model.status.state, .valid)
        XCTAssertFalse(model.isReconnecting)
        XCTAssertEqual(runner.calls, [["backup", "auth", "--status", "--json"], ["backup", "auth", "--json"], ["backup", "auth", "--status", "--json"]])
        XCTAssertFalse(runner.ranOnMainThread, "the CLI never runs on the main thread")
    }

    @MainActor
    func test_every_failed_reconnect_outcome_keeps_the_state() async {
        for json in [#"{"status":"cancelled","reason":"denied"}"#, #"{"status":"timeout","reason":"t"}"#, #"{"status":"error","reason":"e"}"#] {
            let runner = ScriptedRunner(statuses: [result(#"{"state":"invalid"}"#)], reconnect: result(json, exit: 1))
            let model = BrainBarDriveAuthModel(runner: runner, now: { [now] in now })
            await model.refreshStatus()
            await model.reconnect()
            XCTAssertEqual(model.status.state, .invalid, json)
            XCTAssertNotEqual(model.lastOutcome, .ok, json)
            XCTAssertEqual(runner.calls.count, 2, "no status re-read after a failed reconnect: \(json)")
        }
    }

    @MainActor
    func test_an_unconfigured_model_never_runs_anything() async {
        let model = BrainBarDriveAuthModel(runner: nil)
        await model.refreshStatus()
        XCTAssertEqual(model.status.state, .unknown)
    }

    // MARK: never a token

    func test_a_reason_never_shows_a_token() {
        let leaky = DriveAuthStatus.parse(result(#"{"state":"invalid","reason":"refresh 1//0gAbCdEfGhIjKlMnOpQrStUvWxYz-_ failed for ya29.a0AfH6SMBxyz_ABC-123 with GOCSPX-abcdefghijklmnop"}"#))
        let text = [leaky.reason ?? "", present(leaky).detail ?? ""].joined(separator: " ")
        for secret in ["1//0gAbCd", "ya29.a0AfH6", "GOCSPX-abcdef"] {
            XCTAssertFalse(text.contains(secret), "\(secret) leaked: \(text)")
        }
        XCTAssertTrue(text.contains("[REDACTED"), text)
    }

    // MARK: finding the CLI (a GUI app has no shell PATH)

    func test_the_cli_is_resolved_explicitly() {
        let exists: (String) -> Bool = { ["/custom/brainlayer", "/opt/homebrew/opt/brainlayer/bin/brainlayer"].contains($0) }
        XCTAssertEqual(ProcessBrainLayerCLIRunner.resolveExecutable(environment: ["BRAINLAYER_CLI": "/custom/brainlayer"], isExecutable: exists), "/custom/brainlayer")
        XCTAssertEqual(ProcessBrainLayerCLIRunner.resolveExecutable(environment: [:], isExecutable: exists), "/opt/homebrew/opt/brainlayer/bin/brainlayer")
        XCTAssertNil(ProcessBrainLayerCLIRunner.resolveExecutable(environment: ["BRAINLAYER_CLI": "/missing"], isExecutable: { _ in false }))
    }
}
