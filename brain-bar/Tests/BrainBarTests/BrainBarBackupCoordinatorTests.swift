import XCTest
@testable import BrainBar

private final class ArgumentsBox: @unchecked Sendable {
    var value: [String] = []
}

@MainActor
final class BrainBarBackupCoordinatorTests: XCTestCase {
    func testConfiguredRunUsesPinnedModuleAndSavedRoots() async throws {
        let received = ArgumentsBox()
        let runner = BrainLayerBackupCommandRunner { arguments in
            received.value = arguments
            return BrainLayerBackupCommandResult(
                terminationStatus: 0,
                output: #"{"status":"local_verified","receipt":"/tmp/receipt.complete.json"}"#
            )
        }
        let coordinator = BrainLayerBackupCoordinator(commandRunner: runner)
        var config = BrainLayerConfig.defaultConfig
        config.backupEnabled = true
        config.backupDestinationPath = "/tmp/backups"
        config.backupDestinationIdentity = BrainLayerBackupDestinationIdentity(stDev: 1, stIno: 2)
        config.backupTranscriptRoots = ["/tmp/projects"]

        coordinator.startIfConfigured(config: config, configURL: URL(fileURLWithPath: "/tmp/brainlayer.env"))
        try await Task.sleep(for: .milliseconds(20))

        XCTAssertEqual(coordinator.status, .localVerified(receipt: "/tmp/receipt.complete.json"))
        XCTAssertEqual(received.value.first, "-I")
        XCTAssertEqual(Array(received.value[2...5]), ["brainlayer.backup_run_once", "--config", "/tmp/brainlayer.env", "--generation-id"])
        XCTAssertTrue(received.value.contains("--transcript-root"))
    }

    func testExitZeroWithFailurePayloadDoesNotPublishVerified() async throws {
        let coordinator = BrainLayerBackupCoordinator(commandRunner: BrainLayerBackupCommandRunner { _ in
            BrainLayerBackupCommandResult(
                terminationStatus: 0,
                output: #"{"status":"failed","receipt":"/tmp/incomplete.json"}"#
            )
        })
        var config = BrainLayerConfig.defaultConfig
        config.backupEnabled = true
        config.backupDestinationPath = "/tmp/backups"
        config.backupDestinationIdentity = BrainLayerBackupDestinationIdentity(stDev: 1, stIno: 2)
        config.backupTranscriptRoots = ["/tmp/projects"]

        coordinator.startIfConfigured(config: config)
        try await Task.sleep(for: .milliseconds(20))
        XCTAssertEqual(coordinator.status, .failed(message: #"{"status":"failed","receipt":"/tmp/incomplete.json"}"#))
    }

    func testInvalidConfigDoesNotInvokeChild() async throws {
        let invoked = ArgumentsBox()
        let coordinator = BrainLayerBackupCoordinator(commandRunner: BrainLayerBackupCommandRunner { _ in
            invoked.value = ["invoked"]
            return BrainLayerBackupCommandResult(terminationStatus: 0, output: "")
        })
        coordinator.startIfConfigured(config: .defaultConfig)
        try await Task.sleep(for: .milliseconds(20))
        XCTAssertTrue(invoked.value.isEmpty)
        XCTAssertEqual(coordinator.status, .failed(message: "Backup needs a saved folder, filesystem identity, and transcript roots."))
    }
    func testSuccessfulRunPersistsDailyScheduleState() async throws {
        let stateURL = FileManager.default.temporaryDirectory.appendingPathComponent("backup-state-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: stateURL) }
        let instant = Date(timeIntervalSince1970: 1_800_000_000)
        let runner = BrainLayerBackupCommandRunner { _ in
            BrainLayerBackupCommandResult(
                terminationStatus: 0,
                output: #"{"status":"local_verified","receipt":"/tmp/receipt"}"#
            )
        }
        let coordinator = BrainLayerBackupCoordinator(commandRunner: runner, stateURL: stateURL, now: { instant })
        var config = BrainLayerConfig.defaultConfig
        config.backupEnabled = true
        config.backupDestinationPath = "/tmp/backups"
        config.backupDestinationIdentity = BrainLayerBackupDestinationIdentity(stDev: 1, stIno: 2)
        config.backupTranscriptRoots = ["/tmp/projects"]

        coordinator.startIfConfigured(config: config)
        try await Task.sleep(for: .milliseconds(20))
        let state = try JSONDecoder().decode(BrainLayerBackupScheduleState.self, from: Data(contentsOf: stateURL))
        XCTAssertEqual(state.lastAttempt, instant)
        XCTAssertEqual(state.lastSuccess, instant)
        XCTAssertEqual(state.nextDue, instant.addingTimeInterval(24 * 60 * 60))
        XCTAssertEqual(state.lastStatus, "local_verified")
    }

}
