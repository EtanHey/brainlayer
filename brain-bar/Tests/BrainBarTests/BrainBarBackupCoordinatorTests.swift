import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import BrainBar

private final class ArgumentsBox: @unchecked Sendable {
    var value: [String] = []
}

private struct CoordinatorFixture {
    let root: URL
    let destination: URL
    let receipt: URL
    let configURL: URL
    let config: BrainLayerConfig
}

private func digest(_ data: Data) -> String {
    Data(SHA256.hash(data: data)).map { String(format: "%02x", $0) }.joined()
}

private func identity(_ url: URL) -> BrainLayerBackupDestinationIdentity {
    var value = stat()
    precondition(lstat(url.path, &value) == 0)
    return BrainLayerBackupDestinationIdentity(stDev: Int64(value.st_dev), stIno: Int64(value.st_ino))
}

private func makeFixture(name: String = UUID().uuidString) throws -> CoordinatorFixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("backup-coordinator-\(name)")
    let destination = root.appendingPathComponent("selected")
    let generation = destination.appendingPathComponent(".brainlayer-backup-generated")
    let transcriptRoot = root.appendingPathComponent("transcripts")
    try FileManager.default.createDirectory(at: generation, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: transcriptRoot, withIntermediateDirectories: true)
    let db = Data("generated-db\n".utf8)
    let bundle = Data("generated-jsonl-bundle\n".utf8)
    try db.write(to: generation.appendingPathComponent("brainlayer.db"))
    try bundle.write(to: generation.appendingPathComponent("jsonl.bundle.tar"))
    let destinationIdentity = identity(destination)
    let complete: [String: Any] = [
        "status": "local_verified",
        "generation": generation.lastPathComponent,
        "destination_path": destination.standardizedFileURL.path,
        "destination_identity": ["st_dev": destinationIdentity.stDev, "st_ino": destinationIdentity.stIno],
        "db_snapshot": "brainlayer.db",
        "db_snapshot_bytes": db.count,
        "db_snapshot_sha256": digest(db),
        "jsonl_bundle": "jsonl.bundle.tar",
        "jsonl_bundle_bytes": bundle.count,
        "jsonl_bundle_sha256": digest(bundle),
        "local_verified": true,
        "remote_verified": false,
        "cleanup_permitted": false,
        "db_source_mode": "explicit_offline_db",
        "db_source_path": root.appendingPathComponent("source.db").path,
        "scheduled_offline_source_binding": [
            "version": 1,
            "path": root.appendingPathComponent("source.db").path,
            "st_dev": 11,
            "st_ino": 12,
        ],
    ]
    let receipt = generation.appendingPathComponent("receipt.complete.json")
    try JSONSerialization.data(withJSONObject: complete, options: [.sortedKeys]).write(to: receipt)

    var config = BrainLayerConfig.defaultConfig
    config.backupEnabled = true
    config.backupDestinationPath = destination.path
    config.backupDestinationIdentity = destinationIdentity
    config.backupTranscriptRoots = [transcriptRoot.path]
    config.backupTranscriptRootSpecs = [BrainLayerBackupSourceRootSpec(path: transcriptRoot.path)]
    return CoordinatorFixture(
        root: root,
        destination: destination,
        receipt: receipt,
        configURL: root.appendingPathComponent("brainlayer.env"),
        config: config
    )
}

private func output(for fixture: CoordinatorFixture) -> String {
    let data = try! JSONSerialization.data(withJSONObject: [
        "status": "local_verified",
        "receipt": fixture.receipt.path,
        "db_source_mode": "explicit_offline_db",
        "db_source_path": fixture.root.appendingPathComponent("source.db").path,
        "remote_verified": false,
        "cleanup_permitted": false,
    ], options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

@MainActor
private func waitUntil(_ predicate: () -> Bool) async throws {
    let deadline = ContinuousClock.now + .seconds(2)
    while !predicate() {
        if ContinuousClock.now >= deadline {
            XCTFail("coordinator did not reach expected state")
            return
        }
        await Task.yield()
    }
}

@MainActor
final class BrainBarBackupCoordinatorTests: XCTestCase {
    func testActualLocalReceiptFeedsProviderAtManagedDestinationPath() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let received = ArgumentsBox()
        let runner = BrainLayerBackupCommandRunner { arguments in
            received.value = arguments
            return BrainLayerBackupCommandResult(terminationStatus: 0, output: output(for: fixture))
        }
        let stateURL = fixture.root.appendingPathComponent("state.json")
        let coordinator = BrainLayerBackupCoordinator(commandRunner: runner, stateURL: stateURL)

        coordinator.startIfConfigured(config: fixture.config, configURL: fixture.configURL)
        try await waitUntil {
            if case .providerObserved = coordinator.status { return true }
            return false
        }

        let providerReceipt = fixture.destination
            .appendingPathComponent(".brainlayer-provider")
            .appendingPathComponent("icloud-provider.receipt.json")
        XCTAssertEqual(
            coordinator.status,
            .providerObserved(
                localReceipt: fixture.receipt.path,
                providerReceipt: providerReceipt.path,
                state: .staged
            )
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: providerReceipt.path))
        XCTAssertEqual(Array(received.value[0...2]), ["-I", "-B", "-m"])
        XCTAssertTrue(received.value.contains("brainlayer.backup_run_once"))
        XCTAssertTrue(received.value.contains("--transcript-root-spec"))
    }

    func testExitZeroWithFailurePayloadDoesNotInvokeProvider() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let coordinator = BrainLayerBackupCoordinator(
            commandRunner: BrainLayerBackupCommandRunner { _ in
                BrainLayerBackupCommandResult(
                    terminationStatus: 0,
                    output: #"{"status":"failed","receipt":"/tmp/incomplete.json"}"#
                )
            },
            stateURL: fixture.root.appendingPathComponent("state.json")
        )
        coordinator.startIfConfigured(config: fixture.config)
        try await waitUntil {
            if case .failed = coordinator.status { return true }
            return false
        }
        XCTAssertEqual(coordinator.status, .failed(message: #"{"status":"failed","receipt":"/tmp/incomplete.json"}"#))
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: fixture.destination.appendingPathComponent(".brainlayer-provider").path
            )
        )
    }

    func testBackupStatusTitlesUsePlainLanguageWithoutReceiptPaths() {
        XCTAssertEqual(
            BrainLayerBackupCoordinator.Status.providerObserved(
                localReceipt: "/private/generated/receipt.complete.json",
                providerReceipt: "/private/generated/icloud-provider.receipt.json",
                state: .staged
            ).title,
            "Saved locally"
        )
        XCTAssertEqual(
            BrainLayerBackupCoordinator.Status.providerObserved(
                localReceipt: "/private/generated/receipt.complete.json",
                providerReceipt: "/private/generated/icloud-provider.receipt.json",
                state: .uploadPending
            ).title,
            "Waiting for iCloud upload"
        )
        XCTAssertEqual(
            BrainLayerBackupCoordinator.Status.providerObserved(
                localReceipt: "/private/generated/receipt.complete.json",
                providerReceipt: "/private/generated/icloud-provider.receipt.json",
                state: .uploadedOSReported
            ).title,
            "iCloud reports upload complete; not independently verified"
        )
        XCTAssertEqual(
            BrainLayerBackupCoordinator.Status.schedulerObserved(
                status: "refused",
                detail: "saved source changed"
            ).title,
            "Scheduled backup needs attention — saved source changed"
        )
    }

    func testInvalidConfigDoesNotInvokeChild() async throws {
        let invoked = ArgumentsBox()
        let coordinator = BrainLayerBackupCoordinator(commandRunner: BrainLayerBackupCommandRunner { _ in
            invoked.value = ["invoked"]
            return BrainLayerBackupCommandResult(terminationStatus: 0, output: "")
        })
        coordinator.startIfConfigured(config: .defaultConfig)
        XCTAssertTrue(invoked.value.isEmpty)
        XCTAssertEqual(
            coordinator.status,
            .failed(message: "Backup needs a saved folder, filesystem identity, and reviewed source-root specs.")
        )
    }

    func testSuccessfulProviderRunPersistsLocalAndProviderState() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let instant = Date(timeIntervalSince1970: 1_800_000_000)
        let stateURL = fixture.root.appendingPathComponent("state.json")
        let coordinator = BrainLayerBackupCoordinator(
            commandRunner: BrainLayerBackupCommandRunner { _ in
                BrainLayerBackupCommandResult(terminationStatus: 0, output: output(for: fixture))
            },
            stateURL: stateURL,
            now: { instant }
        )
        coordinator.startIfConfigured(config: fixture.config)
        try await waitUntil {
            if case .providerObserved = coordinator.status { return true }
            return false
        }

        let state = try JSONDecoder().decode(BrainLayerBackupScheduleState.self, from: Data(contentsOf: stateURL))
        XCTAssertEqual(state.lastAttempt, instant)
        XCTAssertEqual(state.lastSuccess, instant)
        XCTAssertEqual(state.nextDue, instant.addingTimeInterval(24 * 60 * 60))
        XCTAssertEqual(state.lastStatus, "provider_staged")
        XCTAssertEqual(state.lastReceipt, fixture.receipt.path)
    }

    func testDestinationChangeClearsPriorSuccessBeforeFailedRun() async throws {
        let oldFixture = try makeFixture(name: "old-(UUID().uuidString)")
        let newFixture = try makeFixture(name: "new-(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: oldFixture.root)
            try? FileManager.default.removeItem(at: newFixture.root)
        }
        let stateURL = newFixture.root.appendingPathComponent("state.json")
        let oldState = BrainLayerBackupScheduleState(
            schema: 1,
            lastAttempt: Date(timeIntervalSince1970: 10),
            lastSuccess: Date(timeIntervalSince1970: 20),
            nextDue: Date(timeIntervalSince1970: 30),
            lastStatus: "provider_staged",
            lastReceipt: oldFixture.receipt.path,
            destinationPath: oldFixture.destination.path,
            destinationDevice: oldFixture.config.backupDestinationIdentity?.stDev,
            destinationInode: oldFixture.config.backupDestinationIdentity?.stIno
        )
        try JSONEncoder().encode(oldState).write(to: stateURL)
        let coordinator = BrainLayerBackupCoordinator(
            commandRunner: BrainLayerBackupCommandRunner { _ in
                BrainLayerBackupCommandResult(terminationStatus: 7, output: "generated failure")
            },
            stateURL: stateURL
        )

        coordinator.startIfConfigured(config: newFixture.config)
        try await waitUntil { coordinator.status == .failed(message: "generated failure") }
        let state = try JSONDecoder().decode(BrainLayerBackupScheduleState.self, from: Data(contentsOf: stateURL))
        XCTAssertNil(state.lastSuccess)
        XCTAssertNil(state.lastReceipt)
        XCTAssertEqual(state.lastStatus, "failed")
        XCTAssertEqual(state.destinationPath, newFixture.destination.path)
    }

    func testSameDestinationFailureRetainsExplicitHistoricalSuccess() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let stateURL = fixture.root.appendingPathComponent("state.json")
        let success = Date(timeIntervalSince1970: 20)
        let state = BrainLayerBackupScheduleState(
            schema: 1,
            lastAttempt: Date(timeIntervalSince1970: 10),
            lastSuccess: success,
            nextDue: Date(timeIntervalSince1970: 30),
            lastStatus: "provider_staged",
            lastReceipt: fixture.receipt.path,
            destinationPath: fixture.destination.path,
            destinationDevice: fixture.config.backupDestinationIdentity?.stDev,
            destinationInode: fixture.config.backupDestinationIdentity?.stIno
        )
        try JSONEncoder().encode(state).write(to: stateURL)
        let coordinator = BrainLayerBackupCoordinator(
            commandRunner: BrainLayerBackupCommandRunner { _ in
                BrainLayerBackupCommandResult(terminationStatus: 9, output: "same destination failure")
            },
            stateURL: stateURL
        )

        coordinator.startIfConfigured(config: fixture.config)
        try await waitUntil { coordinator.status == .failed(message: "same destination failure") }
        let persisted = try JSONDecoder().decode(
            BrainLayerBackupScheduleState.self,
            from: Data(contentsOf: stateURL)
        )
        XCTAssertEqual(persisted.lastSuccess, success)
        XCTAssertEqual(persisted.lastReceipt, fixture.receipt.path)
        XCTAssertEqual(persisted.lastStatus, "failed_with_historical_local_success")
    }

    func testDurableSchedulerRefreshObservesReceiptWithoutLaunchingBackup() async throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let calls = ArgumentsBox()
        let scheduledState = fixture.root.appendingPathComponent("durable-state.json")
        let providerReceipt = fixture.destination
            .appendingPathComponent(".brainlayer-provider")
            .appendingPathComponent("icloud-provider.receipt.json")
        try JSONSerialization.data(withJSONObject: [
            "version": 1,
            "status": "local_verified",
            "receipt": fixture.receipt.path,
            "next_due": "2099-01-01T00:00:00+00:00",
        ], options: [.sortedKeys]).write(to: scheduledState)
        let coordinator = BrainLayerBackupCoordinator(
            commandRunner: BrainLayerBackupCommandRunner { arguments in
                calls.value = arguments
                return BrainLayerBackupCommandResult(terminationStatus: 99, output: "must not launch")
            },
            stateURL: fixture.root.appendingPathComponent("manual-state.json"),
            scheduledStateURL: scheduledState
        )

        coordinator.refreshScheduledStatus(config: fixture.config)
        try await waitUntil {
            if case .providerObserved = coordinator.status { return true }
            return false
        }

        XCTAssertTrue(calls.value.isEmpty)
        XCTAssertEqual(
            coordinator.status,
            .providerObserved(
                localReceipt: fixture.receipt.path,
                providerReceipt: fixture.destination
                    .appendingPathComponent(".brainlayer-provider")
                    .appendingPathComponent("icloud-provider.receipt.json").path,
                state: .staged
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: fixture.root.appendingPathComponent("manual-state.json").path
            )
        )

        let initialProviderData = try Data(contentsOf: providerReceipt)
        coordinator.refreshScheduledStatus(config: fixture.config)
        XCTAssertEqual(try Data(contentsOf: providerReceipt), initialProviderData)

        coordinator.refreshScheduledStatus(config: fixture.config, forceProviderRetry: true)
        try await waitUntil {
            if case .providerObserved = coordinator.status { return true }
            return false
        }
        XCTAssertNotEqual(try Data(contentsOf: providerReceipt), initialProviderData)

        var staleConfig = fixture.config
        staleConfig.backupDestinationIdentity = BrainLayerBackupDestinationIdentity(
            stDev: fixture.config.backupDestinationIdentity!.stDev,
            stIno: fixture.config.backupDestinationIdentity!.stIno + 1
        )
        coordinator.refreshScheduledStatus(config: staleConfig)
        XCTAssertEqual(
            coordinator.status,
            .schedulerObserved(
                status: "destination_changed",
                detail: "saved destination identity is not current"
            )
        )
    }

    func testDurableSchedulerNonSuccessIsVisibleAndNeverLaunchesBackup() throws {
        let fixture = try makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }
        let calls = ArgumentsBox()
        let scheduledState = fixture.root.appendingPathComponent("durable-state.json")
        try JSONSerialization.data(withJSONObject: [
            "version": 1,
            "status": "refused",
            "reason": "scheduled offline source changed",
        ], options: [.sortedKeys]).write(to: scheduledState)
        let coordinator = BrainLayerBackupCoordinator(
            commandRunner: BrainLayerBackupCommandRunner { arguments in
                calls.value = arguments
                return BrainLayerBackupCommandResult(terminationStatus: 99, output: "must not launch")
            },
            stateURL: fixture.root.appendingPathComponent("manual-state.json"),
            scheduledStateURL: scheduledState
        )

        coordinator.refreshScheduledStatus(config: fixture.config)

        XCTAssertTrue(calls.value.isEmpty)
        XCTAssertEqual(
            coordinator.status,
            .schedulerObserved(status: "refused", detail: "scheduled offline source changed")
        )
    }

}
