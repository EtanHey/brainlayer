import Darwin
import Foundation

private func require(_ value: @autoclosure () -> Bool, _ message: String) {
    precondition(value(), message)
}

private final class InvocationBox: @unchecked Sendable {
    var invoked = false
}

private func identity(_ url: URL) -> BrainLayerBackupDestinationIdentity {
    var value = stat()
    precondition(lstat(url.path, &value) == 0)
    return BrainLayerBackupDestinationIdentity(stDev: Int64(value.st_dev), stIno: Int64(value.st_ino))
}

@MainActor
private func waitUntil(_ predicate: () -> Bool) async {
    let deadline = ContinuousClock.now + .seconds(5)
    while !predicate() {
        precondition(ContinuousClock.now < deadline, "coordinator did not complete")
        await Task.yield()
    }
}

@main
struct BackupCoordinatorProviderHarness {
    @MainActor
    static func main() async throws {
        let arguments = CommandLine.arguments
        precondition(arguments.count == 5, "usage: harness RECEIPT DESTINATION SOURCE_ROOT STATE")
        let localReceipt = URL(fileURLWithPath: arguments[1]).standardizedFileURL
        let destination = URL(fileURLWithPath: arguments[2]).standardizedFileURL
        let sourceRoot = URL(fileURLWithPath: arguments[3]).standardizedFileURL
        let stateURL = URL(fileURLWithPath: arguments[4]).standardizedFileURL
        var config = BrainLayerConfig.defaultConfig
        config.backupEnabled = true
        config.backupDestinationPath = destination.path
        config.backupDestinationIdentity = identity(destination)
        config.backupTranscriptRoots = [sourceRoot.path]
        config.backupTranscriptRootSpecs = [BrainLayerBackupSourceRootSpec(path: sourceRoot.path)]
        let output = try String(
            data: JSONSerialization.data(withJSONObject: [
                "status": "local_verified",
                "receipt": localReceipt.path,
                "remote_verified": false,
                "cleanup_permitted": false,
            ], options: [.sortedKeys]),
            encoding: .utf8
        ).unwrap()
        let coordinator = BrainLayerBackupCoordinator(
            commandRunner: BrainLayerBackupCommandRunner { _ in
                BrainLayerBackupCommandResult(terminationStatus: 0, output: output)
            },
            stateURL: stateURL
        )
        coordinator.startIfConfigured(config: config)
        await waitUntil {
            if case .providerObserved = coordinator.status { return true }
            return false
        }

        let providerReceipt = destination
            .appendingPathComponent(".brainlayer-provider")
            .appendingPathComponent("icloud-provider.receipt.json")
        require(FileManager.default.fileExists(atPath: providerReceipt.path), "managed provider receipt missing")
        let providerObject = try JSONSerialization.jsonObject(with: Data(contentsOf: providerReceipt)) as? [String: Any]
        require(providerObject?["remoteVerified"] as? Bool == false, "provider claimed remote verification")
        require(providerObject?["cleanupPermitted"] as? Bool == false, "provider authorized cleanup")
        require(providerObject?["state"] as? String == "staged", "local destination was not staged")

        let durableStateURL = stateURL.deletingLastPathComponent().appendingPathComponent("durable-state.json")
        let durableState = try JSONSerialization.data(withJSONObject: [
            "version": 1,
            "status": "local_verified",
            "receipt": localReceipt.path,
        ], options: [.sortedKeys])
        try durableState.write(to: durableStateURL)
        let invocations = InvocationBox()
        let scheduled = BrainLayerBackupCoordinator(
            commandRunner: BrainLayerBackupCommandRunner { _ in
                invocations.invoked = true
                return BrainLayerBackupCommandResult(terminationStatus: 99, output: "must not launch")
            },
            stateURL: stateURL.deletingLastPathComponent().appendingPathComponent("manual-unused.json"),
            scheduledStateURL: durableStateURL
        )
        scheduled.refreshScheduledStatus(config: config)
        await waitUntil {
            if case .providerObserved = scheduled.status { return true }
            return false
        }
        require(!invocations.invoked, "read-only scheduler reconciliation launched a backup")

        let changedDestination = destination.deletingLastPathComponent().appendingPathComponent("changed")
        try FileManager.default.createDirectory(at: changedDestination, withIntermediateDirectories: true)
        config.backupDestinationPath = changedDestination.path
        config.backupDestinationIdentity = identity(changedDestination)
        let failed = BrainLayerBackupCoordinator(
            commandRunner: BrainLayerBackupCommandRunner { _ in
                BrainLayerBackupCommandResult(terminationStatus: 9, output: "generated changed-destination failure")
            },
            stateURL: stateURL
        )
        failed.startIfConfigured(config: config)
        await waitUntil { failed.status == .failed(message: "generated changed-destination failure") }
        let persisted = try JSONDecoder().decode(
            BrainLayerBackupScheduleState.self,
            from: Data(contentsOf: stateURL)
        )
        require(persisted.lastSuccess == nil, "old destination success was reattributed")
        require(persisted.lastReceipt == nil, "old destination receipt was reattributed")
        require(persisted.lastStatus == "failed", "changed-destination failure status was ambiguous")
        print("BACKUP_COORDINATOR_PROVIDER_OK state=staged remote_verified=false cleanup_permitted=false scheduled_refresh=read_only changed_destination_history=cleared")
    }
}

private extension Optional {
    func unwrap(file: StaticString = #filePath, line: UInt = #line) throws -> Wrapped {
        guard let self else {
            throw NSError(
                domain: "BackupCoordinatorProviderHarness",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "unexpected nil at \(file):\(line)"]
            )
        }
        return self
    }
}
