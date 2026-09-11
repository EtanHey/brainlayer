import CryptoKit
import Darwin
import Foundation

struct BrainLayerBackupCommandResult: Equatable, Sendable {
    let terminationStatus: Int32
    let output: String
}

private final class BrainLayerBackupProcessController: @unchecked Sendable {
    private static let outputLimit = 64 * 1024
    private let arguments: [String]
    private let executableURL: URL
    private let timeout: TimeInterval
    private let lock = NSLock()
    private var process: Process?
    private var continuation: CheckedContinuation<BrainLayerBackupCommandResult, Never>?
    private var output = Data()
    private var finished = false
    private var cancelRequested = false

    init(arguments: [String], executableURL: URL, timeout: TimeInterval) {
        self.arguments = arguments
        self.executableURL = executableURL
        self.timeout = timeout
    }

    func run() async -> BrainLayerBackupCommandResult {
        await withCheckedContinuation { continuation in
            lock.lock()
            self.continuation = continuation
            lock.unlock()
            start()
        }
    }

    func cancel() {
        lock.lock()
        cancelRequested = true
        let process = self.process
        lock.unlock()
        if let process, process.isRunning { terminateOwned(process) }
    }

    private func start() {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = executableURL
        process.arguments = arguments
        process.standardOutput = pipe
        process.standardError = pipe
        lock.lock()
        self.process = process
        lock.unlock()
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            self?.append(data)
        }
        process.terminationHandler = { [weak self] process in
            pipe.fileHandleForReading.readabilityHandler = nil
            let remaining = pipe.fileHandleForReading.readDataToEndOfFile()
            self?.append(remaining)
            self?.finish(BrainLayerBackupCommandResult(
                terminationStatus: process.terminationStatus,
                output: self?.readOutput() ?? ""
            ))
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + self.timeout) { [weak self, weak process] in
            guard let process, process.isRunning else { return }
            self?.append(Data("backup process timed out after configured deadline\n".utf8))
            self?.terminateOwned(process)
        }
        do {
            try process.run()
            lock.lock()
            let shouldCancel = cancelRequested
            lock.unlock()
            if shouldCancel { terminateOwned(process) }
        } catch {
            finish(BrainLayerBackupCommandResult(terminationStatus: 127, output: "could not launch pinned BrainLayer Python: \(error)"))
        }
    }

    private func terminateOwned(_ process: Process) {
        process.terminate()
        let pid = process.processIdentifier
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
            if process.isRunning { _ = Darwin.kill(pid, SIGKILL) }
        }
    }

    private func append(_ data: Data) {
        guard !data.isEmpty else { return }
        lock.lock()
        let remaining = max(0, Self.outputLimit - output.count)
        if remaining > 0 { output.append(data.prefix(remaining)) }
        lock.unlock()
    }

    private func readOutput() -> String {
        lock.lock()
        defer { lock.unlock() }
        return String(decoding: output, as: UTF8.self)
    }

    private func finish(_ result: BrainLayerBackupCommandResult) {
        lock.lock()
        guard !finished else { lock.unlock(); return }
        finished = true
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume(returning: result)
    }
}

struct BrainLayerBackupCommandRunner: Sendable {
    let run: @Sendable ([String]) async -> BrainLayerBackupCommandResult

    init(run: @escaping @Sendable ([String]) async -> BrainLayerBackupCommandResult) {
        self.run = run
    }

    init(executableURL: URL = URL(fileURLWithPath: "/opt/homebrew/opt/brainlayer/libexec/venv/bin/python"), timeout: TimeInterval = 30 * 60) {
        self.run = { arguments in
            await Self.execute(arguments: arguments, executableURL: executableURL, timeout: timeout)
        }
    }

    private static func execute(
        arguments: [String], executableURL: URL, timeout: TimeInterval
    ) async -> BrainLayerBackupCommandResult {
        let controller = BrainLayerBackupProcessController(arguments: arguments, executableURL: executableURL, timeout: timeout)
        return await withTaskCancellationHandler(operation: {
            await controller.run()
        }, onCancel: {
            controller.cancel()
        })
    }
}

struct BrainLayerBackupScheduleState: Codable, Equatable, Sendable {
    let schema: Int
    var lastAttempt: Date?
    var lastSuccess: Date?
    var nextDue: Date?
    var lastStatus: String
    var lastReceipt: String?
    var destinationPath: String?
    var destinationDevice: Int64?
    var destinationInode: Int64?

    static let empty = BrainLayerBackupScheduleState(
        schema: 1, lastAttempt: nil, lastSuccess: nil, nextDue: nil, lastStatus: "never", lastReceipt: nil,
        destinationPath: nil, destinationDevice: nil, destinationInode: nil
    )
}

@MainActor
final class BrainLayerBackupCoordinator: ObservableObject {
    enum Status: Equatable, Sendable {
        case idle
        case running
        case localVerified(receipt: String)
        case schedulerObserved(status: String, detail: String?)
        case providerObserved(
            localReceipt: String,
            providerReceipt: String,
            state: BrainBarICloudBackupState
        )
        case failed(message: String)

        var title: String {
            switch self {
            case .idle: "Not run"
            case .running: "Running"
            case .localVerified: "Saved locally"
            case let .schedulerObserved(status, detail):
                Self.schedulerTitle(status: status, detail: detail)
            case let .providerObserved(_, _, state):
                switch state {
                case .staged:
                    "Saved locally"
                case .uploadPending:
                    "Waiting for iCloud upload"
                case .uploadedOSReported:
                    "iCloud reports upload complete; not independently verified"
                case .failed:
                    "iCloud provider check failed"
                }
            case let .failed(message): "Backup failed — \(message)"
            }
        }

        private static func schedulerTitle(status: String, detail: String?) -> String {
            let title = switch status {
            case "disabled": "Scheduled backup disabled"
            case "not_run": "Scheduled backup has not run"
            case "not_due": "Scheduled backup waiting for its next run"
            case "busy": "Scheduled backup is already running"
            case "refused": "Scheduled backup needs attention"
            case "failed": "Scheduled backup failed"
            case "destination_changed": "Backup destination changed"
            default: "Scheduled backup status unavailable"
            }
            guard let detail, !detail.isEmpty else { return title }
            return "\(title) — \(detail)"
        }
    }

    static let shared = BrainLayerBackupCoordinator()
    @Published private(set) var status: Status = .idle
    private static let statusMessageLimit = 4 * 1024
    private static let scheduleInterval: TimeInterval = 24 * 60 * 60
    private static let failureBackoff: TimeInterval = 60 * 60
    private let commandRunner: BrainLayerBackupCommandRunner
    private let provider: BrainBarICloudBackupProvider
    private let stateURL: URL
    private let scheduledStateURL: URL
    private let now: @Sendable () -> Date
    private struct ScheduledObservationKey: Equatable {
        let stateSHA256: String
        let receipt: String
        let destinationPath: String
        let destinationIdentity: BrainLayerBackupDestinationIdentity
    }

    private var task: Task<Void, Never>?
    private var activeRunID: UUID?
    private var lastScheduledObservationKey: ScheduledObservationKey?
    private var lastScheduledObservationStatus: Status?
    private var statusObservers: [UUID: @MainActor (Status) -> Void] = [:]

    init(
        commandRunner: BrainLayerBackupCommandRunner = BrainLayerBackupCommandRunner(),
        provider: BrainBarICloudBackupProvider = BrainBarICloudBackupProvider(),
        stateURL: URL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".local/share/brainlayer/backup-state.json"),
        scheduledStateURL: URL = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent(".local/share/brainlayer/durable-backup-schedule-state.json"),
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.commandRunner = commandRunner
        self.provider = provider
        self.stateURL = stateURL
        self.scheduledStateURL = scheduledStateURL
        self.now = now
    }

    deinit {
        task?.cancel()
    }

    @discardableResult
    func observeStatus(_ observer: @escaping @MainActor (Status) -> Void) -> UUID {
        let id = UUID()
        statusObservers[id] = observer
        observer(status)
        return id
    }

    func removeStatusObserver(_ id: UUID) {
        statusObservers.removeValue(forKey: id)
    }

    private func publish(_ nextStatus: Status, to callback: @escaping @MainActor (Status) -> Void) {
        status = nextStatus
        for observer in statusObservers.values { observer(nextStatus) }
        callback(nextStatus)
    }

    /// Compatibility entry point used by the app launch path. The durable Python
    /// scheduler is the sole automatic trigger; this method only reconciles its
    /// persisted result and observes an already completed local receipt.
    func startScheduledIfDue(
        configURL: URL = BrainLayerConfigStore.defaultConfigURL(),
        forceProviderRetry: Bool = false
    ) {
        do {
            let config = try BrainLayerConfigStore(configURL: configURL).loadDocument().config
            refreshScheduledStatus(config: config, forceProviderRetry: forceProviderRetry)
        } catch {
            publish(.failed(message: "Could not load backup configuration: \(error)"), to: { _ in })
        }
    }

    func refreshScheduledStatus(config: BrainLayerConfig, forceProviderRetry: Bool = false) {
        guard task == nil else { return }
        guard config.backupEnabled else {
            publish(.schedulerObserved(status: "disabled", detail: "backup disabled"), to: { _ in })
            return
        }
        let object: [String: Any]
        let stateSHA256: String
        do {
            let handle = try FileHandle(forReadingFrom: scheduledStateURL)
            defer { try? handle.close() }
            let data = try handle.read(upToCount: Self.statusMessageLimit + 1) ?? Data()
            guard data.count <= Self.statusMessageLimit,
                  let decoded = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let version = decoded["version"] as? NSNumber,
                  CFGetTypeID(version) != CFBooleanGetTypeID(),
                  version.intValue == 1,
                  let status = decoded["status"] as? String,
                  !status.isEmpty else {
                throw NSError(
                    domain: "BrainLayerBackupSchedule",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "malformed durable scheduler state"]
                )
            }
            object = decoded
            stateSHA256 = Data(SHA256.hash(data: data))
                .map { String(format: "%02x", $0) }
                .joined()
        } catch CocoaError.fileReadNoSuchFile {
            publish(.schedulerObserved(status: "not_run", detail: nil), to: { _ in })
            return
        } catch {
            publish(.failed(message: "Could not read durable scheduler state: \(error)"), to: { _ in })
            return
        }

        guard let scheduledStatus = object["status"] as? String else {
            publish(.failed(message: "Durable scheduler state lost its status"), to: { _ in })
            return
        }
        let receipt = object["receipt"] as? String
        if ["local_verified", "not_due"].contains(scheduledStatus),
           let receipt,
           !receipt.isEmpty,
           let destinationPath = config.backupDestinationPath,
           let expectedIdentity = config.backupDestinationIdentity {
            guard let currentIdentity = BrainLayerBackupDestinationIdentity(path: destinationPath),
                  currentIdentity == expectedIdentity else {
                publish(
                    .schedulerObserved(
                        status: "destination_changed",
                        detail: "saved destination identity is not current"
                    ),
                    to: { _ in }
                )
                return
            }
            let key = ScheduledObservationKey(
                stateSHA256: stateSHA256,
                receipt: receipt,
                destinationPath: URL(fileURLWithPath: destinationPath).standardizedFileURL.path,
                destinationIdentity: currentIdentity
            )
            if !forceProviderRetry,
               key == lastScheduledObservationKey,
               let cached = lastScheduledObservationStatus {
                publish(cached, to: { _ in })
                return
            }
            observeScheduledReceipt(receipt, destinationPath: destinationPath, observationKey: key)
            return
        }
        let detail = (object["reason"] as? String) ?? (object["error"] as? String)
        publish(.schedulerObserved(status: scheduledStatus, detail: detail), to: { _ in })
    }

    private func observeScheduledReceipt(
        _ receipt: String,
        destinationPath: String,
        observationKey: ScheduledObservationKey
    ) {
        publish(.running, to: { _ in })
        let runID = UUID()
        activeRunID = runID
        let localReceiptURL = URL(fileURLWithPath: receipt).standardizedFileURL
        let destinationURL = URL(fileURLWithPath: destinationPath).standardizedFileURL
        let providerReceiptURL = destinationURL
            .appendingPathComponent(".brainlayer-provider", isDirectory: true)
            .appendingPathComponent("icloud-provider.receipt.json", isDirectory: false)
        task = Task {
            do {
                let providerReceipt = try await provider.process(BrainBarICloudBackupInput(
                    generationDirectory: localReceiptURL.deletingLastPathComponent(),
                    selectedDestination: destinationURL,
                    providerReceiptURL: providerReceiptURL
                ))
                try Task.checkCancellation()
                guard !providerReceipt.remoteVerified, !providerReceipt.cleanupPermitted else {
                    throw NSError(
                        domain: "BrainLayerBackupProvider",
                        code: 1,
                        userInfo: [NSLocalizedDescriptionKey: "provider attempted to authorize unverified cleanup"]
                    )
                }
                let observedStatus: Status
                if providerReceipt.state == .failed {
                    observedStatus = .failed(
                        message: providerReceipt.failureReason ?? "provider verification failed"
                    )
                } else {
                    observedStatus = .providerObserved(
                        localReceipt: receipt,
                        providerReceipt: providerReceiptURL.path,
                        state: providerReceipt.state
                    )
                }
                lastScheduledObservationKey = observationKey
                lastScheduledObservationStatus = observedStatus
                publish(observedStatus, to: { _ in })
            } catch is CancellationError {
                if activeRunID == runID { task = nil; activeRunID = nil }
                return
            } catch {
                let failedStatus = Status.failed(message: "Provider verification failed: \(error)")
                lastScheduledObservationKey = observationKey
                lastScheduledObservationStatus = failedStatus
                publish(failedStatus, to: { _ in })
            }
            if activeRunID == runID { task = nil; activeRunID = nil }
        }
    }

    private func loadScheduleState() throws -> BrainLayerBackupScheduleState {
        guard FileManager.default.fileExists(atPath: stateURL.path) else { return .empty }
        return try JSONDecoder().decode(BrainLayerBackupScheduleState.self, from: Data(contentsOf: stateURL))
    }

    private func saveScheduleState(_ state: BrainLayerBackupScheduleState) throws {
        let directory = stateURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let temporary = directory.appendingPathComponent(".backup-state.\(UUID().uuidString).tmp")
        try JSONEncoder().encode(state).write(to: temporary, options: .atomic)
        if FileManager.default.fileExists(atPath: stateURL.path) {
            _ = try FileManager.default.replaceItemAt(
                stateURL, withItemAt: temporary, backupItemName: nil, options: .usingNewMetadataOnly
            )
        } else {
            try FileManager.default.moveItem(at: temporary, to: stateURL)
        }
    }

    func startIfConfigured(
        config: BrainLayerConfig,
        configURL: URL = BrainLayerConfigStore.defaultConfigURL(),
        onStatus: @escaping @MainActor (Status) -> Void = { _ in }
    ) {
        guard config.backupEnabled,
              config.backupDestinationPath != nil,
              config.backupDestinationIdentity != nil,
              !config.backupTranscriptRoots.isEmpty,
              !config.backupTranscriptRootSpecs.isEmpty else {
            publish(.failed(message: "Backup needs a saved folder, filesystem identity, and reviewed source-root specs."), to: onStatus)
            return
        }
        var scheduleState: BrainLayerBackupScheduleState
        do {
            scheduleState = try loadScheduleState()
            guard scheduleState.schema == 1 else { throw NSError(domain: "BrainLayerBackupSchedule", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unsupported backup schedule schema \(scheduleState.schema)"]) }
        } catch {
            publish(.failed(message: "Could not load backup schedule: \(error)"), to: onStatus)
            return
        }
        let destinationChanged =
            scheduleState.destinationPath != config.backupDestinationPath
            || scheduleState.destinationDevice != config.backupDestinationIdentity?.stDev
            || scheduleState.destinationInode != config.backupDestinationIdentity?.stIno
        if destinationChanged {
            scheduleState.lastSuccess = nil
            scheduleState.lastReceipt = nil
            scheduleState.nextDue = nil
        }
        scheduleState.lastAttempt = now()
        scheduleState.lastStatus = "running"
        scheduleState.destinationPath = config.backupDestinationPath
        scheduleState.destinationDevice = config.backupDestinationIdentity?.stDev
        scheduleState.destinationInode = config.backupDestinationIdentity?.stIno
        do {
            try saveScheduleState(scheduleState)
        } catch {
            publish(.failed(message: "Could not persist backup schedule: \(error)"), to: onStatus)
            return
        }
        let previous = task
        previous?.cancel()
        publish(.running, to: onStatus)
        let runID = UUID()
        activeRunID = runID
        task = Task { [commandRunner] in
            if let previous { await previous.value }
            guard !Task.isCancelled else {
                if activeRunID == runID { task = nil; activeRunID = nil }
                return
            }
            var arguments = [
                "-I", "-B", "-m", "brainlayer.backup_run_once",
                "--config", configURL.path,
                "--generation-id", UUID().uuidString,
            ]
            for spec in config.backupTranscriptRootSpecs {
                let object: [String: Any] = [
                    "path": spec.path,
                    "include_globs": spec.includeGlobs,
                    "exclude_globs": spec.excludeGlobs,
                    "source_class": spec.sourceClass,
                ]
                guard let data = try? JSONSerialization.data(withJSONObject: object),
                      let encoded = String(data: data, encoding: .utf8) else { continue }
                arguments += ["--transcript-root-spec", encoded]
            }
            let result = await commandRunner.run(arguments)
            guard !Task.isCancelled else {
                if activeRunID == runID { task = nil; activeRunID = nil }
                return
            }
            if result.terminationStatus == 0,
               let payload = result.output.data(using: .utf8),
               let object = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
               object["status"] as? String == "local_verified",
               let receipt = object["receipt"] as? String,
               !receipt.isEmpty,
               let destinationPath = config.backupDestinationPath {
                let localReceiptURL = URL(fileURLWithPath: receipt).standardizedFileURL
                let destinationURL = URL(fileURLWithPath: destinationPath).standardizedFileURL
                let providerReceiptURL = destinationURL
                    .appendingPathComponent(".brainlayer-provider", isDirectory: true)
                    .appendingPathComponent("icloud-provider.receipt.json", isDirectory: false)
                scheduleState.lastSuccess = now()
                scheduleState.lastReceipt = receipt
                scheduleState.nextDue = now().addingTimeInterval(Self.scheduleInterval)
                scheduleState.lastStatus = "local_verified"
                do {
                    try saveScheduleState(scheduleState)
                } catch {
                    publish(.failed(message: "Could not persist local backup verification: \(error)"), to: { _ in })
                    if activeRunID == runID { task = nil; activeRunID = nil }
                    return
                }
                do {
                    let providerReceipt = try await provider.process(BrainBarICloudBackupInput(
                        generationDirectory: localReceiptURL.deletingLastPathComponent(),
                        selectedDestination: destinationURL,
                        providerReceiptURL: providerReceiptURL
                    ))
                    try Task.checkCancellation()
                    guard !providerReceipt.remoteVerified, !providerReceipt.cleanupPermitted else {
                        throw NSError(
                            domain: "BrainLayerBackupProvider",
                            code: 1,
                            userInfo: [NSLocalizedDescriptionKey: "provider attempted to authorize unverified cleanup"]
                        )
                    }
                    if providerReceipt.state == .failed {
                        scheduleState.nextDue = now().addingTimeInterval(Self.failureBackoff)
                        scheduleState.lastStatus = "provider_failed"
                    } else {
                        scheduleState.nextDue = now().addingTimeInterval(Self.scheduleInterval)
                        scheduleState.lastStatus = "provider_\(providerReceipt.state.rawValue)"
                    }
                    try saveScheduleState(scheduleState)
                    if providerReceipt.state == .failed {
                        publish(
                            .failed(message: providerReceipt.failureReason ?? "provider verification failed"),
                            to: { _ in }
                        )
                    } else {
                        publish(.providerObserved(
                            localReceipt: receipt,
                            providerReceipt: providerReceiptURL.path,
                            state: providerReceipt.state
                        ), to: { _ in })
                    }
                } catch is CancellationError {
                    if activeRunID == runID { task = nil; activeRunID = nil }
                    return
                } catch {
                    scheduleState.nextDue = now().addingTimeInterval(Self.failureBackoff)
                    scheduleState.lastStatus = "provider_failed"
                    do {
                        try saveScheduleState(scheduleState)
                    } catch {
                        publish(.failed(message: "Provider and schedule persistence failed: \(error)"), to: { _ in })
                        if activeRunID == runID { task = nil; activeRunID = nil }
                        return
                    }
                    publish(.failed(message: "Provider verification failed: \(error)"), to: { _ in })
                }
            } else {
                var message = String(result.output.trimmingCharacters(in: .whitespacesAndNewlines).prefix(Self.statusMessageLimit))
                if message.isEmpty {
                    message = "backup process exited with status \(result.terminationStatus)"
                }
                scheduleState.nextDue = now().addingTimeInterval(Self.failureBackoff)
                scheduleState.lastStatus =
                    scheduleState.lastSuccess == nil ? "failed" : "failed_with_historical_local_success"
                do {
                    try saveScheduleState(scheduleState)
                } catch {
                    message += " (schedule persistence failed: \(error))"
                }
                publish(.failed(message: message), to: { _ in })
            }
            if activeRunID == runID { task = nil; activeRunID = nil }
        }
    }

    func cancel() {
        task?.cancel()
        publish(.idle, to: { _ in })
    }
}
