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
        case failed(message: String)

        var title: String {
            switch self {
            case .idle: "Not run"
            case .running: "Running"
            case let .localVerified(receipt): "Local backup verified — \(receipt)"
            case let .failed(message): "Backup failed — \(message)"
            }
        }
    }

    static let shared = BrainLayerBackupCoordinator()
    @Published private(set) var status: Status = .idle
    private static let statusMessageLimit = 4 * 1024
    private static let scheduleInterval: TimeInterval = 24 * 60 * 60
    private static let failureBackoff: TimeInterval = 60 * 60
    private let commandRunner: BrainLayerBackupCommandRunner
    private let stateURL: URL
    private let now: @Sendable () -> Date
    private var task: Task<Void, Never>?
    private var activeRunID: UUID?
    private var scheduleTimer: DispatchSourceTimer?
    private var statusObservers: [UUID: @MainActor (Status) -> Void] = [:]

    init(
        commandRunner: BrainLayerBackupCommandRunner = BrainLayerBackupCommandRunner(),
        stateURL: URL = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".local/share/brainlayer/backup-state.json"),
        now: @escaping @Sendable () -> Date = Date.init
    ) {
        self.commandRunner = commandRunner
        self.stateURL = stateURL
        self.now = now
    }

    deinit {
        task?.cancel()
        scheduleTimer?.cancel()
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

    func startScheduledIfDue(configURL: URL = BrainLayerConfigStore.defaultConfigURL()) {
        if scheduleTimer == nil {
            let timer = DispatchSource.makeTimerSource(queue: .main)
            timer.schedule(deadline: .now() + 60 * 60, repeating: 60 * 60)
            timer.setEventHandler { [weak self] in
                self?.startScheduledIfDue(configURL: configURL)
            }
            timer.resume()
            scheduleTimer = timer
        }
        guard task == nil else { return }
        do {
            let config = try BrainLayerConfigStore(configURL: configURL).loadDocument().config
            guard config.backupEnabled else { publish(.idle, to: { _ in }); return }
            let state = try loadScheduleState()
            guard state.schema == 1 else { throw NSError(domain: "BrainLayerBackupSchedule", code: 1, userInfo: [NSLocalizedDescriptionKey: "Unsupported backup schedule schema \(state.schema)"]) }
            let destination = config.backupDestinationPath
            let identity = config.backupDestinationIdentity
            let changed = state.destinationPath != destination || state.destinationDevice != identity?.stDev || state.destinationInode != identity?.stIno
            guard changed || state.nextDue == nil || state.nextDue! <= now() else { return }
            startIfConfigured(config: config, configURL: configURL)
        } catch {
            publish(.failed(message: "Could not load backup schedule: \(error)"), to: { _ in })
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
              !config.backupTranscriptRoots.isEmpty else {
            publish(.failed(message: "Backup needs a saved folder, filesystem identity, and transcript roots."), to: onStatus)
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
        let roots = config.backupTranscriptRoots
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
            for root in roots { arguments += ["--transcript-root", root] }
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
               !receipt.isEmpty {
                scheduleState.lastSuccess = now()
                scheduleState.nextDue = now().addingTimeInterval(Self.scheduleInterval)
                scheduleState.lastStatus = "local_verified"
                scheduleState.lastReceipt = receipt
                do {
                    try saveScheduleState(scheduleState)
                    publish(.localVerified(receipt: receipt), to: { _ in })
                } catch {
                    publish(.failed(message: "Could not persist backup schedule: (error)"), to: { _ in })
                }
            } else {
                var message = String(result.output.trimmingCharacters(in: .whitespacesAndNewlines).prefix(Self.statusMessageLimit))
                scheduleState.nextDue = now().addingTimeInterval(Self.failureBackoff)
                scheduleState.lastStatus = "failed"
                do {
                    try saveScheduleState(scheduleState)
                } catch {
                    message += " (schedule persistence failed: (error))"
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
