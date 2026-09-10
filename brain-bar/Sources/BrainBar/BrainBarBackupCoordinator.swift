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

    @Published private(set) var status: Status = .idle
    private static let statusMessageLimit = 4 * 1024
    private let commandRunner: BrainLayerBackupCommandRunner
    private var task: Task<Void, Never>?

    init(commandRunner: BrainLayerBackupCommandRunner = BrainLayerBackupCommandRunner()) {
        self.commandRunner = commandRunner
    }

    deinit { task?.cancel() }

    func startIfConfigured(
        config: BrainLayerConfig,
        configURL: URL = BrainLayerConfigStore.defaultConfigURL(),
        onStatus: @escaping @MainActor (Status) -> Void = { _ in }
    ) {
        guard config.backupEnabled,
              config.backupDestinationPath != nil,
              config.backupDestinationIdentity != nil,
              !config.backupTranscriptRoots.isEmpty else {
            status = .failed(message: "Backup needs a saved folder, filesystem identity, and transcript roots.")
            onStatus(status)
            return
        }
        let previous = task
        previous?.cancel()
        status = .running
        onStatus(status)
        let roots = config.backupTranscriptRoots
        task = Task { [commandRunner] in
            if let previous { await previous.value }
            guard !Task.isCancelled else { return }
            var arguments = [
                "-I", "-B", "-m", "brainlayer.backup_run_once",
                "--config", configURL.path,
                "--generation-id", UUID().uuidString,
            ]
            for root in roots { arguments += ["--transcript-root", root] }
            let result = await commandRunner.run(arguments)
            guard !Task.isCancelled else { return }
            if result.terminationStatus == 0,
               let payload = result.output.data(using: .utf8),
               let object = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
               object["status"] as? String == "local_verified",
               let receipt = object["receipt"] as? String,
               !receipt.isEmpty {
                status = .localVerified(receipt: receipt)
            } else {
                let message = String(result.output.trimmingCharacters(in: .whitespacesAndNewlines).prefix(Self.statusMessageLimit))
                status = .failed(message: message)
            }
            onStatus(status)
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        status = .idle
    }
}
