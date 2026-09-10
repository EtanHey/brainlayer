import Foundation

struct BrainLayerBackupCommandResult: Equatable, Sendable {
    let terminationStatus: Int32
    let output: String
}

private final class BrainLayerBackupCompletionState: @unchecked Sendable {
    private let lock = NSLock()
    private var finished = false

    func finish(
        _ status: Int32,
        output: String,
        continuation: CheckedContinuation<BrainLayerBackupCommandResult, Never>
    ) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return }
        finished = true
        continuation.resume(returning: BrainLayerBackupCommandResult(terminationStatus: status, output: output))
    }
}

struct BrainLayerBackupCommandRunner: Sendable {
    let run: @Sendable ([String]) async -> BrainLayerBackupCommandResult

    init(run: @escaping @Sendable ([String]) async -> BrainLayerBackupCommandResult = BrainLayerBackupCommandRunner.execute) {
        self.run = run
    }

    private static func execute(arguments: [String]) async -> BrainLayerBackupCommandResult {
        await withCheckedContinuation { continuation in
            let process = Process()
            let pipe = Pipe()
            let state = BrainLayerBackupCompletionState()
            process.executableURL = URL(fileURLWithPath: "/opt/homebrew/opt/brainlayer/libexec/venv/bin/python")
            process.arguments = arguments
            process.standardOutput = pipe
            process.standardError = pipe
            let finish: @Sendable (Int32, String) -> Void = { status, output in
                state.finish(status, output: output, continuation: continuation)
            }
            process.terminationHandler = { process in
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                finish(process.terminationStatus, String(decoding: data, as: UTF8.self))
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + 30 * 60) {
                guard process.isRunning else { return }
                process.terminate()
                finish(124, "backup process timed out after 1800 seconds")
            }
            do {
                try process.run()
            } catch {
                finish(127, "could not launch pinned BrainLayer Python: \(error)")
            }
        }
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
        task?.cancel()
        status = .running
        let roots = config.backupTranscriptRoots
        task = Task { [commandRunner] in
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
               let receipt = object["receipt"] as? String {
                status = .localVerified(receipt: receipt)
                onStatus(status)
            } else {
                status = .failed(message: result.output.trimmingCharacters(in: .whitespacesAndNewlines))
                onStatus(status)
            }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
        status = .idle
    }
}
