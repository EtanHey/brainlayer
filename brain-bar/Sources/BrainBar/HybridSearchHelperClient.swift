import Foundation

struct HybridSearchResponse {
    let text: String
    let metadata: [String: Any]

    init(text: String, metadata: [String: Any] = [:]) {
        self.text = text
        self.metadata = metadata
    }
}

protocol HybridSearchClientProtocol: AnyObject, Sendable {
    func search(arguments: [String: Any]) throws -> HybridSearchResponse
}

final class HybridSearchHelperClient: HybridSearchClientProtocol, @unchecked Sendable {
    private static let maxResponseBytes = 10 * 1024 * 1024
    private let socketPath: String
    private let dbPath: String
    private let pythonExecutable: String
    private let environment: [String: String]
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.hybrid-helper")
    private var process: Process?

    init(
        socketPath: String? = nil,
        dbPath: String,
        pythonExecutable: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.socketPath = socketPath ?? Self.defaultSocketPath()
        self.dbPath = dbPath
        self.pythonExecutable = pythonExecutable ?? Self.resolvePythonExecutable(environment: environment)
        self.environment = environment
    }

    deinit {
        stop()
    }

    static func defaultSocketPath() -> String {
        "/tmp/brainbar-hybrid-\(ProcessInfo.processInfo.processIdentifier).sock"
    }

    static func resolvePythonExecutable(environment: [String: String]) -> String {
        if let explicit = environment["BRAINBAR_PYTHON"], !explicit.isEmpty {
            return explicit
        }
        if let virtualEnv = environment["VIRTUAL_ENV"], !virtualEnv.isEmpty {
            let candidate = "\(virtualEnv)/bin/python"
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        if let repoRoot = normalizedRepoRoot(environment: environment) {
            let candidate = "\(repoRoot)/.venv/bin/python"
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        if let python3 = findExecutable(named: "python3", path: environment["PATH"]) {
            return python3
        }
        if let python = findExecutable(named: "python", path: environment["PATH"]) {
            return python
        }
        return "/usr/bin/env"
    }

    static func resolvePythonPath(environment: [String: String]) -> String? {
        if let existing = environment["PYTHONPATH"], !existing.isEmpty {
            return existing
        }
        guard let repoRoot = normalizedRepoRoot(environment: environment) else {
            return nil
        }
        let sourcePath = "\(repoRoot)/src"
        guard FileManager.default.fileExists(atPath: sourcePath) else {
            return nil
        }
        return sourcePath
    }

    private static func normalizedRepoRoot(environment: [String: String]) -> String? {
        guard let raw = environment["BRAINLAYER_REPO_ROOT"],
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: raw).standardizedFileURL.path
    }

    private static func findExecutable(named name: String, path: String?) -> String? {
        let searchPath = path?.split(separator: ":").map(String.init) ?? []
        for dir in searchPath where !dir.isEmpty {
            let candidate = "\(dir)/\(name)"
            if FileManager.default.isExecutableFile(atPath: candidate) {
                return candidate
            }
        }
        return nil
    }

    func start() {
        queue.sync {
            startLocked()
        }
    }

    func stop() {
        queue.sync {
            process?.terminate()
            process = nil
            unlink(socketPath)
        }
    }

    func search(arguments: [String: Any]) throws -> HybridSearchResponse {
        try queue.sync {
            startLocked()
            return try send(arguments: arguments)
        }
    }

    private func startLocked() {
        if let process, process.isRunning {
            return
        }

        unlink(socketPath)

        let proc = Process()
        if pythonExecutable == "/usr/bin/env" {
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            proc.arguments = [
                "python3",
                "-m",
                "brainlayer.brainbar_hybrid_helper",
                "--socket-path",
                socketPath,
                "--db-path",
                dbPath
            ]
        } else {
            proc.executableURL = URL(fileURLWithPath: pythonExecutable)
            proc.arguments = [
                "-m",
                "brainlayer.brainbar_hybrid_helper",
                "--socket-path",
                socketPath,
                "--db-path",
                dbPath
            ]
        }

        var env = environment
        env["BRAINLAYER_DB"] = dbPath
        if let pythonPath = Self.resolvePythonPath(environment: env) {
            env["PYTHONPATH"] = pythonPath
        }
        env["PYTHONUNBUFFERED"] = "1"
        proc.environment = env
        proc.standardInput = Pipe()
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.standardError

        do {
            try proc.run()
            process = proc
            NSLog("[BrainBar] Hybrid search helper started pid=%d socket=%@", proc.processIdentifier, socketPath)
        } catch {
            NSLog("[BrainBar] Failed to start hybrid search helper: %@", String(describing: error))
            process = nil
        }
    }

    private func send(arguments: [String: Any]) throws -> HybridSearchResponse {
        let fd = try connectWithRetry()
        defer { close(fd) }

        let payload: [String: Any] = [
            "method": "brain_search",
            "arguments": arguments
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        var framed = data
        framed.append(0x0A)
        try writeAll(fd: fd, data: framed)
        let responseData = try readLine(fd: fd)
        let decoded = try JSONSerialization.jsonObject(with: responseData) as? [String: Any]
        guard let decoded else {
            throw HybridSearchHelperError.invalidResponse
        }
        if let ok = decoded["ok"] as? Bool, !ok {
            let message = decoded["error"] as? String ?? "unknown helper error"
            throw HybridSearchHelperError.helperError(message)
        }
        guard let text = decoded["text"] as? String else {
            throw HybridSearchHelperError.invalidResponse
        }
        let metadata = decoded["metadata"] as? [String: Any] ?? [:]
        return HybridSearchResponse(text: text, metadata: metadata)
    }

    private func connectWithRetry() throws -> Int32 {
        var lastErrno: Int32 = 0
        for attempt in 0..<50 {
            let fd = socket(AF_UNIX, SOCK_STREAM, 0)
            guard fd >= 0 else {
                throw HybridSearchHelperError.socket(errno)
            }

            var addr = sockaddr_un()
            addr.sun_family = sa_family_t(AF_UNIX)
            let pathBytes = socketPath.utf8CString
            guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
                close(fd)
                throw HybridSearchHelperError.socketPathTooLong(socketPath)
            }
            withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
                ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                    pathBytes.withUnsafeBufferPointer { src in
                        _ = memcpy(dest, src.baseAddress!, src.count)
                    }
                }
            }

            let rc = withUnsafePointer(to: &addr) { addrPtr in
                addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                    connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
                }
            }
            if rc == 0 {
                return fd
            }
            lastErrno = errno
            close(fd)
            usleep(useconds_t(min(50_000 + attempt * 10_000, 250_000)))
        }
        throw HybridSearchHelperError.connect(lastErrno)
    }

    private func writeAll(fd: Int32, data: Data) throws {
        try data.withUnsafeBytes { rawBuffer in
            guard let base = rawBuffer.baseAddress else { return }
            var written = 0
            while written < data.count {
                let count = write(fd, base.advanced(by: written), data.count - written)
                if count < 0 {
                    if errno == EINTR { continue }
                    throw HybridSearchHelperError.write(errno)
                }
                if count == 0 {
                    throw HybridSearchHelperError.write(EPIPE)
                }
                written += count
            }
        }
    }

    private func readLine(fd: Int32) throws -> Data {
        var result = Data()
        var byte = UInt8(0)
        while true {
            let count = read(fd, &byte, 1)
            if count < 0 {
                if errno == EINTR { continue }
                throw HybridSearchHelperError.read(errno)
            }
            if count == 0 {
                break
            }
            if byte == 0x0A {
                break
            }
            result.append(byte)
            if result.count > Self.maxResponseBytes {
                throw HybridSearchHelperError.responseTooLarge(Self.maxResponseBytes)
            }
        }
        guard !result.isEmpty else {
            throw HybridSearchHelperError.invalidResponse
        }
        return result
    }
}

enum HybridSearchHelperError: LocalizedError {
    case socket(Int32)
    case socketPathTooLong(String)
    case connect(Int32)
    case write(Int32)
    case read(Int32)
    case responseTooLarge(Int)
    case invalidResponse
    case helperError(String)

    var errorDescription: String? {
        switch self {
        case .socket(let code):
            return "hybrid helper socket failed: errno \(code)"
        case .socketPathTooLong(let path):
            return "hybrid helper socket path too long: \(path)"
        case .connect(let code):
            return "hybrid helper connect failed: errno \(code)"
        case .write(let code):
            return "hybrid helper write failed: errno \(code)"
        case .read(let code):
            return "hybrid helper read failed: errno \(code)"
        case .responseTooLarge(let limit):
            return "hybrid helper response exceeded \(limit) bytes"
        case .invalidResponse:
            return "hybrid helper returned an invalid response"
        case .helperError(let message):
            return "hybrid helper error: \(message)"
        }
    }
}
