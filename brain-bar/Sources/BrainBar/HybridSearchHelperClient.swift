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
    private static let defaultSocketIOTimeout: TimeInterval = 60
    private static let queueKey = DispatchSpecificKey<UUID>()
    private let socketPath: String
    private let dbPath: String
    private let pythonExecutable: String
    private let environment: [String: String]
    private let socketIOTimeout: TimeInterval
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.hybrid-helper")
    private let queueID = UUID()
    private var process: Process?

    init(
        socketPath: String? = nil,
        dbPath: String,
        pythonExecutable: String? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        socketIOTimeout: TimeInterval = defaultSocketIOTimeout
    ) {
        self.socketPath = socketPath ?? Self.defaultSocketPath()
        self.dbPath = dbPath
        self.pythonExecutable = pythonExecutable ?? Self.resolvePythonExecutable(environment: environment)
        self.environment = environment
        self.socketIOTimeout = socketIOTimeout
        queue.setSpecific(key: Self.queueKey, value: queueID)
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

    var isRunning: Bool {
        if DispatchQueue.getSpecific(key: Self.queueKey) == queueID {
            return process?.isRunning == true
        }
        return queue.sync {
            process?.isRunning == true
        }
    }

    func start() throws {
        try queue.sync {
            try startLocked()
        }
    }

    func stop() {
        if DispatchQueue.getSpecific(key: Self.queueKey) == queueID {
            stopLocked()
            return
        }
        queue.sync {
            stopLocked()
        }
    }

    func search(arguments: [String: Any]) throws -> HybridSearchResponse {
        let profileQueryID = (arguments["_profile_query_id"] as? String)
            ?? (SearchProfileLogger.isEnabled ? SearchProfileLogger.newQueryID() : nil)
        let profileStartedAt = SearchProfileLogger.now()
        SearchProfileLogger.log(scope: "search.brainbar", step: "helper_rpc_start", queryID: profileQueryID)
        do {
            let response = try queue.sync {
                try startLocked()
                do {
                    return try send(arguments: arguments)
                } catch {
                    if Self.shouldRestartHelper(after: error) {
                        stopLocked()
                    }
                    throw error
                }
            }
            SearchProfileLogger.log(
                scope: "search.brainbar",
                step: "helper_rpc_done",
                queryID: profileQueryID,
                durMS: SearchProfileLogger.durationMS(since: profileStartedAt)
            )
            return response
        } catch {
            SearchProfileLogger.log(
                scope: "search.brainbar",
                step: "helper_rpc_done",
                queryID: profileQueryID,
                durMS: SearchProfileLogger.durationMS(since: profileStartedAt),
                fields: ["error": String(describing: error)]
            )
            throw error
        }
    }

    private func stopLocked() {
        if let process, process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
        process = nil
        unlink(socketPath)
    }

    private func startLocked() throws {
        if let process, process.isRunning {
            return
        }

        try Self.validateSocketPath(socketPath)
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
            throw HybridSearchHelperError.launch(String(describing: error))
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
            do {
                try Self.configureNoSigpipe(fd: fd)
            } catch {
                close(fd)
                throw error
            }

            var addr = sockaddr_un()
            addr.sun_family = sa_family_t(AF_UNIX)
            let pathBytes = try Self.validateSocketPath(socketPath)
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
                do {
                    try Self.configureSocketTimeouts(fd: fd, timeout: socketIOTimeout)
                    return fd
                } catch {
                    close(fd)
                    throw error
                }
            }
            lastErrno = errno
            close(fd)
            usleep(useconds_t(min(50_000 + attempt * 10_000, 250_000)))
        }
        throw HybridSearchHelperError.connect(lastErrno)
    }

    @discardableResult
    static func validateSocketPath(_ path: String) throws -> ContiguousArray<CChar> {
        let pathBytes = path.utf8CString
        let addr = sockaddr_un()
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            throw HybridSearchHelperError.socketPathTooLong(path)
        }
        return ContiguousArray(pathBytes)
    }

    static func configureNoSigpipe(fd: Int32) throws {
        var nosigpipe: Int32 = 1
        if setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &nosigpipe, socklen_t(MemoryLayout<Int32>.size)) != 0 {
            throw HybridSearchHelperError.configureSocket(errno)
        }
    }

    static func configureSocketTimeouts(fd: Int32, timeout: TimeInterval) throws {
        let boundedTimeout = max(timeout, 0.001)
        let seconds = Int(boundedTimeout)
        let microseconds = Int((boundedTimeout - Double(seconds)) * 1_000_000)
        var timeval = timeval(tv_sec: seconds, tv_usec: Int32(microseconds))
        let size = socklen_t(MemoryLayout<timeval>.size)

        if setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeval, size) != 0 {
            throw HybridSearchHelperError.configureSocket(errno)
        }
        if setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeval, size) != 0 {
            throw HybridSearchHelperError.configureSocket(errno)
        }
    }

    private static func shouldRestartHelper(after error: Error) -> Bool {
        guard let helperError = error as? HybridSearchHelperError else {
            return false
        }
        switch helperError {
        case .connect, .configureSocket, .write, .read, .responseTooLarge, .invalidResponse:
            return true
        case .socket, .socketPathTooLong, .launch, .helperError:
            return false
        }
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
        let bufferSize = 8192
        var buffer = [UInt8](repeating: 0, count: bufferSize)
        while true {
            let count = buffer.withUnsafeMutableBytes { rawBuffer in
                read(fd, rawBuffer.baseAddress, bufferSize)
            }
            if count < 0 {
                if errno == EINTR { continue }
                throw HybridSearchHelperError.read(errno)
            }
            if count == 0 {
                break
            }

            let chunk = buffer[..<count]
            let newlineIndex = chunk.firstIndex(of: 0x0A)
            let endIndex = newlineIndex ?? count
            if endIndex > 0 {
                if result.count + endIndex > Self.maxResponseBytes {
                    throw HybridSearchHelperError.responseTooLarge(Self.maxResponseBytes)
                }
                result.append(contentsOf: buffer[..<endIndex])
            }
            if newlineIndex != nil {
                break
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
    case configureSocket(Int32)
    case write(Int32)
    case read(Int32)
    case launch(String)
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
        case .configureSocket(let code):
            return "hybrid helper socket timeout configuration failed: errno \(code)"
        case .write(let code):
            return "hybrid helper write failed: errno \(code)"
        case .read(let code):
            return "hybrid helper read failed: errno \(code)"
        case .launch(let message):
            return "hybrid helper launch failed: \(message)"
        case .responseTooLarge(let limit):
            return "hybrid helper response exceeded \(limit) bytes"
        case .invalidResponse:
            return "hybrid helper returned an invalid response"
        case .helperError(let message):
            return "hybrid helper error: \(message)"
        }
    }
}
