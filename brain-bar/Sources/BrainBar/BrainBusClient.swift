import Darwin
import Foundation

protocol BrainBusEventSource: Sendable {
    func events() -> AsyncStream<BrainBusEvent>
}

final class BrainBusClient: BrainBusEventSource, @unchecked Sendable {
    private let socketPath: String
    private let reconnectDelay: TimeInterval
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.brain-bus-client")

    init(socketPath: String = BrainBarServer.defaultSocketPath(), reconnectDelay: TimeInterval = 1.0) {
        self.socketPath = socketPath
        self.reconnectDelay = reconnectDelay
    }

    func events() -> AsyncStream<BrainBusEvent> {
        AsyncStream { continuation in
            let run = BrainBusClientRun(
                socketPath: socketPath,
                reconnectDelay: reconnectDelay,
                continuation: continuation
            )
            continuation.onTermination = { @Sendable _ in
                run.cancel()
            }
            queue.async {
                run.start()
            }
        }
    }
}

private final class BrainBusClientRun: @unchecked Sendable {
    private let socketPath: String
    private let reconnectDelay: TimeInterval
    private let continuation: AsyncStream<BrainBusEvent>.Continuation
    private let lock = NSLock()
    private var cancelled = false
    private var currentFD: Int32 = -1

    init(
        socketPath: String,
        reconnectDelay: TimeInterval,
        continuation: AsyncStream<BrainBusEvent>.Continuation
    ) {
        self.socketPath = socketPath
        self.reconnectDelay = reconnectDelay
        self.continuation = continuation
    }

    func start() {
        while !isCancelled {
            autoreleasepool {
                if let fd = try? connect() {
                    setCurrentFD(fd)
                    sendWatchCommand(fd: fd)
                    readEvents(fd: fd)
                    close(fd)
                    setCurrentFD(-1)
                }
            }
            if !isCancelled {
                Thread.sleep(forTimeInterval: reconnectDelay)
            }
        }
        continuation.finish()
    }

    func cancel() {
        let fd: Int32 = lock.withLock {
            cancelled = true
            return currentFD
        }
        if fd >= 0 {
            shutdown(fd, SHUT_RDWR)
        }
    }

    private var isCancelled: Bool {
        lock.withLock { cancelled }
    }

    private func setCurrentFD(_ fd: Int32) {
        lock.withLock {
            currentFD = fd
        }
    }

    private func connect() throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw BrainBusClientError.socket(errno) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            close(fd)
            throw BrainBusClientError.socketPathTooLong(socketPath)
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                pathBytes.withUnsafeBufferPointer { src in
                    _ = memcpy(dest, src.baseAddress!, src.count)
                }
            }
        }

        let result = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                Darwin.connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard result == 0 else {
            let code = errno
            close(fd)
            throw BrainBusClientError.connect(code)
        }
        return fd
    }

    private func sendWatchCommand(fd: Int32) {
        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "watch-brain-bus",
        ]
        guard var data = try? JSONSerialization.data(withJSONObject: request) else { return }
        data.append(0x0A)
        data.withUnsafeBytes { ptr in
            _ = write(fd, ptr.baseAddress!, data.count)
        }
    }

    private func readEvents(fd: Int32) {
        var buffer = Data()
        var readBuffer = [UInt8](repeating: 0, count: 8192)
        while !isCancelled {
            let count = read(fd, &readBuffer, readBuffer.count)
            if count > 0 {
                buffer.append(contentsOf: readBuffer[0..<count])
                drainLines(from: &buffer)
            } else {
                break
            }
        }
    }

    private func drainLines(from buffer: inout Data) {
        while let newlineIndex = buffer.firstIndex(of: 0x0A) {
            let line = Data(buffer[..<newlineIndex])
            buffer.removeSubrange(buffer.startIndex...newlineIndex)
            guard !line.isEmpty,
                  let message = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                  let params = message["params"] as? [String: Any],
                  let eventData = try? JSONSerialization.data(withJSONObject: params),
                  let event = try? JSONDecoder().decode(BrainBusEvent.self, from: eventData) else {
                continue
            }
            continuation.yield(event)
        }
    }
}

private enum BrainBusClientError: Error {
    case socket(Int32)
    case connect(Int32)
    case socketPathTooLong(String)
}
