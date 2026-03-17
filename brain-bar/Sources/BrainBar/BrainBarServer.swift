// BrainBarServer.swift — Integrated socket server + MCP router + database.
//
// Owns:
// - Unix domain socket on /tmp/brainbar.sock
// - MCP Content-Length framing parser
// - JSON-RPC router
// - SQLite database (single-writer)

import Foundation

final class BrainBarServer: @unchecked Sendable {
    private let socketPath: String
    private let dbPath: String
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.server", qos: .userInitiated)
    private var listenFD: Int32 = -1
    private var listenSource: DispatchSourceRead?
    private var clients: [Int32: ClientState] = [:]
    private var router: MCPRouter!
    private var database: BrainDatabase!

    struct ClientState {
        var source: DispatchSourceRead
        var framing: MCPFraming
    }

    init(socketPath: String = "/tmp/brainbar.sock", dbPath: String? = nil) {
        self.socketPath = socketPath
        self.dbPath = dbPath ?? Self.defaultDBPath()
    }

    static func defaultDBPath() -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/.local/share/brainlayer/brainlayer.db"
    }

    func start() {
        queue.async { [weak self] in
            self?.startOnQueue()
        }
    }

    func stop() {
        queue.sync {
            self.cleanup()
        }
    }

    private func startOnQueue() {
        // Initialize database and router
        database = BrainDatabase(path: dbPath)
        router = MCPRouter()
        router.setDatabase(database)

        // Clean up stale socket
        unlink(socketPath)

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            NSLog("[BrainBar] Failed to create socket: errno %d", errno)
            return
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            NSLog("[BrainBar] Socket path too long (%d > %d): %@",
                  pathBytes.count, MemoryLayout.size(ofValue: addr.sun_path), socketPath)
            close(fd)
            return
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                pathBytes.withUnsafeBufferPointer { src in
                    _ = memcpy(dest, src.baseAddress!, src.count)
                }
            }
        }

        let bindResult = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                bind(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bindResult == 0 else {
            NSLog("[BrainBar] Failed to bind: errno %d", errno)
            close(fd)
            return
        }

        chmod(socketPath, 0o600)

        guard listen(fd, 16) == 0 else {
            NSLog("[BrainBar] Failed to listen: errno %d", errno)
            close(fd)
            unlink(socketPath)
            return
        }

        listenFD = fd

        let source = DispatchSource.makeReadSource(fileDescriptor: fd, queue: queue)
        source.setEventHandler { [weak self] in
            self?.acceptClient()
        }
        source.setCancelHandler { [weak self] in
            guard let self else { return }
            close(fd)
            listenFD = -1
        }
        source.resume()
        listenSource = source

        NSLog("[BrainBar] Server listening on %@", socketPath)
    }

    private func acceptClient() {
        let clientFD = accept(listenFD, nil, nil)
        guard clientFD >= 0 else { return }

        let flags = fcntl(clientFD, F_GETFL)
        _ = fcntl(clientFD, F_SETFL, flags | O_NONBLOCK)

        var nosigpipe: Int32 = 1
        setsockopt(clientFD, SOL_SOCKET, SO_NOSIGPIPE, &nosigpipe, socklen_t(MemoryLayout<Int32>.size))

        let readSource = DispatchSource.makeReadSource(fileDescriptor: clientFD, queue: queue)
        readSource.setEventHandler { [weak self] in
            self?.readFromClient(fd: clientFD)
        }
        readSource.setCancelHandler {
            close(clientFD)
        }
        readSource.resume()

        clients[clientFD] = ClientState(source: readSource, framing: MCPFraming())
        NSLog("[BrainBar] Client connected (fd: %d)", clientFD)
    }

    private func readFromClient(fd: Int32) {
        var buf = [UInt8](repeating: 0, count: 65536)
        let n = read(fd, &buf, buf.count)

        if n <= 0 {
            if n == -1, errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR {
                return
            }
            clients[fd]?.source.cancel()
            clients.removeValue(forKey: fd)
            return
        }

        guard var state = clients[fd] else { return }
        state.framing.append(Data(buf[0..<n]))

        let messages = state.framing.extractMessages()
        for msg in messages {
            let response = router.handle(msg)
            if !response.isEmpty {
                sendResponse(fd: fd, response: response)
            }
        }

        clients[fd] = state
    }

    private func sendResponse(fd: Int32, response: [String: Any]) {
        guard let framed = try? MCPFraming.encode(response) else { return }
        framed.withUnsafeBytes { ptr in
            var totalWritten = 0
            while totalWritten < framed.count {
                let n = write(fd, ptr.baseAddress!.advanced(by: totalWritten), framed.count - totalWritten)
                if n < 0 {
                    if errno == EAGAIN || errno == EWOULDBLOCK {
                        // Kernel buffer full — brief retry
                        usleep(1000) // 1ms
                        continue
                    }
                    break // Real error
                }
                if n == 0 { break } // EOF
                totalWritten += n
            }
        }
    }

    private func cleanup() {
        listenSource?.cancel()
        listenSource = nil
        for (_, state) in clients {
            state.source.cancel()
        }
        clients.removeAll()
        if listenFD >= 0 { listenFD = -1 }
        unlink(socketPath)
        database?.close()
        NSLog("[BrainBar] Server stopped")
    }
}
