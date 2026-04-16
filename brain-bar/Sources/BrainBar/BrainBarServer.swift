// BrainBarServer.swift — Integrated socket server + MCP router + database.
//
// Owns:
// - Unix domain socket on /tmp/brainbar.sock
// - MCP Content-Length framing parser
// - JSON-RPC router
// - SQLite database (single-writer)

import Darwin
import Foundation

final class BrainBarServer: @unchecked Sendable {
    private struct SubscriptionPayload: Encodable {
        let status: String
        let agentID: String
        let generation: Int
        let tags: [String]
        let lastDeliveredSeq: Int64
        let lastAckedSeq: Int64
        let unreadCount: Int

        enum CodingKeys: String, CodingKey {
            case status
            case agentID = "agent_id"
            case generation
            case tags
            case lastDeliveredSeq = "last_delivered_seq"
            case lastAckedSeq = "last_acked_seq"
            case unreadCount = "unread_count"
        }
    }

    private struct ChannelNotification: Encodable {
        let jsonrpc = "2.0"
        let method = "notifications/claude/channel"
        let params: Params

        struct Params: Encodable {
            let content: String
            let meta: Meta
        }

        struct Meta: Encodable {
            let chunkID: String
            let rowID: String
            let agentID: String
            let tags: String
            let importance: String

            enum CodingKeys: String, CodingKey {
                case chunkID = "chunk_id"
                case rowID = "rowid"
                case agentID = "agent_id"
                case tags
                case importance
            }
        }
    }

    private struct StoreResultPayload: Decodable {
        let chunkID: String
        let rowID: Int64

        enum CodingKeys: String, CodingKey {
            case chunkID = "chunk_id"
            case rowID = "rowid"
        }
    }

    private struct PendingWrite {
        let data: Data
        var totalWritten: Int
        var lastProgressAt: UInt64
        let onDelivered: (() -> Void)?
    }

    private let socketPath: String
    private let dbPath: String
    private let providedDatabase: BrainDatabase?
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.server", qos: .userInitiated)
    private var listenFD: Int32 = -1
    private var listenSource: DispatchSourceRead?
    private var clients: [Int32: ClientState] = [:]
    private var router: MCPRouter!
    private var database: BrainDatabase!
    var onDatabaseReady: (@Sendable (BrainDatabase) -> Void)?
    /// Max time to wait for a backpressured client to become writable again.
    /// Temporary bursts should survive; truly dead peers still get disconnected.
    static let writeStallTimeoutMilliseconds: Int32 = 250
    static let writeChunkSize = 4_096
    static let writeRetrySleepMicroseconds: useconds_t = 2_000
    private let debugLogPath = "/tmp/brainbar-debug.log"

    private func debugLog(_ msg: String) {
        let ts = ISO8601DateFormatter().string(from: Date())
        let line = "[\(ts)] \(msg)\n"
        if let fh = FileHandle(forWritingAtPath: debugLogPath) {
            fh.seekToEndOfFile()
            fh.write(Data(line.utf8))
            fh.closeFile()
        } else {
            FileManager.default.createFile(atPath: debugLogPath, contents: Data(line.utf8))
        }
    }

    private func debugLogData(_ label: String, _ data: Data) {
        let hex = data.prefix(256).map { String(format: "%02x", $0) }.joined(separator: " ")
        let text = String(data: data.prefix(512), encoding: .utf8) ?? "<non-utf8>"
        debugLog("\(label) (\(data.count) bytes)\n  HEX: \(hex)\n  TEXT: \(text)")
    }

    private struct ClientState {
        var source: DispatchSourceRead
        var framing: MCPFraming
        /// Whether this client uses Content-Length framing (LSP-style).
        /// false = newline-delimited JSON-RPC (Claude Code v2.1+).
        var usesContentLengthFraming: Bool = true
        var agentID: String?
        var subscribedTags: Set<String> = []
        var pendingWrites: [PendingWrite] = []
        var hasScheduledWriteRetry = false
    }

    init(socketPath: String? = nil, dbPath: String? = nil, database: BrainDatabase? = nil) {
        self.socketPath = socketPath ?? Self.defaultSocketPath()
        self.dbPath = dbPath ?? Self.defaultDBPath()
        providedDatabase = database
    }

    static func defaultSocketPath() -> String {
        if let override = ProcessInfo.processInfo.environment["BRAINBAR_SOCKET_PATH"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return override
        }
        return "/tmp/brainbar.sock"
    }

    static func defaultDBPath() -> String {
        if let override = ProcessInfo.processInfo.environment["BRAINBAR_DB_PATH"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return override
        }
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
        // 1. Create router FIRST (no DB dependency).
        //    initialize + tools/list work without a database.
        router = MCPRouter()

        // 2. Bind socket BEFORE database init.
        //    After a restart the socket must exist before Claude Code tries
        //    to connect via socat.  Connections queue in the listen backlog
        //    while the DB opens.
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
        debugLog("SERVER STARTED — listening on \(socketPath)")

        // 3. NOW open the database (may take time on cold start with 8 GB file).
        //    Connections accepted above queue in the listen backlog.
        //    initialize / tools/list already work; tools/call returns a
        //    graceful error until the DB is ready.
        let db = providedDatabase ?? BrainDatabase(path: dbPath)
        if db.isOpen {
            database = db
            router.setDatabase(db)
            onDatabaseReady?(db)
            NSLog("[BrainBar] Database ready (%@)", dbPath)
        } else {
            NSLog("[BrainBar] ⚠️ DATABASE FAILED TO OPEN — tools/call will return errors (%@)", dbPath)
        }
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
        debugLog("CLIENT CONNECTED fd=\(clientFD) (total clients: \(clients.count))")
    }

    private func readFromClient(fd: Int32) {
        var buf = [UInt8](repeating: 0, count: 65536)
        let n = read(fd, &buf, buf.count)

        if n <= 0 {
            if n == -1, errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR {
                return
            }
            disconnectClient(fd: fd)
            return
        }

        guard var state = clients[fd] else { return }
        let incoming = Data(buf[0..<n])
        debugLogData("RECV fd=\(fd)", incoming)
        state.framing.append(incoming)

        let messages = state.framing.extractMessages()
        // Detect framing mode from first message extraction
        if !messages.isEmpty {
            state.usesContentLengthFraming = state.framing.lastExtractUsedContentLength
        }
        debugLog("EXTRACTED \(messages.count) messages from fd=\(fd) (framing=\(state.usesContentLengthFraming ? "content-length" : "newline-json"), buffer remaining: \(state.framing.bufferCount) bytes)")
        for msg in messages {
            let method = msg["method"] as? String ?? "<no method>"
            let id = msg["id"]
            debugLog("  MSG fd=\(fd): method=\(method) id=\(String(describing: id))")
            let response = handleMessage(fd: fd, request: msg)
            if !response.isEmpty {
                sendResponse(fd: fd, response: response, useContentLength: state.usesContentLengthFraming)
                debugLog("  SENT response for method=\(method)")
            } else {
                debugLog("  NO RESPONSE for method=\(method) (notification)")
            }
            // sendResponse may have called disconnectClient — stop processing
            if clients[fd] == nil { return }
        }

        if var latest = clients[fd] {
            latest.framing = state.framing
            latest.usesContentLengthFraming = state.usesContentLengthFraming
            clients[fd] = latest
        }
    }

    private func handleMessage(fd: Int32, request: [String: Any]) -> [String: Any] {
        if let toolCall = parseToolCall(request) {
            switch toolCall.name {
            case "brain_subscribe":
                return handleSubscribeTool(fd: fd, id: request["id"], arguments: toolCall.arguments)
            case "brain_unsubscribe":
                return handleUnsubscribeTool(fd: fd, id: request["id"], arguments: toolCall.arguments)
            case "brain_ack":
                return handleAckTool(id: request["id"], arguments: toolCall.arguments)
            default:
                let response = router.handle(request)
                if toolCall.name == "brain_store", !isToolError(response) {
                    publishStoredChunk(response: response, arguments: toolCall.arguments)
                }
                return response
            }
        }
        if let method = request["method"] as? String {
            switch method {
            default:
                break
            }
        }
        return router.handle(request)
    }

    @discardableResult
    private func sendResponse(
        fd: Int32,
        response: [String: Any],
        useContentLength: Bool = true,
        onDelivered: (() -> Void)? = nil
    ) -> Bool {
        let framed: Data
        if useContentLength {
            guard let data = try? MCPFraming.encode(response) else { return false }
            framed = data
        } else {
            // Newline-delimited JSON-RPC (Claude Code v2.1+ / MCP 2025-11-25)
            guard let jsonData = try? JSONSerialization.data(withJSONObject: response) else { return false }
            var data = jsonData
            data.append(0x0A) // trailing \n
            framed = data
        }
        return enqueueWrite(fd: fd, data: framed, onDelivered: onDelivered)
    }

    @discardableResult
    private func enqueueWrite(fd: Int32, data: Data, onDelivered: (() -> Void)? = nil) -> Bool {
        guard var state = clients[fd] else { return false }
        state.pendingWrites.append(
            PendingWrite(
                data: data,
                totalWritten: 0,
                lastProgressAt: DispatchTime.now().uptimeNanoseconds,
                onDelivered: onDelivered
            )
        )
        clients[fd] = state
        return flushPendingWrites(fd: fd)
    }

    @discardableResult
    private func flushPendingWrites(fd: Int32) -> Bool {
        guard var state = clients[fd] else { return false }
        state.hasScheduledWriteRetry = false
        clients[fd] = state

        while var pending = clients[fd]?.pendingWrites.first {
            let remaining = pending.data.count - pending.totalWritten
            let nextChunkSize = min(remaining, Self.writeChunkSize)
            let n = pending.data.withUnsafeBytes { ptr in
                write(fd, ptr.baseAddress!.advanced(by: pending.totalWritten), nextChunkSize)
            }
            if n < 0 {
                if errno == EINTR {
                    continue
                }
                if errno == EAGAIN || errno == EWOULDBLOCK {
                    let now = DispatchTime.now().uptimeNanoseconds
                    let stallDeadline = pending.lastProgressAt
                        + UInt64(Self.writeStallTimeoutMilliseconds) * 1_000_000
                    guard now < stallDeadline else {
                        NSLog(
                            "[BrainBar] ⚠️ Write stalled on fd %d for %d ms — disconnecting dead client",
                            fd,
                            Self.writeStallTimeoutMilliseconds
                        )
                        disconnectClient(fd: fd)
                        return false
                    }
                    scheduleWriteRetryIfNeeded(fd: fd)
                    return true
                }
                NSLog("[BrainBar] Write error on fd %d: errno %d", fd, errno)
                disconnectClient(fd: fd)
                return false
            }
            if n == 0 {
                NSLog("[BrainBar] Write returned 0 on fd %d — peer closed", fd)
                disconnectClient(fd: fd)
                return false
            }

            pending.totalWritten += n
            pending.lastProgressAt = DispatchTime.now().uptimeNanoseconds

            guard var latest = clients[fd] else { return false }
            latest.pendingWrites[0] = pending

            if pending.totalWritten == pending.data.count {
                let onDelivered = pending.onDelivered
                latest.pendingWrites.removeFirst()
                clients[fd] = latest
                onDelivered?()
            } else {
                clients[fd] = latest
            }
        }

        return clients[fd] != nil
    }

    private func scheduleWriteRetryIfNeeded(fd: Int32) {
        guard var state = clients[fd], !state.hasScheduledWriteRetry else { return }
        state.hasScheduledWriteRetry = true
        clients[fd] = state

        let retryDelay = DispatchTimeInterval.microseconds(Int(Self.writeRetrySleepMicroseconds))
        queue.asyncAfter(deadline: .now() + retryDelay) { [weak self] in
            guard let self, self.clients[fd] != nil else { return }
            _ = self.flushPendingWrites(fd: fd)
        }
    }

    private func disconnectClient(fd: Int32) {
        if let agentID = clients[fd]?.agentID {
            try? database?.markSubscriberDisconnected(agentID: agentID)
        }
        clients[fd]?.source.cancel()
        clients.removeValue(forKey: fd)
        NSLog("[BrainBar] Client disconnected (fd: %d)", fd)
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
        if providedDatabase == nil {
            database?.close()
        }
        NSLog("[BrainBar] Server stopped")
    }

    private func handleSubscribeTool(fd: Int32, id: Any?, arguments: [String: Any]) -> [String: Any] {
        guard let agentID = (arguments["agent_id"] as? String) ?? (arguments["subscriber_id"] as? String),
              let tags = arguments["tags"] as? [String],
              let database else {
            return toolErrorResponse(id: id, message: "Database not available")
        }

        do {
            let existing = try database.subscription(agentID: agentID)
            let incrementGeneration = prepareAgentTakeover(fd: fd, agentID: agentID, recordExists: existing != nil)
            let record = try database.upsertSubscription(agentID: agentID, tags: tags, incrementGeneration: incrementGeneration)
            if var client = clients[fd] {
                client.agentID = agentID
                client.subscribedTags = Set(record.tags)
                clients[fd] = client
            }
            let unreadCount = try database.unreadCount(agentID: agentID, tags: record.tags)
            let payload = SubscriptionPayload(
                status: "subscribed",
                agentID: agentID,
                generation: record.generation,
                tags: record.tags,
                lastDeliveredSeq: record.lastDeliveredSeq,
                lastAckedSeq: record.lastAckedSeq,
                unreadCount: unreadCount
            )
            return jsonRPCTextResult(id: id, text: jsonString(payload))
        } catch {
            return toolErrorResponse(id: id, message: error.localizedDescription)
        }
    }

    private func handleUnsubscribeTool(fd: Int32, id: Any?, arguments: [String: Any]) -> [String: Any] {
        guard let agentID = (arguments["agent_id"] as? String) ?? (arguments["subscriber_id"] as? String),
              let database else {
            return toolErrorResponse(id: id, message: "Database not available")
        }

        do {
            let tags = arguments["tags"] as? [String]
            let record = try database.removeSubscription(agentID: agentID, tags: tags)
            if var client = clients[fd] {
                client.agentID = agentID
                client.subscribedTags = Set(record.tags)
                clients[fd] = client
            }
            let payload = SubscriptionPayload(
                status: "unsubscribed",
                agentID: agentID,
                generation: record.generation,
                tags: record.tags,
                lastDeliveredSeq: record.lastDeliveredSeq,
                lastAckedSeq: record.lastAckedSeq,
                unreadCount: try database.unreadCount(agentID: agentID, tags: record.tags)
            )
            return jsonRPCTextResult(id: id, text: jsonString(payload))
        } catch {
            return toolErrorResponse(id: id, message: error.localizedDescription)
        }
    }

    private func handleAckTool(id: Any?, arguments: [String: Any]) -> [String: Any] {
        guard let agentID = (arguments["agent_id"] as? String) ?? (arguments["subscriber_id"] as? String),
              let database else {
            return toolErrorResponse(id: id, message: "Database not available")
        }

        let seq: Int64
        if let intSeq = arguments["seq"] as? Int {
            seq = Int64(intSeq)
        } else if let int64Seq = arguments["seq"] as? Int64 {
            seq = int64Seq
        } else {
            return toolErrorResponse(id: id, message: "Missing or invalid seq")
        }

        do {
            try database.acknowledge(agentID: agentID, seq: seq)
            return jsonRPCTextResult(id: id, text: #"{"status":"acked"}"#)
        } catch {
            return toolErrorResponse(id: id, message: error.localizedDescription)
        }
    }

    private func prepareAgentTakeover(fd: Int32, agentID: String, recordExists: Bool) -> Bool {
        var incrementGeneration = recordExists
        for (otherFD, otherClient) in Array(clients) where otherFD != fd && otherClient.agentID == agentID {
            incrementGeneration = true
            disconnectClient(fd: otherFD)
        }
        if clients[fd]?.agentID == agentID {
            return false
        }
        return incrementGeneration
    }

    private func toolErrorResponse(id: Any?, message: String) -> [String: Any] {
        var response: [String: Any] = [
            "jsonrpc": "2.0",
            "result": [
                "content": [
                    ["type": "text", "text": "Error: \(message)"]
                ],
                "isError": true
            ] as [String: Any]
        ]
        if let id {
            response["id"] = id
        }
        return response
    }

    private func jsonRPCTextResult(id: Any?, text: String) -> [String: Any] {
        var response: [String: Any] = [
            "jsonrpc": "2.0",
            "result": [
                "content": [
                    ["type": "text", "text": text]
                ]
            ] as [String: Any]
        ]
        if let id {
            response["id"] = id
        }
        return response
    }

    private func jsonRPCResult(id: Any?, result: [String: Any]) -> [String: Any] {
        var response: [String: Any] = [
            "jsonrpc": "2.0",
            "result": result
        ]
        if let id {
            response["id"] = id
        }
        return response
    }

    private func jsonRPCError(id: Any?, code: Int, message: String) -> [String: Any] {
        var response: [String: Any] = [
            "jsonrpc": "2.0",
            "error": [
                "code": code,
                "message": message
            ]
        ]
        if let id {
            response["id"] = id
        }
        return response
    }

    private func isToolError(_ response: [String: Any]) -> Bool {
        let result = response["result"] as? [String: Any]
        return result?["isError"] as? Bool == true
    }

    private func parseToolCall(_ request: [String: Any]) -> (name: String, arguments: [String: Any])? {
        guard let method = request["method"] as? String, method == "tools/call",
              let params = request["params"] as? [String: Any],
              let name = params["name"] as? String else {
            return nil
        }
        let arguments = params["arguments"] as? [String: Any] ?? [:]
        return (name, arguments)
    }

    private func publishStoredChunk(response: [String: Any], arguments: [String: Any]) {
        guard let stored = extractStoredChunk(from: response),
              let content = arguments["content"] as? String,
              let tags = arguments["tags"] as? [String],
              !tags.isEmpty else {
            return
        }

        let importance = arguments["importance"] as? Int ?? 5
        let tagSet = Set(tags)
        for (clientFD, client) in Array(clients) {
            if let agentID = client.agentID,
               !client.subscribedTags.isDisjoint(with: tagSet) {
                let notification = ChannelNotification(
                    params: .init(
                        content: content,
                        meta: .init(
                            chunkID: stored.chunkID,
                            rowID: String(stored.rowID),
                            agentID: agentID,
                            tags: tags.joined(separator: ","),
                            importance: String(importance)
                        )
                    )
                )
                guard let notificationObject = jsonObject(notification) else {
                    continue
                }

                let delivered = sendResponse(
                    fd: clientFD,
                    response: notificationObject,
                    useContentLength: client.usesContentLengthFraming,
                    onDelivered: { [weak database] in
                        try? database?.markDelivered(agentID: agentID, seq: stored.rowID)
                    }
                )
                if !delivered {
                    continue
                }
            }
        }
    }

    private func extractStoredChunk(from response: [String: Any]) -> StoreResultPayload? {
        guard let result = response["result"] as? [String: Any] else {
            return nil
        }
        if let stored = result["_brainbarStoredChunk"] as? [String: Any],
           let chunkID = stored["chunk_id"] as? String {
            let rowID: Int64
            if let intRowID = stored["rowid"] as? Int64 {
                rowID = intRowID
            } else if let intRowID = stored["rowid"] as? Int {
                rowID = Int64(intRowID)
            } else {
                return nil
            }
            return StoreResultPayload(chunkID: chunkID, rowID: rowID)
        }
        guard let content = result["content"] as? [[String: Any]],
              let text = content.first?["text"] as? String,
              let data = text.data(using: .utf8),
              let payload = try? JSONDecoder().decode(StoreResultPayload.self, from: data) else {
            return nil
        }
        return payload
    }

    private func jsonString<T: Encodable>(_ payload: T) -> String {
        guard let data = try? JSONEncoder().encode(payload),
              let text = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return text
    }

    private func jsonObject<T: Encodable>(_ payload: T) -> [String: Any]? {
        guard let data = try? JSONEncoder().encode(payload),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return object
    }
}
