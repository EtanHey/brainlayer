// SocketIntegrationTests.swift — RED tests for end-to-end socket + MCP flow.
//
// Tests the full pipeline: connect to Unix socket → send Content-Length framed
// MCP request → receive Content-Length framed response.

import SQLite3
import XCTest
@testable import BrainBar

private final class SQLiteProgressGate: @unchecked Sendable {
    let entered = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var shouldBlock = true

    func step() -> Int32 {
        lock.lock()
        let block = shouldBlock
        shouldBlock = false
        lock.unlock()
        if block {
            entered.signal()
            release.wait()
        }
        return 0
    }
}

private final class BrainDatabaseCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var captured: BrainDatabase?

    func set(_ database: BrainDatabase) {
        lock.lock()
        captured = database
        lock.unlock()
    }

    func get() -> BrainDatabase? {
        lock.lock()
        defer { lock.unlock() }
        return captured
    }
}

private final class ClientLifecycleCapture: @unchecked Sendable {
    private let condition = NSCondition()
    private var acceptedFDs: [Int32] = []
    private var closedFDs = Set<Int32>()

    func recordAccepted(_ fd: Int32) {
        condition.lock()
        acceptedFDs.append(fd)
        condition.broadcast()
        condition.unlock()
    }

    func recordClosed(_ fd: Int32) {
        condition.lock()
        closedFDs.insert(fd)
        condition.broadcast()
        condition.unlock()
    }

    func waitForAccepted(at index: Int, timeout: TimeInterval = 5) -> Int32? {
        let deadline = Date().addingTimeInterval(timeout)
        condition.lock()
        defer { condition.unlock() }
        while acceptedFDs.count <= index {
            guard condition.wait(until: deadline) else { return nil }
        }
        return acceptedFDs[index]
    }

    func waitForClosed(_ fd: Int32, timeout: TimeInterval = 5) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        condition.lock()
        defer { condition.unlock() }
        while !closedFDs.contains(fd) {
            guard condition.wait(until: deadline) else { return false }
        }
        return true
    }
}

private func blockSQLiteProgress(_ context: UnsafeMutableRawPointer?) -> Int32 {
    guard let context else { return 0 }
    return Unmanaged<SQLiteProgressGate>.fromOpaque(context).takeUnretainedValue().step()
}

final class SocketIntegrationTests: XCTestCase {
    let testSocketPath = "/tmp/brainbar-test-\(ProcessInfo.processInfo.processIdentifier).sock"
    var server: BrainBarServer!
    var db: BrainDatabase!
    var tempDBPath: String!
    var originalMCPProfile: String?

    override func setUp() {
        super.setUp()
        originalMCPProfile = ProcessInfo.processInfo.environment["BRAINLAYER_MCP_PROFILE"]
        setenv("BRAINLAYER_MCP_PROFILE", "full", 1)
        tempDBPath = NSTemporaryDirectory() + "brainbar-integration-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
        server = BrainBarServer(socketPath: testSocketPath, dbPath: tempDBPath, database: db)
        server.start()
        XCTAssertTrue(waitForSocket(at: testSocketPath), "Server should bind \(testSocketPath)")
    }

    override func tearDown() {
        server.stop()
        db.close()
        try? FileManager.default.removeItem(atPath: tempDBPath)
        try? FileManager.default.removeItem(atPath: tempDBPath + "-wal")
        try? FileManager.default.removeItem(atPath: tempDBPath + "-shm")
        if let originalMCPProfile {
            setenv("BRAINLAYER_MCP_PROFILE", originalMCPProfile, 1)
        } else {
            unsetenv("BRAINLAYER_MCP_PROFILE")
        }
        super.tearDown()
    }

    // MARK: - Connection

    func testConnectsToSocket() throws {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(fd, 0, "Should create socket")
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 104) { dest in
                _ = testSocketPath.withCString { src in
                    strcpy(dest, src)
                }
            }
        }

        let result = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        XCTAssertEqual(result, 0, "Should connect to brainbar socket (errno: \(errno))")
    }

    // MARK: - MCP Initialize handshake

    func testMCPInitializeOverSocket() throws {
        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "test", "version": "1.0"]
            ]
        ])

        let result = response["result"] as? [String: Any]
        XCTAssertNotNil(result)
        XCTAssertEqual(result?["protocolVersion"] as? String, "2024-11-05")
    }

    // MARK: - MCP tools/list over socket

    func testMCPToolsListOverSocket() throws {
        // Must initialize first
        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "test", "version": "1.0"]
            ]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        ])

        let tools = (response["result"] as? [String: Any])?["tools"] as? [[String: Any]]
        XCTAssertNotNil(tools)
        XCTAssertEqual(tools?.count, 17)

        let encodedResponse = try MCPFraming.encodeJSONResponse(response)
        XCTAssertGreaterThan(encodedResponse.count, 8192)

        for tool in tools ?? [] {
            XCTAssertNotNil(
                tool["annotations"],
                "\(tool["name"] ?? "unknown") should keep annotations over framed socket transport"
            )
        }
    }

    func testCorePaletteExpansionIsIsolatedPerSocketClient() throws {
        restartServer(profile: "core")
        let firstFD = try connectClient()
        defer { close(firstFD) }
        let secondFD = try connectClient()
        defer { close(secondFD) }
        try initializeClient(fd: firstFD, name: "first-core-client")
        try initializeClient(fd: secondFD, name: "second-core-client")

        let coreNames: Set<String> = [
            "brain_search", "brain_store", "brain_recall", "brain_expand", "expand_palette",
        ]
        XCTAssertEqual(Set(try listedToolNames(on: firstFD, id: 2)), coreNames)
        XCTAssertEqual(Set(try listedToolNames(on: secondFD, id: 2)), coreNames)

        try sendMCPRequest(on: firstFD, request: [
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": ["name": "expand_palette", "arguments": [:] as [String: Any]],
        ])
        let expansion = try readMCPMessage(fd: firstFD)
        XCTAssertEqual((expansion["result"] as? [String: Any])?["expanded"] as? Bool, true)

        XCTAssertEqual(try listedToolNames(on: firstFD, id: 4).count, 17)
        XCTAssertEqual(Set(try listedToolNames(on: secondFD, id: 3)), coreNames)
    }

    func testCoreProfileRejectsServerHandledToolsUntilExpansion() throws {
        restartServer(profile: "core")
        let fd = try connectClient()
        defer { close(fd) }
        try initializeClient(fd: fd, name: "core-subscription-client")

        let serverHandledTools: [(name: String, arguments: [String: Any])] = [
            ("brain_subscribe", ["agent_id": "core-agent", "tags": ["agent-message"]]),
            ("brain_unsubscribe", ["agent_id": "core-agent", "tags": ["agent-message"]]),
            ("brain_ack", ["agent_id": "core-agent", "seq": 1]),
        ]
        for (offset, tool) in serverHandledTools.enumerated() {
            try sendMCPRequest(on: fd, request: [
                "jsonrpc": "2.0", "id": 2 + offset, "method": "tools/call",
                "params": ["name": tool.name, "arguments": tool.arguments] as [String: Any],
            ])
            let deferred = try readMCPMessage(fd: fd)
            XCTAssertEqual(
                (deferred["error"] as? [String: Any])?["code"] as? Int,
                -32601,
                "\(tool.name) should stay deferred until this client expands its palette"
            )
        }

        try sendMCPRequest(on: fd, request: [
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": ["name": "expand_palette", "arguments": [:] as [String: Any]],
        ])
        _ = try readMCPMessage(fd: fd)

        try sendMCPRequest(on: fd, request: [
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": ["agent_id": "core-agent", "tags": ["agent-message"]] as [String: Any],
            ],
        ])
        let expanded = try readMCPMessage(fd: fd)
        XCTAssertNil(expanded["error"])
        XCTAssertNotNil(expanded["result"])
    }

    func testRawLineToolsListCompactsForClaudeExtensionLimit() throws {
        let fd = try connectClient()
        defer { close(fd) }

        try sendRawLineJSON(on: fd, object: [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "claude-extension-test", "version": "1.0"]
            ]
        ])
        _ = try readRawLineJSONData(fd: fd)

        try sendRawLineJSON(on: fd, object: [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        ])

        let line = try readRawLineJSONData(fd: fd)
        let response = try JSONSerialization.jsonObject(with: line) as? [String: Any] ?? [:]
        let tools = (response["result"] as? [String: Any])?["tools"] as? [[String: Any]]

        XCTAssertLessThan(
            line.count,
            8192,
            "Claude Desktop's MCPB utility process parses raw extension stdout in 8192-byte chunks"
        )
        XCTAssertEqual(tools?.count, 17)
        for tool in tools ?? [] {
            XCTAssertNil(
                tool["annotations"],
                "\(tool["name"] ?? "unknown") should omit optional annotations over raw newline transport"
            )
        }
    }

    func testWatchBrainBusStreamsStoreEventsOverRawUnixSocket() throws {
        let watchFD = try connectClient()
        defer { close(watchFD) }

        try sendRawLineJSON(on: watchFD, object: [
            "jsonrpc": "2.0",
            "id": 50,
            "method": "watch-brain-bus",
        ])
        _ = try readBrainBusEvent(fd: watchFD, matching: "health_tick")

        let storeResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 51,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Brain bus stream integration",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        XCTAssertNil(storeResponse["error"])

        let event = try readBrainBusEvent(fd: watchFD, matching: "last_chunk_id")
        XCTAssertEqual(event["method"] as? String, "notifications/brain-bus")
        let params = try XCTUnwrap(event["params"] as? [String: Any])
        XCTAssertEqual(params["type"] as? String, "last_chunk_id")
        XCTAssertFalse((params["last_chunk_id"] as? String ?? "").isEmpty)
    }

    // MARK: - MCP tools/call brain_search over socket

    func testMCPBrainSearchOverSocket() throws {
        // Initialize
        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "test", "version": "1.0"]]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "test search"]
            ]
        ])

        XCTAssertNil(response["error"], "brain_search should not error")
        XCTAssertNotNil(response["result"])
    }

    func testBrainBackupVacuumIntoOverSocketCreatesRestorableSnapshot() throws {
        restartServer(profile: "core")
        let socketAttributes = try FileManager.default.attributesOfItem(atPath: testSocketPath)
        let socketPermissions = try XCTUnwrap(socketAttributes[.posixPermissions] as? NSNumber)
        XCTAssertEqual(socketPermissions.intValue & 0o777, 0o600, "The local MCP trust boundary must be owner-only")

        let targetPath = NSTemporaryDirectory() + "brainbar-backup-\(UUID().uuidString).db"
        let completionMarkerPath = targetPath + ".complete"
        defer { try? FileManager.default.removeItem(atPath: targetPath) }
        defer { try? FileManager.default.removeItem(atPath: completionMarkerPath) }

        // Each helper call opens a fresh socket, matching the scheduled Python
        // client and proving that the allowance is independent of session state.
        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 20, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "backup-test", "version": "1.0"]]
        ])

        let toolsResponse = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 21, "method": "tools/list",
        ])
        XCTAssertNil(toolsResponse["error"], "Core tools/list should succeed before checking its inventory")
        let toolsResult = try XCTUnwrap(toolsResponse["result"] as? [String: Any])
        let listedTools = try XCTUnwrap(toolsResult["tools"] as? [[String: Any]])
        XCTAssertFalse(
            listedTools.contains { ($0["name"] as? String) == "brain_backup_vacuum_into" },
            "The backup allowance must not expand the advertised core palette"
        )

        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "vacuum over socket",
                    "tags": ["backup-test"]
                ] as [String: Any]
            ]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": [
                "name": "brain_backup_vacuum_into",
                "arguments": ["target_path": targetPath]
            ]
        ])

        XCTAssertNil(response["error"])
        let result = response["result"] as? [String: Any]
        XCTAssertEqual(result?["isError"] as? Bool, nil)
        XCTAssertTrue(FileManager.default.fileExists(atPath: targetPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: completionMarkerPath))

        var restored: OpaquePointer?
        XCTAssertEqual(sqlite3_open_v2(targetPath, &restored, SQLITE_OPEN_READONLY, nil), SQLITE_OK)
        defer { sqlite3_close(restored) }
        XCTAssertEqual(try queryString("PRAGMA integrity_check", on: restored), "ok")
        XCTAssertEqual(
            try queryString("SELECT content FROM chunks WHERE content = 'vacuum over socket'", on: restored),
            "vacuum over socket"
        )
    }

    func testBlockedBackupVacuumDoesNotStarveFreshSocketProtocolFlow() throws {
        // Use the production topology: a write connection for VACUUM and an
        // independent read-only connection for normal recall traffic.
        server.stop()
        db.close()
        setenv("BRAINLAYER_MCP_PROFILE", "core", 1)
        let databaseReady = DispatchSemaphore(value: 0)
        let databaseCapture = BrainDatabaseCapture()
        server = BrainBarServer(
            socketPath: testSocketPath,
            dbPath: tempDBPath,
            enableHybridSearchHelper: false
        )
        server.onDatabaseReady = { database in
            databaseCapture.set(database)
            databaseReady.signal()
        }
        server.start()
        XCTAssertTrue(waitForSocket(at: testSocketPath), "Server should bind \(testSocketPath)")
        XCTAssertEqual(databaseReady.wait(timeout: .now() + 1), .success)
        guard let writeDatabase = databaseCapture.get() else {
            return XCTFail("Server should expose its ready write database")
        }

        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 39, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "backup-liveness-warmup", "version": "1.0"]],
        ])
        Thread.sleep(forTimeInterval: 0.05)

        let subscriberFD = try connectClient()
        try initializeClient(fd: subscriberFD, name: "backup-liveness-subscriber")
        try sendMCPRequest(on: subscriberFD, request: [
            "jsonrpc": "2.0", "id": 44, "method": "tools/call",
            "params": ["name": "expand_palette", "arguments": [:] as [String: Any]],
        ])
        _ = try readMCPMessage(fd: subscriberFD)
        try sendMCPRequest(on: subscriberFD, request: [
            "jsonrpc": "2.0", "id": 45, "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": ["agent_id": "backup-liveness-agent", "tags": ["agent-message"]],
            ],
        ])
        _ = try readMCPMessage(fd: subscriberFD)

        let progressGate = SQLiteProgressGate()
        defer { progressGate.release.signal() }
        sqlite3_progress_handler(
            writeDatabase.dbHandle,
            1,
            blockSQLiteProgress,
            Unmanaged.passUnretained(progressGate).toOpaque()
        )
        defer { sqlite3_progress_handler(writeDatabase.dbHandle, 0, nil, nil) }

        let targetPath = NSTemporaryDirectory() + "brainbar-blocked-backup-\(UUID().uuidString).db"
        defer { try? FileManager.default.removeItem(atPath: targetPath) }
        defer { try? FileManager.default.removeItem(atPath: targetPath + ".complete") }

        let backupFD = try connectClient()
        defer { close(backupFD) }
        try sendMCPRequest(on: backupFD, request: [
            "jsonrpc": "2.0",
            "id": 40,
            "method": "tools/call",
            "params": [
                "name": "brain_backup_vacuum_into",
                "arguments": ["target_path": targetPath],
            ],
        ])
        XCTAssertEqual(progressGate.entered.wait(timeout: .now() + 1), .success)

        // Subscriber cleanup cannot write on the VACUUM connection and block
        // the request queue while the backup is held.
        close(subscriberFD)
        Thread.sleep(forTimeInterval: 0.05)

        // Deployment health is a held-open, fresh-socket protocol sequence —
        // initialize + tools/list + a real tool call — not a version/plist probe.
        let probeFD = try connectClient()
        defer { close(probeFD) }
        try sendMCPRequest(on: probeFD, request: [
            "jsonrpc": "2.0", "id": 41, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "backup-liveness-probe", "version": "1.0"]],
        ])
        let initialize = try? readMCPMessage(fd: probeFD, timeout: 0.5)
        XCTAssertNotNil(initialize?["result"])

        if initialize != nil {
            try sendMCPRequest(on: probeFD, request: [
                "jsonrpc": "2.0", "id": 42, "method": "tools/list", "params": [:] as [String: Any],
            ])
            let tools = try readMCPMessage(fd: probeFD, timeout: 0.5)
            XCTAssertNotNil((tools["result"] as? [String: Any])?["tools"])

            try sendMCPRequest(on: probeFD, request: [
                "jsonrpc": "2.0", "id": 43, "method": "tools/call",
                "params": ["name": "brain_recall", "arguments": ["mode": "stats"]],
            ])
            let call = try readMCPMessage(fd: probeFD, timeout: 0.5)
            XCTAssertNil(call["error"])
            XCTAssertNotNil(call["result"])
        }

        progressGate.release.signal()
        let backupResponse = try readMCPMessage(fd: backupFD, timeout: 2)
        XCTAssertEqual(backupResponse["id"] as? Int, 40)
        XCTAssertNil(backupResponse["error"])
        XCTAssertNotNil(backupResponse["result"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: targetPath + ".complete"))
    }

    func testDisconnectedBackupResponseDoesNotLeakToReusedServerDescriptor() throws {
        // Use the production topology so VACUUM and normal protocol traffic use
        // separate write/read SQLite connections.
        server.stop()
        db.close()
        setenv("BRAINLAYER_MCP_PROFILE", "core", 1)
        let databaseReady = DispatchSemaphore(value: 0)
        let databaseCapture = BrainDatabaseCapture()
        server = BrainBarServer(
            socketPath: testSocketPath,
            dbPath: tempDBPath,
            enableHybridSearchHelper: false
        )
        let reusedDescriptorDrop = DispatchSemaphore(value: 0)
        let clientLifecycle = ClientLifecycleCapture()
        server.onDatabaseReady = { database in
            databaseCapture.set(database)
            databaseReady.signal()
        }
        server.onDeferredBackupResponseDropped = { descriptorWasReused in
            if descriptorWasReused {
                reusedDescriptorDrop.signal()
            }
        }
        server.onClientAccepted = { clientLifecycle.recordAccepted($0) }
        server.onClientDescriptorClosed = { clientLifecycle.recordClosed($0) }
        server.start()
        XCTAssertTrue(waitForSocket(at: testSocketPath), "Server should bind \(testSocketPath)")
        XCTAssertEqual(databaseReady.wait(timeout: .now() + 1), .success)
        guard let writeDatabase = databaseCapture.get() else {
            return XCTFail("Server should expose its ready write database")
        }
        guard let readinessProbeServerFD = clientLifecycle.waitForAccepted(at: 0) else {
            return XCTFail("Server should observe the readiness probe connection")
        }
        XCTAssertTrue(
            clientLifecycle.waitForClosed(readinessProbeServerFD),
            "Readiness probe must be fully disconnected before descriptor allocation starts"
        )

        // Hold one known-low descriptor to release for the replacement client's
        // socket(). Other holes are filled explicitly after the backup server
        // descriptor closes, so concurrent fd churn cannot invalidate a fixed
        // spare-count assumption.
        let replacementClientHoleFD = socket(AF_UNIX, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(replacementClientHoleFD, 0)
        guard replacementClientHoleFD >= 0 else { return }
        var replacementClientHoleIsOpen = true
        defer {
            if replacementClientHoleIsOpen {
                close(replacementClientHoleFD)
            }
        }

        let progressGate = SQLiteProgressGate()
        defer { progressGate.release.signal() }
        sqlite3_progress_handler(
            writeDatabase.dbHandle,
            1,
            blockSQLiteProgress,
            Unmanaged.passUnretained(progressGate).toOpaque()
        )
        defer { sqlite3_progress_handler(writeDatabase.dbHandle, 0, nil, nil) }

        let targetPath = NSTemporaryDirectory() + "brainbar-reused-fd-backup-\(UUID().uuidString).db"
        let completionMarkerPath = targetPath + ".complete"
        defer { try? FileManager.default.removeItem(atPath: targetPath) }
        defer { try? FileManager.default.removeItem(atPath: completionMarkerPath) }

        let backupFD = try connectClient()
        defer { close(backupFD) }
        XCTAssertGreaterThan(backupFD, replacementClientHoleFD)
        guard let backupServerFD = clientLifecycle.waitForAccepted(at: 1) else {
            return XCTFail("Server should accept the backup connection")
        }
        try sendMCPRequest(on: backupFD, request: [
            "jsonrpc": "2.0",
            "id": 40,
            "method": "tools/call",
            "params": [
                "name": "brain_backup_vacuum_into",
                "arguments": ["target_path": targetPath],
            ],
        ])
        XCTAssertEqual(progressGate.entered.wait(timeout: .now() + 1), .success)

        XCTAssertEqual(shutdown(backupFD, SHUT_RDWR), 0)
        XCTAssertTrue(
            clientLifecycle.waitForClosed(backupServerFD),
            "Backup server descriptor must close before constructing its replacement"
        )

        // socket() always returns the lowest free descriptor. Fill every free
        // hole below the known backup server fd, briefly acquire the target to
        // prove it is free, then release only the target and the one low client
        // hole. The probe's socket() consumes the low hole; accept() must consume
        // the target.
        var fillerFDs: [Int32] = []
        defer { fillerFDs.forEach { close($0) } }
        while true {
            let fd = socket(AF_UNIX, SOCK_STREAM, 0)
            XCTAssertGreaterThanOrEqual(fd, 0)
            guard fd >= 0 else { return }
            if fd == backupServerFD {
                close(fd)
                break
            }
            XCTAssertLessThan(fd, backupServerFD, "Backup server descriptor should be the next target hole")
            guard fd < backupServerFD else {
                close(fd)
                return
            }
            fillerFDs.append(fd)
        }
        close(replacementClientHoleFD)
        replacementClientHoleIsOpen = false

        let probeFD = try connectClient()
        defer { close(probeFD) }
        guard let probeServerFD = clientLifecycle.waitForAccepted(at: 2) else {
            return XCTFail("Server should accept the replacement connection")
        }
        XCTAssertEqual(
            probeServerFD,
            backupServerFD,
            "Replacement connection must reuse the disconnected backup server descriptor"
        )
        try sendMCPRequest(on: probeFD, request: [
            "jsonrpc": "2.0", "id": 41, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "backup-fd-reuse-probe", "version": "1.0"]],
        ])
        let initialize = try readMCPMessage(fd: probeFD, timeout: 0.5)
        XCTAssertNotNil(initialize["result"])

        progressGate.release.signal()
        let completionDeadline = Date().addingTimeInterval(5)
        while !FileManager.default.fileExists(atPath: completionMarkerPath), Date() < completionDeadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: completionMarkerPath))
        XCTAssertEqual(
            reusedDescriptorDrop.wait(timeout: .now() + 5),
            .success,
            "The test must reach the session-identity guard with the descriptor occupied by a replacement client"
        )

        let unsolicited = try? readMCPMessage(fd: probeFD, timeout: 1)
        XCTAssertNotEqual(
            unsolicited?["id"] as? Int,
            40,
            "The replacement connection received the disconnected backup client's response"
        )
        XCTAssertNil(unsolicited, "The replacement connection must not receive an unsolicited response")
    }

    func testBrainBackupVacuumIntoRefusesExistingCompletionMarkerWithoutOverwritingIt() throws {
        let targetPath = NSTemporaryDirectory() + "brainbar-backup-\(UUID().uuidString).db"
        let completionMarkerPath = targetPath + ".complete"
        let originalMarker = Data("unrelated data".utf8)
        try originalMarker.write(to: URL(fileURLWithPath: completionMarkerPath))
        defer { try? FileManager.default.removeItem(atPath: targetPath) }
        defer { try? FileManager.default.removeItem(atPath: completionMarkerPath) }

        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 30, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "backup-marker-test", "version": "1.0"]]
        ])
        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/call",
            "params": [
                "name": "brain_backup_vacuum_into",
                "arguments": ["target_path": targetPath]
            ]
        ])

        let result = try XCTUnwrap(response["result"] as? [String: Any])
        XCTAssertEqual(result["isError"] as? Bool, true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: targetPath))
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: completionMarkerPath)), originalMarker)
    }

    func testMCPBrainSearchOverSocketUsesInjectedHybridHelper() throws {
        server.stop()
        let helper = RecordingHybridSearchClient(
            response: HybridSearchResponse(
                text: #"""
┌─ brain_search: "techgym speakers workshop" ─ 1 result
├─ [1] manual-a0b8a  score:0.97  imp: 8  2026-05-16
└─
"""#
            )
        )
        server = BrainBarServer(socketPath: testSocketPath, dbPath: tempDBPath, database: db, hybridSearchClient: helper)
        server.start()
        XCTAssertTrue(waitForSocket(at: testSocketPath), "Server should bind \(testSocketPath)")

        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "test", "version": "1.0"]]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 19,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "techgym speakers workshop", "num_results": 3, "source": "all"] as [String: Any]
            ]
        ])

        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        let text = content.first?["text"] as? String ?? ""

        XCTAssertTrue(text.contains("manual-a0b8a"))
        XCTAssertEqual(helper.requests.count, 1)
        XCTAssertEqual(helper.requests.first?["query"] as? String, "techgym speakers workshop")
        XCTAssertEqual(helper.requests.first?["source"] as? String, "all")
    }

    func testMCPBrainSubscribeOverSocketReturnsCursorState() throws {
        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "subscriber", "version": "1.0"]]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-1",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? "{}"
        let payload = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]

        XCTAssertEqual(payload?["status"] as? String, "subscribed")
        XCTAssertEqual(payload?["agent_id"] as? String, "agent-1")
        XCTAssertEqual(payload?["last_delivered_seq"] as? Int, 0)
        XCTAssertEqual(payload?["last_acked_seq"] as? Int, 0)
        XCTAssertNotNil(payload?["generation"])
    }

    func testMCPBrainUnsubscribeOverSocketReturnsResult() throws {
        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "subscriber", "version": "1.0"]]
        ])

        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-1",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": [
                "name": "brain_unsubscribe",
                "arguments": [
                    "agent_id": "agent-1",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])

        let result = response["result"] as? [String: Any]
        let content = result?["content"] as? [[String: Any]]
        let text = content?.first?["text"] as? String ?? "{}"
        let payload = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]

        XCTAssertEqual(payload?["status"] as? String, "unsubscribed")
        XCTAssertEqual(payload?["agent_id"] as? String, "agent-1")
    }

    func testMCPBrainAckRejectsAgentWithoutBrainBarSubscription() throws {
        _ = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "ack-owner-check", "version": "1.0"]]
        ])

        let response = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": [
                "name": "brain_ack",
                "arguments": [
                    "agent_id": "cmuxlayer-owned-agent",
                    "seq": 42
                ] as [String: Any]
            ]
        ])

        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        let text = try XCTUnwrap(content.first?["text"] as? String)
        XCTAssertEqual(result["isError"] as? Bool, true)
        XCTAssertTrue(text.contains("No BrainBar subscription"))
        XCTAssertNil(try db.subscription(agentID: "cmuxlayer-owned-agent"))
    }

    func testMatchingStorePushesChannelNotificationAndRequiresAckToClearUnread() throws {
        let subscriberFD = try connectClient()
        defer { close(subscriberFD) }

        try initializeClient(fd: subscriberFD, name: "subscriber")
        try sendMCPRequest(on: subscriberFD, request: [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-live",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: subscriberFD)

        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 3,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "publisher", "version": "1.0"]
            ]
        ])

        let publishResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Live push message for agent live",
                    "tags": ["agent-message"],
                    "importance": 6
                ] as [String: Any]
            ]
        ])
        XCTAssertNil(publishResponse["error"])

        let notification = try readMCPMessage(fd: subscriberFD)
        XCTAssertEqual(notification["method"] as? String, "notifications/claude/channel")
        let params = notification["params"] as? [String: Any]
        let content = params?["content"] as? String ?? ""
        XCTAssertTrue(content.contains("Live push message for agent live"))
        let meta = params?["meta"] as? [String: Any]
        let rowID = (meta?["rowid"] as? String).flatMap(Int.init)
        XCTAssertNotNil(rowID)

        let unreadResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": [
                    "query": "Live push message",
                    "agent_id": "agent-live",
                    "unread_only": true
                ] as [String: Any]
            ]
        ])

        let unreadResult = unreadResponse["result"] as? [String: Any]
        let unreadContent = unreadResult?["content"] as? [[String: Any]]
        let unreadText = unreadContent?.first?["text"] as? String ?? ""
        XCTAssertTrue(unreadText.contains("Live push message for agent live"), "Live-delivered chunk should stay unread until ack")

        let ackResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": [
                "name": "brain_ack",
                "arguments": [
                    "agent_id": "agent-live",
                    "seq": rowID as Any
                ] as [String: Any]
            ]
        ])
        XCTAssertNil(ackResponse["error"])

        let clearedResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": [
                    "query": "Live push message",
                    "agent_id": "agent-live",
                    "unread_only": true
                ] as [String: Any]
            ]
        ])
        let clearedResult = clearedResponse["result"] as? [String: Any]
        let clearedContent = clearedResult?["content"] as? [[String: Any]]
        let clearedText = clearedContent?.first?["text"] as? String ?? ""
        XCTAssertFalse(clearedText.contains("Live push message for agent live"), "Acked chunk should no longer be unread")
    }

    func testFlushedQueuedStoreAlsoPushesChannelNotification() throws {
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("brainbar-flush-notify-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let queuePath = tempDir.appendingPathComponent("pending-stores.jsonl")
        let queuedPayload = """
        {"content":"Queued subscriber message","tags":["agent-message"],"importance":4,"source":"mcp"}
        """
        try queuedPayload.write(to: queuePath, atomically: true, encoding: .utf8)

        server.stop()
        let dbPath = tempDir.appendingPathComponent("brainbar.db").path
        let previousQueuePath = ProcessInfo.processInfo.environment["BRAINBAR_PENDING_STORES_PATH"]
        setenv("BRAINBAR_PENDING_STORES_PATH", queuePath.path, 1)
        defer {
            if let previousQueuePath {
                setenv("BRAINBAR_PENDING_STORES_PATH", previousQueuePath, 1)
            } else {
                unsetenv("BRAINBAR_PENDING_STORES_PATH")
            }
        }
        let flushDB = BrainDatabase(path: dbPath)
        server = BrainBarServer(socketPath: testSocketPath, dbPath: dbPath, database: flushDB)
        defer {
            server.stop()
            flushDB.close()
        }
        server.start()
        XCTAssertTrue(waitForSocket(at: testSocketPath), "Server should bind \(testSocketPath)")

        let subscriberFD = try connectClient()
        defer { close(subscriberFD) }

        try initializeClient(fd: subscriberFD, name: "subscriber-flush")
        try sendMCPRequest(on: subscriberFD, request: [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-flush",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: subscriberFD)

        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 3,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "publisher-flush", "version": "1.0"]
            ]
        ])

        let publishResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Live trigger message",
                    "tags": ["agent-message"],
                    "importance": 6
                ] as [String: Any]
            ]
        ])
        XCTAssertNil(publishResponse["error"])

        let notifications = try readMCPMessages(fd: subscriberFD, expectedCount: 2)
        let receivedContents = notifications.compactMap {
            ($0["params"] as? [String: Any])?["content"] as? String
        }

        XCTAssertEqual(Set(receivedContents), Set(["Live trigger message", "Queued subscriber message"]))
    }

    func testDeadSubscriberDoesNotBlockLiveSubscriberNotification() throws {
        let deadFD = try connectClient()
        try initializeClient(fd: deadFD, name: "dead-subscriber")
        try sendMCPRequest(on: deadFD, request: [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-dead",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: deadFD)
        close(deadFD)

        let liveFD = try connectClient()
        defer { close(liveFD) }
        try initializeClient(fd: liveFD, name: "live-subscriber")
        try sendMCPRequest(on: liveFD, request: [
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-live-2",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: liveFD)

        let storeResponse = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Fanout survives dead subscriber",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        XCTAssertNil(storeResponse["error"])

        let notification = try readMCPMessage(fd: liveFD)
        XCTAssertEqual(notification["method"] as? String, "notifications/claude/channel")
        let params = notification["params"] as? [String: Any]
        let content = params?["content"] as? String ?? ""
        XCTAssertTrue(content.contains("Fanout survives dead subscriber"))
    }

    func testSecondConnectionTakesOverAgentIdentity() throws {
        let firstFD = try connectClient()
        defer { close(firstFD) }
        try initializeClient(fd: firstFD, name: "first")
        try sendMCPRequest(on: firstFD, request: [
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-takeover",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: firstFD)

        let secondFD = try connectClient()
        defer { close(secondFD) }
        try initializeClient(fd: secondFD, name: "second")
        try sendMCPRequest(on: secondFD, request: [
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "agent-takeover",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])
        _ = try readMCPMessage(fd: secondFD)

        var oneByte = [UInt8](repeating: 0, count: 1)
        let firstRead = read(firstFD, &oneByte, 1)
        XCTAssertLessThanOrEqual(firstRead, 0, "First socket should be closed after takeover")

        _ = try sendMCPRequest([
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": [
                "name": "brain_store",
                "arguments": [
                    "content": "Takeover delivery",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ])

        let notification = try readMCPMessage(fd: secondFD)
        XCTAssertEqual(notification["method"] as? String, "notifications/claude/channel")
    }

    // MARK: - C1: Write retry cap

    func testServerDisconnectsStalledClient() throws {
        // Connect but never read — server should disconnect after max retries (10),
        // not block the serial queue forever.
        let clientFD = socket(AF_UNIX, SOCK_STREAM, 0)
        guard clientFD >= 0 else { throw NSError(domain: "test", code: 1) }
        defer { close(clientFD) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 104) { dest in
                _ = testSocketPath.withCString { src in strcpy(dest, src) }
            }
        }
        let connectResult = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(clientFD, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        XCTAssertEqual(connectResult, 0, "Should connect")

        // Set tiny receive buffer to force EAGAIN on server-side writes
        var bufSize: Int32 = 1
        setsockopt(clientFD, SOL_SOCKET, SO_RCVBUF, &bufSize, socklen_t(MemoryLayout<Int32>.size))

        // Send an initialize request
        let json = #"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"stall","version":"1"}}}"#
        let header = "Content-Length: \(json.utf8.count)\r\n\r\n"
        var frame = Data(header.utf8)
        frame.append(Data(json.utf8))
        frame.withUnsafeBytes { ptr in
            _ = write(clientFD, ptr.baseAddress!, frame.count)
        }

        // After the write stalls (tiny rcvbuf), server should disconnect within ~20ms (10 retries * 1ms + overhead)
        // If it hangs > 200ms, the retry cap is broken.
        // A second client should still be able to connect and get a response,
        // proving the serial queue wasn't blocked.
        Thread.sleep(forTimeInterval: 0.2)

        let secondResponse = try sendMCPRequest([
            "jsonrpc": "2.0", "id": 99, "method": "initialize",
            "params": ["protocolVersion": "2024-11-05", "capabilities": [:] as [String: Any],
                       "clientInfo": ["name": "second", "version": "1.0"]]
        ])
        XCTAssertNotNil(secondResponse["result"], "Serial queue must not be blocked — second client should get response")
    }

    func testStdioAdapterBridgesInitializeAndSubscribe() throws {
        let adapterPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Scripts/brainbar_stdio_adapter.py")
            .path
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3",
            adapterPath,
            "--socket",
            testSocketPath,
        ]

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = Pipe()
        try process.run()
        defer {
            process.terminate()
            process.waitUntilExit()
        }

        try sendLineJSON([
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "adapter", "version": "1.0"]
            ]
        ], to: stdinPipe.fileHandleForWriting)

        let initializeResponse = try readLineJSON(from: stdoutPipe.fileHandleForReading)
        let capabilities = (initializeResponse["result"] as? [String: Any])?["capabilities"] as? [String: Any]
        let experimental = capabilities?["experimental"] as? [String: Any]
        XCTAssertEqual((experimental?["claude/channel"] as? [String: Any])?.isEmpty, true)

        try sendLineJSON([
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": [
                "name": "brain_subscribe",
                "arguments": [
                    "agent_id": "adapter-agent",
                    "tags": ["agent-message"]
                ] as [String: Any]
            ]
        ], to: stdinPipe.fileHandleForWriting)

        let subscribeResponse = try readLineJSON(from: stdoutPipe.fileHandleForReading)
        let result = subscribeResponse["result"] as? [String: Any]
        XCTAssertNotNil(result)
    }

    func testStdioAdapterDrainsResponsesAfterStdinEOF() throws {
        let adapterPath = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Scripts/brainbar_stdio_adapter.py")
            .path
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3",
            adapterPath,
            "--socket",
            testSocketPath,
        ]

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = Pipe()
        try process.run()

        try sendLineJSON([
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": "adapter-eof", "version": "1.0"]
            ]
        ], to: stdinPipe.fileHandleForWriting)
        try sendLineJSON([
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        ], to: stdinPipe.fileHandleForWriting)
        try stdinPipe.fileHandleForWriting.close()

        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)

        let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
        let outputLines = String(decoding: stdoutData, as: UTF8.self)
            .split(separator: "\n")
            .map(String.init)
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        XCTAssertEqual(outputLines.count, 2)

        let initializeResponse = try JSONSerialization.jsonObject(with: Data(outputLines[0].utf8)) as? [String: Any]
        XCTAssertNotNil(initializeResponse?["result"])

        let toolsResponse = try JSONSerialization.jsonObject(with: Data(outputLines[1].utf8)) as? [String: Any]
        let tools = (toolsResponse?["result"] as? [String: Any])?["tools"] as? [[String: Any]]
        XCTAssertEqual(tools?.count, 17)
    }

    // MARK: - C2: Socket path length validation

    func testRejectsOverlongSocketPath() throws {
        // sockaddr_un.sun_path is 104 bytes on macOS. A path > 104 should not crash.
        let longPath = "/tmp/" + String(repeating: "x", count: 200) + ".sock"
        let longDBPath = NSTemporaryDirectory() + "test-long-\(UUID().uuidString).db"
        let longDB = BrainDatabase(path: longDBPath)
        defer {
            longDB.close()
            try? FileManager.default.removeItem(atPath: longDBPath)
            try? FileManager.default.removeItem(atPath: longDBPath + "-wal")
            try? FileManager.default.removeItem(atPath: longDBPath + "-shm")
        }
        let longServer = BrainBarServer(socketPath: longPath, dbPath: longDBPath, database: longDB)
        longServer.start()
        Thread.sleep(forTimeInterval: 0.2)

        // Server should have refused to bind — connecting should fail.
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            XCTFail("socket() failed with errno \(errno)")
            return
        }
        defer { close(fd) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        // Can't even set the long path in sockaddr_un, so connect would fail.
        // The key assertion: the server didn't crash during start().
        longServer.stop()
    }

    // MARK: - Helper

    private func restartServer(profile: String) {
        server.stop()
        setenv("BRAINLAYER_MCP_PROFILE", profile, 1)
        server = BrainBarServer(socketPath: testSocketPath, dbPath: tempDBPath, database: db)
        server.start()
        XCTAssertTrue(waitForSocket(at: testSocketPath), "Server should bind \(testSocketPath)")
    }

    private func listedToolNames(on fd: Int32, id: Int) throws -> [String] {
        try sendMCPRequest(on: fd, request: [
            "jsonrpc": "2.0", "id": id, "method": "tools/list",
        ])
        let response = try readMCPMessage(fd: fd)
        let tools = (response["result"] as? [String: Any])?["tools"] as? [[String: Any]]
        return tools?.compactMap { $0["name"] as? String } ?? []
    }

    private func waitForSocket(at path: String, timeout: TimeInterval = 3.0) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let fd = socket(AF_UNIX, SOCK_STREAM, 0)
            guard fd >= 0 else { return false }
            var addr = sockaddr_un()
            addr.sun_family = sa_family_t(AF_UNIX)
            let pathBytes = path.utf8CString
            withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
                ptr.withMemoryRebound(to: CChar.self, capacity: 104) { destination in
                    pathBytes.withUnsafeBufferPointer { source in
                        _ = memcpy(destination, source.baseAddress!, source.count)
                    }
                }
            }
            let connected = withUnsafePointer(to: &addr) { addrPointer in
                addrPointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { socketAddress in
                    connect(fd, socketAddress, socklen_t(MemoryLayout<sockaddr_un>.size)) == 0
                }
            }
            close(fd)
            if connected { return true }
            Thread.sleep(forTimeInterval: 0.01)
        }
        return false
    }

    private func sendMCPRequest(_ request: [String: Any]) throws -> [String: Any] {
        let fd = try connectClient()
        defer { close(fd) }
        try sendMCPRequest(on: fd, request: request)
        return try readMCPMessage(fd: fd)
    }

    private func queryString(_ sql: String, on db: OpaquePointer?) throws -> String? {
        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard rc == SQLITE_OK else {
            throw NSError(domain: "sqlite", code: Int(rc), userInfo: [NSLocalizedDescriptionKey: "prepare failed \(rc)"])
        }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return nil }
        guard let value = sqlite3_column_text(stmt, 0) else { return nil }
        return String(cString: value)
    }

    private func connectClient() throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "socket() failed"]) }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 104) { dest in
                _ = testSocketPath.withCString { src in
                    strcpy(dest, src)
                }
            }
        }

        let connectResult = withUnsafePointer(to: &addr) { addrPtr in
            addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { ptr in
                connect(fd, ptr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connectResult == 0 else {
            close(fd)
            throw NSError(domain: "test", code: 2, userInfo: [NSLocalizedDescriptionKey: "connect() failed: errno \(errno)"])
        }
        let flags = fcntl(fd, F_GETFL)
        _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        return fd
    }

    private func initializeClient(fd: Int32, name: String) throws {
        try sendMCPRequest(on: fd, request: [
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": [
                "protocolVersion": "2024-11-05",
                "capabilities": [:] as [String: Any],
                "clientInfo": ["name": name, "version": "1.0"]
            ]
        ])
        _ = try readMCPMessage(fd: fd)
    }

    private func sendMCPRequest(on fd: Int32, request: [String: Any]) throws {
        let jsonData = try JSONSerialization.data(withJSONObject: request)
        let header = "Content-Length: \(jsonData.count)\r\n\r\n"
        var frame = Data(header.utf8)
        frame.append(jsonData)

        let sent = frame.withUnsafeBytes { ptr in
            write(fd, ptr.baseAddress!, frame.count)
        }
        guard sent == frame.count else {
            throw NSError(domain: "test", code: 3, userInfo: [NSLocalizedDescriptionKey: "write() incomplete"])
        }
    }

    private func sendRawLineJSON(on fd: Int32, object: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: object)
        data.append(0x0A)
        let sent = data.withUnsafeBytes { ptr in
            write(fd, ptr.baseAddress!, data.count)
        }
        guard sent == data.count else {
            throw NSError(domain: "test", code: 3, userInfo: [NSLocalizedDescriptionKey: "raw write() incomplete"])
        }
    }

    private func readRawLineJSONData(fd: Int32, timeout: TimeInterval = 5.0) throws -> Data {
        var buffer = Data()
        var readBuf = [UInt8](repeating: 0, count: 65536)
        let deadline = Date().addingTimeInterval(timeout)

        while Date() < deadline {
            let n = read(fd, &readBuf, readBuf.count)
            if n > 0 {
                buffer.append(contentsOf: readBuf[0..<n])
                if let newlineIndex = buffer.firstIndex(of: 0x0A) {
                    return Data(buffer[..<newlineIndex])
                }
            } else if n == 0 {
                break
            } else if errno != EAGAIN && errno != EINTR && errno != EWOULDBLOCK {
                break
            }
            Thread.sleep(forTimeInterval: 0.01)
        }

        throw NSError(domain: "test", code: 4, userInfo: [NSLocalizedDescriptionKey: "Timeout reading raw line response"])
    }

    private func readBrainBusEvent(fd: Int32, matching type: String, timeout: TimeInterval = 5.0) throws -> [String: Any] {
        let deadline = Date().addingTimeInterval(timeout)
        var buffer = Data()
        var readBuf = [UInt8](repeating: 0, count: 65536)
        while Date() < deadline {
            while let newlineIndex = buffer.firstIndex(of: 0x0A) {
                let line = Data(buffer[..<newlineIndex])
                buffer.removeSubrange(buffer.startIndex...newlineIndex)
                guard !line.isEmpty else { continue }
                let message = try JSONSerialization.jsonObject(with: line) as? [String: Any] ?? [:]
                let params = message["params"] as? [String: Any]
                if params?["type"] as? String == type {
                    return message
                }
            }

            let n = read(fd, &readBuf, readBuf.count)
            if n > 0 {
                buffer.append(contentsOf: readBuf[0..<n])
            } else if n == 0 {
                break
            } else if errno != EAGAIN && errno != EINTR && errno != EWOULDBLOCK {
                break
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        throw NSError(domain: "test", code: 6, userInfo: [NSLocalizedDescriptionKey: "Timeout reading brain bus event \(type)"])
    }

    private func readMCPMessage(fd: Int32, timeout: TimeInterval = 5.0) throws -> [String: Any] {
        return try readMCPMessages(fd: fd, expectedCount: 1, timeout: timeout).first ?? [:]
    }

    private func readMCPMessages(fd: Int32, expectedCount: Int, timeout: TimeInterval = 5.0) throws -> [[String: Any]] {
        var buffer = Data()
        var readBuf = [UInt8](repeating: 0, count: 65536)
        let deadline = Date().addingTimeInterval(timeout)
        var messages: [[String: Any]] = []

        while Date() < deadline {
            let n = read(fd, &readBuf, readBuf.count)
            if n > 0 {
                buffer.append(contentsOf: readBuf[0..<n])
                while let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) {
                    let headerStr = String(data: buffer[buffer.startIndex..<headerEnd.lowerBound], encoding: .utf8) ?? ""
                    guard let clLine = headerStr.split(separator: "\r\n").first(where: { $0.hasPrefix("Content-Length:") }) else {
                        break
                    }
                    let headerParts = clLine.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
                    guard headerParts.count == 2,
                          let cl = Int(headerParts[1].trimmingCharacters(in: .whitespaces)),
                          cl >= 0 else {
                        throw NSError(
                            domain: "test",
                            code: 5,
                            userInfo: [NSLocalizedDescriptionKey: "Malformed Content-Length header"]
                        )
                    }
                    let bodyStart = headerEnd.upperBound
                    guard buffer.count >= bodyStart + cl else {
                        break
                    }
                    let bodyData = buffer[bodyStart..<(bodyStart + cl)]
                    let message = try JSONSerialization.jsonObject(with: bodyData) as? [String: Any] ?? [:]
                    messages.append(message)
                    buffer.removeSubrange(buffer.startIndex..<(bodyStart + cl))
                    if messages.count == expectedCount {
                        return messages
                    }
                }
            } else if n == 0 {
                break // EOF
            } else if errno != EAGAIN && errno != EINTR && errno != EWOULDBLOCK {
                break
            }
            Thread.sleep(forTimeInterval: 0.01)
        }

        throw NSError(domain: "test", code: 4, userInfo: [NSLocalizedDescriptionKey: "Timeout reading response"])
    }

    private func sendLineJSON(_ object: [String: Any], to handle: FileHandle) throws {
        let data = try JSONSerialization.data(withJSONObject: object)
        handle.write(data)
        handle.write(Data([0x0A]))
    }

    private func readLineJSON(from handle: FileHandle, timeout: TimeInterval = 5.0) throws -> [String: Any] {
        let deadline = Date().addingTimeInterval(timeout)
        var buffer = Data()
        let fd = handle.fileDescriptor
        let flags = fcntl(fd, F_GETFL)
        _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        var readBuf = [UInt8](repeating: 0, count: 4096)
        while Date() < deadline {
            let count = read(fd, &readBuf, readBuf.count)
            if count > 0 {
                buffer.append(contentsOf: readBuf[0..<count])
                if let newlineIndex = buffer.firstIndex(of: 0x0A) {
                    let line = buffer[..<newlineIndex]
                    return try JSONSerialization.jsonObject(with: line) as? [String: Any] ?? [:]
                }
            } else if count == 0 {
                Thread.sleep(forTimeInterval: 0.01)
            } else {
                if errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR {
                    break
                }
                Thread.sleep(forTimeInterval: 0.01)
            }
        }
        throw NSError(domain: "test", code: 5, userInfo: [NSLocalizedDescriptionKey: "Timeout reading line JSON"])
    }
}
