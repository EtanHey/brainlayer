// BrainBarDebugLogGateTests.swift — BL-0.2: the debug log is off by default,
// 0600 + rotated behind BRAINBAR_DEBUG_LOG=1, and never carries a payload.

import BrainBarLifecycle
import Darwin
import XCTest
@testable import BrainBar

final class BrainBarLogFileTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-logfile-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func path(_ name: String) -> String { directory.appendingPathComponent(name).path }

    private func mode(_ path: String) -> mode_t? {
        var info = stat()
        guard lstat(path, &info) == 0 else { return nil }
        return info.st_mode & 0o777
    }

    private func size(_ path: String) -> Int {
        var info = stat()
        guard lstat(path, &info) == 0 else { return -1 }
        return Int(info.st_size)
    }

    func testDebugLogIsOffUnlessTheFlagIsExactlyOne() {
        XCTAssertEqual(BrainBarLog.debugLogFlag, "BRAINBAR_DEBUG_LOG")
        XCTAssertEqual(BrainBarLog.debugLogPath, "/tmp/brainbar-debug.log")
        XCTAssertNil(BrainBarLog.debugLogFile(environment: [:]))
        XCTAssertNil(BrainBarLog.debugLogFile(environment: ["BRAINBAR_DEBUG_LOG": "0"]))
        XCTAssertNil(BrainBarLog.debugLogFile(environment: ["BRAINBAR_DEBUG_LOG": ""]))
        let file = BrainBarLog.debugLogFile(environment: ["BRAINBAR_DEBUG_LOG": "1"], path: path("debug.log"))
        XCTAssertEqual(file?.path, path("debug.log"))
        XCTAssertEqual(file?.maxBytes, 5 * 1024 * 1024)
    }

    func testAppendCreatesTheFileWithMode0600() {
        let target = path("new.log")
        BrainBarLogFile(path: target, maxBytes: 4096).append("hello")
        XCTAssertEqual(mode(target), 0o600)
        XCTAssertTrue(((try? String(contentsOfFile: target, encoding: .utf8)) ?? "").contains("hello"))
    }

    func testAppendTightensAnExisting0644File() throws {
        let target = path("old.log")
        FileManager.default.createFile(atPath: target, contents: Data("legacy\n".utf8), attributes: [.posixPermissions: 0o644])
        XCTAssertEqual(mode(target), 0o644)
        BrainBarLogFile(path: target, maxBytes: 4096).append("next")
        XCTAssertEqual(mode(target), 0o600)
    }

    func testTightenExistingOnlyChmodsAndNeverCreates() {
        let missing = path("missing.log")
        BrainBarLogFile(path: missing, maxBytes: 4096).tightenExisting()
        XCTAssertFalse(FileManager.default.fileExists(atPath: missing))

        let existing = path("existing.log")
        FileManager.default.createFile(atPath: existing, contents: Data("keep\n".utf8), attributes: [.posixPermissions: 0o644])
        BrainBarLogFile(path: existing, maxBytes: 4096).tightenExisting()
        XCTAssertEqual(mode(existing), 0o600)
        XCTAssertEqual(try? String(contentsOfFile: existing, encoding: .utf8), "keep\n", "Tightening must not rewrite or truncate.")
    }

    func testRotationCapsTheFileAtMaxBytesTimesTwo() {
        let target = path("rotating.log")
        let file = BrainBarLogFile(path: target, maxBytes: 512)
        for index in 0..<200 {
            file.append("line \(index) padding padding padding")
        }
        XCTAssertLessThanOrEqual(size(target), 512)
        XCTAssertGreaterThan(size(target), 0)
        XCTAssertLessThanOrEqual(size(file.rotatedPath), 512)
        XCTAssertEqual(mode(target), 0o600)
        XCTAssertEqual(mode(file.rotatedPath), 0o600)
        XCTAssertFalse(FileManager.default.fileExists(atPath: target + ".2"), "Only one rotated generation is kept.")
        let current = (try? String(contentsOfFile: target, encoding: .utf8)) ?? ""
        XCTAssertTrue(current.contains("line 199"), current)
    }

    func testAppendRefusesASymlinkedPath() throws {
        let victim = path("victim.txt")
        FileManager.default.createFile(atPath: victim, contents: Data("untouched".utf8), attributes: [.posixPermissions: 0o644])
        let link = path("link.log")
        try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: victim)
        BrainBarLogFile(path: link, maxBytes: 4096).append("should not land")
        XCTAssertEqual(try String(contentsOfFile: victim, encoding: .utf8), "untouched")
        XCTAssertEqual(mode(victim), 0o644, "A symlink must not let the log chmod another file.")
    }

    func testAppendCreatesAMissingParentDirectoryPrivately() {
        let nested = directory.appendingPathComponent("Logs/BrainBar/lifecycle.log").path
        BrainBarLogFile(path: nested, maxBytes: 4096).append("started")
        XCTAssertEqual(mode(nested), 0o600)
        XCTAssertEqual(mode((nested as NSString).deletingLastPathComponent), 0o700)
    }

    func testLifecycleLogIsSeparateFromTheDebugLog() {
        XCTAssertNotEqual(BrainBarLog.lifecycleLogPath, BrainBarLog.debugLogPath)
        XCTAssertTrue(BrainBarLog.lifecycleLogPath.hasSuffix("/Library/Logs/BrainBar/lifecycle.log"), BrainBarLog.lifecycleLogPath)
        XCTAssertFalse(BrainBarLog.lifecycleLogPath.hasPrefix("/tmp/"))
    }
}

/// The real server on a scratch socket, with scratch log paths. Never touches
/// /tmp/brainbar.sock, /tmp/brainbar-debug.log or the real lifecycle log.
final class BrainBarServerDebugLogGateTests: XCTestCase {
    private static let syntheticQuery = "zqx-synthetic-query-4471"
    private static let syntheticToken = "sk-ant-api03-SYNTHETICTOKEN0000000000000000000000000000"
    private static let syntheticContent = "synthetic-stored-content-9912"
    private static let syntheticID = "id-synthetic-7310"
    private static let syntheticMethod = "method-synthetic-5528"

    private var directory: URL!
    private var socketPath: String!
    private var server: BrainBarServer?
    private var db: BrainDatabase?

    override func setUpWithError() throws {
        let short = UUID().uuidString.prefix(8)
        directory = URL(fileURLWithPath: "/tmp/bbdl-\(short)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        socketPath = directory.appendingPathComponent("s.sock").path
    }

    override func tearDownWithError() throws {
        server?.stop()
        db?.close()
        try? FileManager.default.removeItem(at: directory)
    }

    private var debugLogPath: String { directory.appendingPathComponent("debug.log").path }
    private var lifecycleLogPath: String { directory.appendingPathComponent("lifecycle.log").path }

    private func startServer(environment: [String: String]) throws {
        let dbPath = directory.appendingPathComponent("brainbar.db").path
        let database = BrainDatabase(path: dbPath)
        db = database
        let server = BrainBarServer(
            socketPath: socketPath,
            dbPath: dbPath,
            database: database,
            enableHybridSearchHelper: false,
            diagnostics: .daemon(
                environment: environment,
                debugLogPath: debugLogPath,
                lifecycleLogPath: lifecycleLogPath
            )
        )
        self.server = server
        server.start()
        let deadline = Date().addingTimeInterval(5)
        var answered = false
        while !answered, Date() < deadline {
            answered = BrainBarLifecycleWatchdog.socketAnswersPing(path: socketPath, timeout: 1)
            if !answered { Thread.sleep(forTimeInterval: 0.02) }
        }
        XCTAssertTrue(answered, "The scratch server must answer on \(socketPath)")
    }

    /// Sends payload-shaped traffic: a search query, a stored secret, a
    /// payload-shaped request id, and an unknown method name.
    private func sendPayloadTraffic() throws {
        let fd = try connect()
        defer { close(fd) }
        let requests: [[String: Any]] = [
            ["jsonrpc": "2.0", "id": 1, "method": "initialize", "params": ["protocolVersion": "2024-11-05", "capabilities": [:], "clientInfo": ["name": Self.syntheticToken, "version": "1"]]],
            ["jsonrpc": "2.0", "id": Self.syntheticID, "method": "tools/call", "params": ["name": "brain_search", "arguments": ["query": Self.syntheticQuery]]],
            ["jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": ["name": "brain_store", "arguments": ["content": "\(Self.syntheticContent) \(Self.syntheticToken)"]]],
            ["jsonrpc": "2.0", "id": 4, "method": Self.syntheticMethod, "params": ["token": Self.syntheticToken]],
        ]
        for request in requests {
            let body = try JSONSerialization.data(withJSONObject: request)
            var framed = Data("Content-Length: \(body.count)\r\n\r\n".utf8)
            framed.append(body)
            _ = framed.withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
            _ = readFramedResponse(fd)
        }
    }

    private func connect() throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let bytes = Array(socketPath.utf8)
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            raw.copyBytes(from: bytes)
            raw[bytes.count] = 0
        }
        let rc = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard rc == 0 else {
            close(fd)
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        var timeout = timeval(tv_sec: 5, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        return fd
    }

    private func readFramedResponse(_ fd: Int32) -> Data {
        var buffer = Data()
        var chunk = [UInt8](repeating: 0, count: 65536)
        while true {
            if let header = buffer.range(of: Data("\r\n\r\n".utf8)),
               let text = String(data: buffer[..<header.lowerBound], encoding: .utf8),
               let length = Int(text.replacingOccurrences(of: "Content-Length:", with: "").trimmingCharacters(in: .whitespaces)),
               buffer.count >= header.upperBound + length {
                return buffer
            }
            let n = read(fd, &chunk, chunk.count)
            guard n > 0 else { return buffer }
            buffer.append(contentsOf: chunk[0..<n])
        }
    }

    private func contents(_ path: String) -> String {
        (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
    }

    private func assertNoPayload(in text: String, _ label: String, file: StaticString = #filePath, line: UInt = #line) {
        for secret in [Self.syntheticQuery, Self.syntheticToken, Self.syntheticContent, Self.syntheticID, Self.syntheticMethod, "HEX:", "TEXT:"] {
            XCTAssertFalse(text.contains(secret), "\(label) leaked \(secret):\n\(text)", file: file, line: line)
        }
    }

    private func mode(_ path: String) -> mode_t? {
        var info = stat()
        guard lstat(path, &info) == 0 else { return nil }
        return info.st_mode & 0o777
    }

    func testDefaultServerNeverCreatesTheDebugLog() throws {
        try startServer(environment: [:])
        try sendPayloadTraffic()
        XCTAssertFalse(FileManager.default.fileExists(atPath: debugLogPath), contents(debugLogPath))
    }

    func testDefaultServerDoesNotAppendToAnExistingDebugLog() throws {
        FileManager.default.createFile(atPath: debugLogPath, contents: Data("legacy\n".utf8), attributes: [.posixPermissions: 0o644])
        try startServer(environment: [:])
        try sendPayloadTraffic()
        XCTAssertEqual(contents(debugLogPath), "legacy\n")
        XCTAssertEqual(mode(debugLogPath), 0o600, "A stale world-readable debug log is tightened at startup, never appended.")
    }

    func testServerStartIsRecordedInTheLifecycleLogWithoutPayloads() throws {
        try startServer(environment: [:])
        try sendPayloadTraffic()
        let lifecycle = contents(lifecycleLogPath)
        XCTAssertTrue(lifecycle.contains("SERVER STARTED"), lifecycle)
        XCTAssertEqual(mode(lifecycleLogPath), 0o600)
        assertNoPayload(in: lifecycle, "lifecycle log")
    }

    func testFlaggedDebugLogIs0600AndCarriesNoPayloads() throws {
        try startServer(environment: ["BRAINBAR_DEBUG_LOG": "1"])
        try sendPayloadTraffic()
        let debug = contents(debugLogPath)
        XCTAssertTrue(debug.contains("CLIENT CONNECTED"), debug)
        XCTAssertTrue(debug.contains("method=tools/call"), debug)
        XCTAssertTrue(debug.contains("method=<other>"), debug)
        XCTAssertEqual(mode(debugLogPath), 0o600)
        assertNoPayload(in: debug, "debug log")
        assertNoPayload(in: contents(lifecycleLogPath), "lifecycle log")
    }

    func testLoggableMethodIsAnAllowlist() {
        XCTAssertEqual(BrainBarServer.loggableMethod(["method": "tools/call"]), "tools/call")
        XCTAssertEqual(BrainBarServer.loggableMethod(["method": "initialize"]), "initialize")
        XCTAssertEqual(BrainBarServer.loggableMethod(["method": Self.syntheticToken]), "<other>")
        XCTAssertEqual(BrainBarServer.loggableMethod([:]), "<no method>")
    }
}
