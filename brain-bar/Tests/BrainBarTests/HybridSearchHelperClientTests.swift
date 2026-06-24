import Darwin
import XCTest
@testable import BrainBar

final class HybridSearchHelperClientTests: XCTestCase {
    deinit {}

    func testResolvePythonExecutableUsesInstallTimeRepoRoot() throws {
        let repoRoot = NSTemporaryDirectory() + "brainbar-helper-repo-\(UUID().uuidString)"
        let pythonPath = "\(repoRoot)/.venv/bin/python"
        try FileManager.default.createDirectory(
            atPath: "\(repoRoot)/.venv/bin",
            withIntermediateDirectories: true
        )
        FileManager.default.createFile(atPath: pythonPath, contents: Data("#!/bin/sh\n".utf8))
        chmod(pythonPath, 0o755)
        defer { try? FileManager.default.removeItem(atPath: repoRoot) }

        let resolved = HybridSearchHelperClient.resolvePythonExecutable(environment: [
            "BRAINLAYER_REPO_ROOT": repoRoot,
            "PATH": "/nonexistent"
        ])

        XCTAssertEqual(resolved, pythonPath)
    }

    func testResolvePythonExecutablePrefersHomebrewVenvOverBarePython3() throws {
        // On a Homebrew clone with no repo checkout, bare `python3` on PATH is the
        // system interpreter (e.g. python3.14) that lacks the `brainlayer` module.
        // The formula venv at <prefix>/opt/brainlayer/libexec/venv/bin/python MUST win.
        let prefix = NSTemporaryDirectory() + "brainbar-brew-\(UUID().uuidString)"
        let venvPython = "\(prefix)/opt/brainlayer/libexec/venv/bin/python"
        try FileManager.default.createDirectory(
            atPath: "\(prefix)/opt/brainlayer/libexec/venv/bin",
            withIntermediateDirectories: true
        )
        FileManager.default.createFile(atPath: venvPython, contents: Data("#!/bin/sh\n".utf8))
        chmod(venvPython, 0o755)

        let pathDir = NSTemporaryDirectory() + "brainbar-brew-path-\(UUID().uuidString)"
        let barePython3 = "\(pathDir)/python3"
        try FileManager.default.createDirectory(atPath: pathDir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: barePython3, contents: Data("#!/bin/sh\n".utf8))
        chmod(barePython3, 0o755)

        defer {
            try? FileManager.default.removeItem(atPath: prefix)
            try? FileManager.default.removeItem(atPath: pathDir)
        }

        let resolved = HybridSearchHelperClient.resolvePythonExecutable(environment: [
            "HOMEBREW_PREFIX": prefix,
            "PATH": pathDir
        ])

        XCTAssertEqual(resolved, venvPython)
    }

    func testResolvePythonPathPrefersInstalledPackageWhenUnset() throws {
        let repoRoot = NSTemporaryDirectory() + "brainbar-helper-src-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: "\(repoRoot)/src",
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(atPath: repoRoot) }

        let resolved = HybridSearchHelperClient.resolvePythonPath(environment: [
            "BRAINLAYER_REPO_ROOT": repoRoot
        ])

        XCTAssertNil(resolved)
    }

    func testResolvePythonPathUsesRepoSourceDirectoryOnlyWhenFallbackRequested() throws {
        let repoRoot = NSTemporaryDirectory() + "brainbar-helper-src-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: "\(repoRoot)/src",
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(atPath: repoRoot) }

        let resolved = HybridSearchHelperClient.resolvePythonPath(environment: [
            "BRAINLAYER_REPO_ROOT": repoRoot,
            "BRAINLAYER_SOURCE_FALLBACK": "1"
        ])

        XCTAssertEqual(resolved, "\(repoRoot)/src")
    }

    func testResolvePythonPathPreservesExistingPythonPath() throws {
        let resolved = HybridSearchHelperClient.resolvePythonPath(environment: [
            "PYTHONPATH": "/custom/pythonpath",
            "BRAINLAYER_REPO_ROOT": "/ignored"
        ])

        XCTAssertEqual(resolved, "/custom/pythonpath")
    }

    func testSearchReportsLaunchFailureWithoutSocketRetry() throws {
        let client = HybridSearchHelperClient(
            socketPath: "/tmp/bb-missing-\(UUID().uuidString).sock",
            dbPath: "/tmp/brainlayer-test.db",
            pythonExecutable: "/no/such/python",
            environment: [:]
        )

        do {
            _ = try client.search(arguments: ["query": "techgym speakers workshop"])
            XCTFail("Expected launch failure")
        } catch let error as HybridSearchHelperError {
            guard case .launch = error else {
                return XCTFail("Expected launch error, got \(error)")
            }
        }
    }

    func testSearchRejectsTooLongSocketPathBeforeLaunch() throws {
        let longPath = "/tmp/" + String(repeating: "x", count: 200) + ".sock"
        let client = HybridSearchHelperClient(
            socketPath: longPath,
            dbPath: "/tmp/brainlayer-test.db",
            pythonExecutable: "/no/such/python",
            environment: [:]
        )

        do {
            _ = try client.search(arguments: ["query": "techgym speakers workshop"])
            XCTFail("Expected socket path validation failure")
        } catch let error as HybridSearchHelperError {
            guard case .socketPathTooLong(let path) = error else {
                return XCTFail("Expected socket path error, got \(error)")
            }
            XCTAssertEqual(path, longPath)
        }
    }

    func testConfigureSocketTimeoutsMakesReadsFinite() throws {
        var fds: [Int32] = [0, 0]
        XCTAssertEqual(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds), 0)
        defer {
            close(fds[0])
            close(fds[1])
        }

        try HybridSearchHelperClient.configureSocketTimeouts(fd: fds[0], timeout: 0.05)

        var byte = UInt8(0)
        let startedAt = Date()
        let count = read(fds[0], &byte, 1)
        let elapsed = Date().timeIntervalSince(startedAt)

        XCTAssertEqual(count, -1)
        XCTAssertTrue(errno == EAGAIN || errno == EWOULDBLOCK)
        XCTAssertLessThan(elapsed, 1.0)
    }

    func testConfigureNoSigpipeAcceptsOpenUnixSocket() throws {
        var fds: [Int32] = [0, 0]
        XCTAssertEqual(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds), 0)
        defer {
            close(fds[0])
            close(fds[1])
        }

        try HybridSearchHelperClient.configureNoSigpipe(fd: fds[0])

        var value: Int32 = 0
        var length = socklen_t(MemoryLayout<Int32>.size)
        XCTAssertEqual(getsockopt(fds[0], SOL_SOCKET, SO_NOSIGPIPE, &value, &length), 0)
        XCTAssertEqual(value, 1)
    }

    func testBrainBarHelperSocketTimeoutIsDecoupledFromSearchBudget() {
        XCTAssertGreaterThan(
            BrainBarServer.hybridHelperSocketIOTimeoutSeconds,
            BrainBarServer.hybridSearchBudgetSeconds * 10
        )
    }

    func testStartWarmingMarksHelperReadyAfterStatusHandshake() throws {
        let fixture = try makeFakeHybridHelperFixture(searchDelay: 0.01, warmDelay: 0.15)
        defer { fixture.cleanup() }

        let client = HybridSearchHelperClient(
            socketPath: fixture.socketPath,
            dbPath: fixture.dbPath,
            pythonExecutable: fixture.executablePath,
            environment: [:],
            socketIOTimeout: 2.0,
            readinessTimeout: 2.0,
            readinessProbeInterval: 0.02
        )
        defer { client.stop() }

        client.startWarming()

        XCTAssertTrue(waitUntil(timeout: 2.0) { client.isReady })
        XCTAssertTrue(client.isRunning)

        let response = try client.search(arguments: ["query": "ready helper"])
        XCTAssertEqual(response.text, "fake hybrid result")
    }

    func testRouterBudgetMissDoesNotStopReadyHelperProcess() throws {
        let fixture = try makeFakeHybridHelperFixture(searchDelay: 0.25, warmDelay: 0.0)
        defer { fixture.cleanup() }
        let tempDB = fixture.dbPath
        let db = BrainDatabase(path: tempDB)
        defer { db.close() }
        try db.insertChunk(
            id: "budget-miss-fallback",
            content: "Budget miss fallback content from Swift database",
            sessionId: "s1",
            project: "brainlayer",
            contentType: "assistant_text",
            importance: 7
        )

        let client = HybridSearchHelperClient(
            socketPath: fixture.socketPath,
            dbPath: fixture.dbPath,
            pythonExecutable: fixture.executablePath,
            environment: [:],
            socketIOTimeout: 2.0,
            readinessTimeout: 5.0,
            readinessProbeInterval: 0.02,
            readinessProbeTimeout: 0.05
        )
        defer { client.stop() }
        client.startWarming()
        XCTAssertTrue(waitUntil(timeout: 5.0) { client.isReady })

        let router = MCPRouter(hybridSearchClient: client, hybridSearchBudget: 0.05)
        router.setDatabase(db)

        let started = Date()
        let response = router.handle([
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": [
                "name": "brain_search",
                "arguments": ["query": "budget miss fallback", "num_results": 5]
            ] as [String: Any]
        ])
        let elapsed = Date().timeIntervalSince(started)

        XCTAssertLessThan(elapsed, 0.20)
        let text = try toolText(response)
        XCTAssertTrue(text.contains("Budget miss fallback content"), text)
        XCTAssertTrue(waitUntil(timeout: 1.0) { client.isRunning && client.isReady })
    }

    func testStopDoesNotWaitForeverWhenHelperIgnoresTerminate() throws {
        let fixture = try makeFakeHybridHelperFixture(
            searchDelay: 0.01,
            warmDelay: 0.0,
            ignoreTerminate: true
        )
        defer { fixture.cleanup() }

        let client = HybridSearchHelperClient(
            socketPath: fixture.socketPath,
            dbPath: fixture.dbPath,
            pythonExecutable: fixture.executablePath,
            environment: [:],
            socketIOTimeout: 2.0,
            readinessTimeout: 2.0,
            readinessProbeInterval: 0.02
        )
        client.startWarming()
        XCTAssertTrue(waitUntil(timeout: 2.0) { client.isReady })

        let started = Date()
        client.stop()

        XCTAssertLessThan(Date().timeIntervalSince(started), 3.5)
        XCTAssertFalse(client.isRunning)
    }

    private struct FakeHybridHelperFixture {
        let directory: URL
        let executablePath: String
        let socketPath: String
        let dbPath: String

        func cleanup() {
            try? FileManager.default.removeItem(at: directory)
            try? FileManager.default.removeItem(atPath: socketPath)
        }
    }

    private func makeFakeHybridHelperFixture(
        searchDelay: TimeInterval,
        warmDelay: TimeInterval,
        ignoreTerminate: Bool = false
    ) throws -> FakeHybridHelperFixture {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("brainbar-fake-helper-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let scriptURL = directory.appendingPathComponent("fake-helper.py")
        let socketPath = "/tmp/bb-fake-\(UUID().uuidString.prefix(8)).sock"
        try? FileManager.default.removeItem(atPath: socketPath)
        let dbPath = directory.appendingPathComponent("brain.db").path
        let script = """
#!/usr/bin/env python3
import json
import os
import signal
import socket
import sys
import time

socket_path = sys.argv[sys.argv.index("--socket-path") + 1]
if \(ignoreTerminate ? "True" : "False"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
try:
    os.unlink(socket_path)
except FileNotFoundError:
    pass
time.sleep(\(warmDelay))
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(socket_path)
server.listen(8)
while True:
    conn, _ = server.accept()
    with conn:
        data = b""
        while b"\\n" not in data:
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
        if not data:
            continue
        request = json.loads(data.split(b"\\n", 1)[0].decode("utf-8"))
        if request.get("method") == "status":
            response = {"ok": True, "ready": True}
        elif request.get("method") == "brain_search":
            time.sleep(\(searchDelay))
            response = {"ok": True, "text": "fake hybrid result", "metadata": {}}
        else:
            response = {"ok": False, "error": "unsupported"}
        conn.sendall(json.dumps(response).encode("utf-8") + b"\\n")
"""
        try script.write(to: scriptURL, atomically: true, encoding: .utf8)
        chmod(scriptURL.path, 0o755)
        return FakeHybridHelperFixture(
            directory: directory,
            executablePath: scriptURL.path,
            socketPath: socketPath,
            dbPath: dbPath
        )
    }

    private func waitUntil(timeout: TimeInterval, predicate: () -> Bool) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if predicate() {
                return true
            }
            RunLoop.current.run(until: Date().addingTimeInterval(0.01))
        }
        return predicate()
    }

    private func toolText(_ response: [String: Any]) throws -> String {
        let result = try XCTUnwrap(response["result"] as? [String: Any])
        let content = try XCTUnwrap(result["content"] as? [[String: Any]])
        return content.compactMap { $0["text"] as? String }.joined(separator: "\n")
    }
}
