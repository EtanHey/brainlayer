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
}
