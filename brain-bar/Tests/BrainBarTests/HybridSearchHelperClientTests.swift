import Darwin
import XCTest
@testable import BrainBar

final class HybridSearchHelperClientTests: XCTestCase {
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

    func testResolvePythonPathUsesRepoSourceDirectoryWhenUnset() throws {
        let repoRoot = NSTemporaryDirectory() + "brainbar-helper-src-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: "\(repoRoot)/src",
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(atPath: repoRoot) }

        let resolved = HybridSearchHelperClient.resolvePythonPath(environment: [
            "BRAINLAYER_REPO_ROOT": repoRoot
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
}
