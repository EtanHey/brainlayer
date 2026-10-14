import Foundation
import XCTest
@testable import BrainBar

final class BrainBarWorkspaceIsolationTests: XCTestCase {
    func testFixtureRootIsSuiteOwnedWhenUsingTheLeakRunner() {
        guard let root = ProcessInfo.processInfo.environment["BRAINBAR_TEST_TMP_ROOT"] else { return }
        XCTAssertEqual(brainBarTestTemporaryDirectory().standardizedFileURL.path, URL(fileURLWithPath: root).standardizedFileURL.path)
    }

    func testConstructionProbe() {
        guard ProcessInfo.processInfo.environment["BRAINBAR_WORKSPACE_PROBE"] == "1" else { return }
        _ = BrainBarWorkspace()
    }

    func testRealWorkspaceConstructionFailsUnderXCTest() throws {
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        child.arguments = ["xctest", "-XCTest", "BrainBarTests.BrainBarWorkspaceIsolationTests/testConstructionProbe", Bundle(for: Self.self).bundleURL.path]
        child.environment = ProcessInfo.processInfo.environment.merging([
            "BRAINBAR_WORKSPACE_PROBE": "1"
        ]) { _, new in new }
        let output = Pipe()
        child.standardOutput = output
        child.standardError = output
        try child.run()
        let text = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        child.waitUntilExit()
        XCTAssertNotEqual(child.terminationStatus, 0, text)
        XCTAssertTrue(text.contains("Inject BrainBarWorkspaceActing in tests"), text)
    }

    func testEnvironmentGuardAlsoProtectsAnAlreadyConstructedWorkspace() throws {
        let root = brainBarTestTemporaryDirectory().appendingPathComponent("brainbar-workspace-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: root) }
        let main = root.appendingPathComponent("main.swift")
        try """
        import Foundation
        let workspace = BrainBarWorkspace()
        setenv("XCTestConfigurationFilePath", "/test/configuration", 1)
        switch CommandLine.arguments[1] {
        case "reveal": workspace.reveal(URL(fileURLWithPath: "/nonexistent/fixture"))
        case "open": workspace.open(URL(fileURLWithPath: "/nonexistent/fixture"))
        default: workspace.copy("fixture")
        }
        """.write(to: main, atomically: true, encoding: .utf8)
        let binary = root.appendingPathComponent("probe")
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/BrainBarWorkspace.swift")
        let compiler = Process()
        compiler.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        compiler.arguments = ["swiftc", source.path, main.path, "-o", binary.path]
        try compiler.run()
        compiler.waitUntilExit()
        XCTAssertEqual(compiler.terminationStatus, 0)
        for action in ["reveal", "open", "copy"] {
            let child = Process()
            child.executableURL = binary
            child.arguments = [action]
            var environment = ProcessInfo.processInfo.environment
            environment.removeValue(forKey: "XCTestConfigurationFilePath")
            child.environment = environment
            let output = Pipe()
            child.standardOutput = output
            child.standardError = output
            try child.run()
            let text = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            child.waitUntilExit()
            XCTAssertNotEqual(child.terminationStatus, 0, action)
            XCTAssertTrue(text.contains("Inject BrainBarWorkspaceActing in tests"), text)
        }
    }
}
