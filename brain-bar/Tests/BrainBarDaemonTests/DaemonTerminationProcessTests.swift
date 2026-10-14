import Foundation
import XCTest

final class DaemonTerminationProcessTests: XCTestCase {
    func testSIGTERMRetainsFinalDropsAcrossRealProcessRestart() throws { try replay("SIGTERM") }
    func testSIGINTRetainsFinalDropsAcrossRealProcessRestart() throws { try replay("SIGINT") }

    private func replay(_ number: String) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let scripts = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("BrainBarTests/Scripts")
        let binary = Bundle(for: Self.self).bundleURL.deletingLastPathComponent().appendingPathComponent("BrainBarDaemon")
        XCTAssertTrue(FileManager.default.isExecutableFile(atPath: binary.path), "The real daemon product is required")
        let library = directory.appendingPathComponent("guard.dylib")
        let compiler = Process()
        compiler.executableURL = URL(fileURLWithPath: "/usr/bin/clang")
        compiler.arguments = ["-dynamiclib", "-lsqlite3", scripts.appendingPathComponent("daemon_shutdown_guard.c").path, "-o", library.path]
        try compiler.run()
        compiler.waitUntilExit()
        XCTAssertEqual(compiler.terminationStatus, 0)
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = [scripts.appendingPathComponent("daemon_shutdown_process.py").path, binary.path, library.path, number]
        process.standardOutput = output
        process.standardError = output
        let ended = expectation(description: "owned daemon replay reaped")
        process.terminationHandler = { _ in ended.fulfill() }
        try process.run()
        wait(for: [ended], timeout: 40)
        if process.isRunning { process.terminate(); process.waitUntilExit() }
        let text = String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        XCTAssertEqual(process.terminationStatus, 0, text)
        XCTAssertTrue(text.contains("\"requests_dropped\": 3"), text)
        print("real_daemon_termination_receipt=" + text)
    }
}
