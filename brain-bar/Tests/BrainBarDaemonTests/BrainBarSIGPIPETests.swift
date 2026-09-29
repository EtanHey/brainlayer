import Foundation
import XCTest

final class BrainBarSIGPIPETests: XCTestCase {
    func testClosedChildStdinReturnsHandledErrorWithoutKillingProcess() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let lifecycleSource = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBarLifecycle/BrainBarSignalSafety.swift")
        let probeSource = temporary.appendingPathComponent("probe.swift")
        let probeBinary = temporary.appendingPathComponent("probe")
        try #"""
        import Foundation

        @main
        struct Probe {
            static func main() throws {
                BrainBarSignalSafety.ignoreSIGPIPE()
                let child = Process()
                child.executableURL = URL(fileURLWithPath: "/bin/sh")
                child.arguments = ["-c", "exec 0<&-; echo ready; sleep 1"]
                let input = Pipe()
                let output = Pipe()
                child.standardInput = input
                child.standardOutput = output
                try child.run()
                _ = output.fileHandleForReading.availableData
                let payload = Data(repeating: 65, count: 1024)
                var handled = 0
                for _ in 0..<16 {
                    if !BrainBarSignalSafety.write(payload, to: input.fileHandleForWriting, context: "probe") {
                        handled += 1
                    }
                }
                child.waitUntilExit()
                if handled > 0 {
                    print("HANDLED_EPIPE")
                } else {
                    Foundation.exit(2)
                }
            }
        }
        """#.write(to: probeSource, atomically: true, encoding: .utf8)

        let compiler = Process()
        compiler.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        compiler.arguments = ["swiftc", "-parse-as-library", lifecycleSource.path, probeSource.path, "-o", probeBinary.path]
        let compilerOutput = Pipe()
        compiler.standardOutput = compilerOutput
        compiler.standardError = compilerOutput
        try compiler.run()
        let compileMessage = String(data: compilerOutput.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        compiler.waitUntilExit()
        XCTAssertEqual(compiler.terminationStatus, 0, compileMessage)
        guard compiler.terminationStatus == 0 else { return }

        let probe = Process()
        probe.executableURL = probeBinary
        let probeOutput = Pipe()
        probe.standardOutput = probeOutput
        probe.standardError = probeOutput
        try probe.run()
        let output = String(data: probeOutput.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        probe.waitUntilExit()
        XCTAssertEqual(probe.terminationStatus, 0, "probe terminated: \(probe.terminationReason) \(output)")
        XCTAssertTrue(output.contains("HANDLED_EPIPE"), output)
    }
}
