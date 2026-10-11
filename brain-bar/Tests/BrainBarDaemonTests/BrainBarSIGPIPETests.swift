import Foundation
import XCTest

final class BrainBarSIGPIPETests: XCTestCase {
    func testPreclosedAcceptedSocketSurvivesSIGPIPE() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }

        let lifecycleSource = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/BrainBarLifecycle/BrainBarSignalSafety.swift")
        let probeSource = temporary.appendingPathComponent("socket-probe.swift")
        let probeBinary = temporary.appendingPathComponent("socket-probe")
        try #"""
        import Darwin
        import Foundation

        @main
        struct Probe {
            static func main() {
                _ = Darwin.signal(SIGPIPE, SIG_DFL)
                if CommandLine.arguments.contains("--guard") {
                    BrainBarSignalSafety.ignoreSIGPIPE()
                }

                let listener = socket(AF_UNIX, SOCK_STREAM, 0)
                precondition(listener >= 0)
                defer { close(listener) }
                var address = sockaddr_un()
                address.sun_family = sa_family_t(AF_UNIX)
                address.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
                withUnsafeMutablePointer(to: &address.sun_path) {
                    $0.withMemoryRebound(to: CChar.self, capacity: 104) {
                        _ = strcpy($0, "socket-probe.sock")
                    }
                }
                let addressLength = socklen_t(MemoryLayout<sockaddr_un>.size)
                let bound = withUnsafePointer(to: &address) {
                    $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                        bind(listener, $0, addressLength)
                    }
                }
                precondition(bound == 0)
                precondition(listen(listener, 1) == 0)

                let client = socket(AF_UNIX, SOCK_STREAM, 0)
                precondition(client >= 0)
                let connected = withUnsafePointer(to: &address) {
                    $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                        connect(client, $0, addressLength)
                    }
                }
                precondition(connected == 0)
                let request = Array("{\"jsonrpc\":\"2.0\"}\n".utf8)
                precondition(request.withUnsafeBytes { Darwin.write(client, $0.baseAddress!, $0.count) } == request.count)
                close(client) // Fully closed before accept, unlike shutdown(SHUT_WR).

                let accepted = accept(listener, nil, nil)
                precondition(accepted >= 0)
                defer { close(accepted) }
                var noSigpipe: Int32 = 1
                let optionResult = setsockopt(accepted, SOL_SOCKET, SO_NOSIGPIPE, &noSigpipe, socklen_t(MemoryLayout<Int32>.size))
                precondition(optionResult == -1 && errno == EINVAL, "expected preclosed-peer EINVAL")
                print("NOSIGPIPE_EINVAL")
                fflush(stdout)

                var buffer = [UInt8](repeating: 0, count: 256)
                let readCount = Darwin.read(accepted, &buffer, buffer.count)
                precondition(readCount == request.count)
                let response = Array("{\"result\":{}}\n".utf8)
                let writeCount = response.withUnsafeBytes { Darwin.write(accepted, $0.baseAddress!, $0.count) }
                precondition(writeCount == -1 && errno == EPIPE, "expected EPIPE after SIGPIPE ignore")
                print("HANDLED_EPIPE")
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

        func runProbe(_ argument: String) throws -> (Process, String) {
            let probe = Process()
            probe.executableURL = probeBinary
            probe.arguments = [argument]
            probe.currentDirectoryURL = temporary
            let outputPipe = Pipe()
            probe.standardOutput = outputPipe
            probe.standardError = outputPipe
            try probe.run()
            let output = String(data: outputPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            probe.waitUntilExit()
            return (probe, output)
        }

        let (control, controlOutput) = try runProbe("--control")
        XCTAssertEqual(control.terminationReason, .uncaughtSignal, controlOutput)
        XCTAssertEqual(control.terminationStatus, SIGPIPE, controlOutput)
        XCTAssertTrue(controlOutput.contains("NOSIGPIPE_EINVAL"), controlOutput)

        // The control dies before its socket path is unlinked; remove it for the guarded run.
        try FileManager.default.removeItem(at: temporary.appendingPathComponent("socket-probe.sock"))
        let (guarded, guardedOutput) = try runProbe("--guard")
        XCTAssertEqual(guarded.terminationReason, .exit, guardedOutput)
        XCTAssertEqual(guarded.terminationStatus, 0, guardedOutput)
        XCTAssertTrue(guardedOutput.contains("NOSIGPIPE_EINVAL"), guardedOutput)
        XCTAssertTrue(guardedOutput.contains("HANDLED_EPIPE"), guardedOutput)
    }

    func testBothEntrypointsInstallSIGPIPEGuardFirst() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources")
        let daemon = try String(contentsOf: sources.appendingPathComponent("BrainBarDaemon/BrainBarDaemonMain.swift"), encoding: .utf8)
        let mainPattern = #"static\s+func\s+main\s*\(\s*\)\s*\{\s*BrainBarSignalSafety\.ignoreSIGPIPE\(\)"#
        XCTAssertNotNil(daemon.range(of: mainPattern, options: .regularExpression), "daemon guard must be the first main statement")

        let ui = try String(contentsOf: sources.appendingPathComponent("BrainBar/main.swift"), encoding: .utf8)
        let firstUIStatement = ui.split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty && !$0.hasPrefix("import ") && !$0.hasPrefix("//") }
        XCTAssertEqual(firstUIStatement, "BrainBarSignalSafety.ignoreSIGPIPE()")
    }

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
