import XCTest
@testable import BrainBar

extension AgentActivityMonitor {
    /// A monitor for tests: an empty process table (measured, no sessions) and no kernel
    /// path lookups, so a collector refresh never touches live processes (#990 N1).
    static let synthetic = AgentActivityMonitor(snapshotProvider: { "" }, executablePathResolver: { _ in nil })
}

/// #990 N1: no unit test samples the live process table. A `StatsCollector` built without an
/// injected monitor ran real `ps` and then `proc_pidpath` on live PIDs whenever it refreshed.
/// Every collector and monitor a test builds takes a synthetic snapshot and resolver.
final class AgentActivityMonitorIsolationTests: XCTestCase {
    private static let constructorsNeedingAMonitor = ["StatsCollector(", "makeStatsCollector(", "makeUIStatsCollector("]
    private static let liveMonitorTokens = ["AgentActivityMonitor.live", "agentActivityMonitor: .live", "kernelExecutablePath"]

    func test_no_test_builds_a_collector_or_monitor_that_samples_live_processes() throws {
        let testsDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let thisFile = URL(fileURLWithPath: #filePath).lastPathComponent
        let files = try FileManager.default.contentsOfDirectory(at: testsDirectory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "swift" && $0.lastPathComponent != thisFile }
        XCTAssertFalse(files.isEmpty)

        var offenders: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            let name = file.lastPathComponent
            for constructor in Self.constructorsNeedingAMonitor {
                for (line, arguments) in Self.calls(of: constructor, in: text)
                where !arguments.contains("agentActivityMonitor:") {
                    offenders.append("\(name):\(line) \(constructor)…) without agentActivityMonitor:")
                }
            }
            for (line, arguments) in Self.calls(of: "AgentActivityMonitor(", in: text)
            where !(arguments.contains("snapshotProvider:") && arguments.contains("executablePathResolver:")) {
                offenders.append("\(name):\(line) AgentActivityMonitor(…) without a synthetic snapshot and resolver")
            }
            for token in Self.liveMonitorTokens where text.contains(token) {
                offenders.append("\(name): uses \(token)")
            }
        }
        XCTAssertEqual(offenders, [], "unit tests must inject a synthetic AgentActivityMonitor")
    }

    /// Each call of `callee` (not part of a longer identifier) with its line number and the
    /// text between its balanced parentheses.
    private static func calls(of callee: String, in text: String) -> [(line: Int, arguments: String)] {
        var results: [(Int, String)] = []
        var searchStart = text.startIndex
        while let match = text.range(of: callee, range: searchStart..<text.endIndex) {
            searchStart = match.upperBound
            if match.lowerBound > text.startIndex {
                let previous = text[text.index(before: match.lowerBound)]
                if previous.isLetter || previous.isNumber || previous == "_" { continue }
            }
            var depth = 1
            var index = match.upperBound
            while index < text.endIndex, depth > 0 {
                if text[index] == "(" { depth += 1 } else if text[index] == ")" { depth -= 1 }
                index = text.index(after: index)
            }
            let line = text[..<match.lowerBound].filter { $0 == "\n" }.count + 1
            results.append((line, String(text[match.upperBound..<index])))
        }
        return results
    }
}
