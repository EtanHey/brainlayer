import Foundation

// Compile with the production reader/status/formatter and the production process-evidence enum.
@main
struct WatcherHeartbeatProbe {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count == 4, let now = WatcherHealthReader.date(args[2]) else {
            throw NSError(domain: "heartbeat-probe-arguments", code: 1)
        }
        let process: WatcherLaunchdEvidence = args[3] == "stopped" ? .notRunning("private producer exited") : .running
        let status = WatcherHealthStatus.derive(
            launchd: process, file: WatcherHealthReader.read(url: URL(fileURLWithPath: args[1])), now: now
        )
        let state: String = switch status {
        case .running: "running"
        case .degraded: "degraded"
        case .stopped: "stopped"
        case .unknown: "unknown"
        }
        let output = ["state": state, "reason": status.reasonText(now: now) ?? ""]
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]))
    }
}
