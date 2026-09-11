import BrainBarLifecycle
import Foundation

@main
enum BrainBarDaemonMain {
    static func main() {
        switch BrainBarDaemonStartupMode.parse(
            arguments: CommandLine.arguments,
            environment: ProcessInfo.processInfo.environment
        ) {
        case .failure(let reason):
            fputs("[BrainBarDaemon] Startup rejected: \(reason)\n", stderr)
            Foundation.exit(2)
        case .success(let mode):
            let uiWatchdog = mode == .normal ? startUIWatchdog() : nil
            let server = BrainBarServer(
                socketPath: mode.socketPath,
                dbPath: mode.dbPath,
                enableHybridSearchHelper: mode == .normal,
                enableDaemonHeartbeat: mode == .normal
            )
        server.onStartRejected = { reason in
            NSLog("[BrainBarDaemon] Startup rejected: %@", reason)
            Foundation.exit(1)
        }
        server.onDatabaseReady = { _ in
            NSLog("[BrainBarDaemon] Database ready")
        }
        server.start()
        NSLog("[BrainBarDaemon] Started on %@", mode.socketPath ?? BrainBarServer.defaultSocketPath())
            withExtendedLifetime(server) {
                withExtendedLifetime(uiWatchdog) {
                    RunLoop.main.run()
                }
            }
        }
    }

    private static func startUIWatchdog() -> BrainBarLifecycleWatchdog {
        let watchdog = BrainBarLifecycleWatchdog.makeUIWatchdog()
        watchdog.start()
        return watchdog
    }
}
