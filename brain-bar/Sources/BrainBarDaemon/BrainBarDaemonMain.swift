import BrainBarLifecycle
import Foundation

@main
enum BrainBarDaemonMain {
    static func main() {
        BrainBarSignalSafety.ignoreSIGPIPE()
        let uiWatchdog = startUIWatchdog()
        // BRAINBAR_DEBUG_LOG=1 in this process's environment (the LaunchAgent's
        // EnvironmentVariables, or `launchctl setenv` before a kickstart) turns
        // on /tmp/brainbar-debug.log. Without it the server writes only unified
        // logging and the lifecycle log.
        let server = BrainBarServer(diagnostics: .daemon())
        server.onStartRejected = { reason in
            NSLog("[BrainBarDaemon] Startup rejected: %@", reason)
            Foundation.exit(1)
        }
        server.onDatabaseReady = { _ in
            NSLog("[BrainBarDaemon] Database ready")
        }
        server.start()
        NSLog("[BrainBarDaemon] Started on %@", BrainBarServer.defaultSocketPath())
        withExtendedLifetime(server) {
            withExtendedLifetime(uiWatchdog) {
                RunLoop.main.run()
            }
        }
    }

    private static func startUIWatchdog() -> BrainBarLifecycleWatchdog {
        let watchdog = BrainBarLifecycleWatchdog.makeUIWatchdog()
        watchdog.start()
        return watchdog
    }
}
