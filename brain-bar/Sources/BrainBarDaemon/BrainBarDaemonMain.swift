import BrainBarLifecycle
import Foundation

@main
enum BrainBarDaemonMain {
    static func main() {
        let uiWatchdog = startUIWatchdog()
        let server = BrainBarServer()
        server.onStartRejected = { reason in
            NSLog("[BrainBarDaemon] Startup rejected: %@", reason)
            Foundation.exit(1)
        }
        server.onDatabaseReady = { _ in
            NSLog("[BrainBarDaemon] Database ready")
        }
        server.start()
        NSLog("[BrainBarDaemon] Started on %@", BrainBarServer.defaultSocketPath())
        withExtendedLifetime(uiWatchdog) {
            RunLoop.main.run()
        }
    }

    private static func startUIWatchdog() -> BrainBarLifecycleWatchdog {
        let watchdog = BrainBarLifecycleWatchdog.makeUIWatchdog()
        watchdog.start()
        return watchdog
    }
}
