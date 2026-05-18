import Foundation

@main
enum BrainBarDaemonMain {
    static func main() {
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
        RunLoop.main.run()
    }
}
