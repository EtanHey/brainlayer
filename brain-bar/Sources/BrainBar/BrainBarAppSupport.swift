import AppKit
import Darwin
import Foundation

enum BrainBarAppSupport {
    static func hotkeyPermissionFailureMessage(permissions: HotkeyPermissionStatus) -> String {
        "BrainBar could not start the fallback hotkey listener. Enable \(permissions.missingPermissionsMessage) in System Settings. The CGEventTap fallback requires both Input Monitoring and Accessibility."
    }

    @MainActor
    static func makeStatsCollector(
        dbPath: String,
        targetPID: pid_t,
        brainBusEvents: BrainBusEventSource? = BrainBusClient(),
        watcherProcessProbe: any WatcherProcessProbing = LaunchctlWatcherProcessProbe(),
        databaseOpenConfiguration: BrainDatabase.OpenConfiguration = BrainDatabase.OpenConfiguration()
    ) -> StatsCollector {
        makeStatsCollector(
            dbPath: dbPath,
            daemonPIDResolver: FixedDaemonPIDResolver(pid: targetPID),
            brainBusEvents: brainBusEvents,
            watcherProcessProbe: watcherProcessProbe,
            databaseOpenConfiguration: databaseOpenConfiguration
        )
    }

    @MainActor
    static func makeStatsCollector(
        dbPath: String,
        daemonPIDResolver: any DaemonPIDResolving,
        brainBusEvents: BrainBusEventSource? = BrainBusClient(),
        watcherProcessProbe: any WatcherProcessProbing = LaunchctlWatcherProcessProbe(),
        databaseOpenConfiguration: BrainDatabase.OpenConfiguration = BrainDatabase.OpenConfiguration()
    ) -> StatsCollector {
        StatsCollector(
            dbPath: dbPath,
            daemonMonitor: DaemonHealthMonitor(pidResolver: daemonPIDResolver),
            watcherProcessProbe: watcherProcessProbe,
            brainBusEvents: brainBusEvents,
            databaseOpenConfiguration: databaseOpenConfiguration
        )
    }

    @MainActor
    static func makeUIStatsCollector(
        dbPath: String,
        brainBusEvents: BrainBusEventSource? = BrainBusClient(),
        daemonPIDResolver: any DaemonPIDResolving = LiveDaemonPIDResolver()
    ) -> StatsCollector {
        // The resolver runs on every sample, never once at launch: the daemon
        // restarts (wake, crash, upgrade) while this UI keeps running (#972).
        makeStatsCollector(
            dbPath: dbPath,
            daemonPIDResolver: daemonPIDResolver,
            brainBusEvents: brainBusEvents,
            databaseOpenConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
    }

    // AIDEV-NOTE: Wires the UI runtime's database after PR #312
    // removed the FastAPI daemon. Pre-#312 the daemon owned the writer and the UI
    // process consumed via socket; post-#312 each consumer opens SQLite directly.
    //
    // On a missing DB, bootstrap the schema once before installing the read-only
    // handle so fresh installs still work. The UI runtime stays read-only so the
    // writer pidfile remains uncontended with the Python enrich supervisor + drain.
    @MainActor
    static func wireRuntime(
        _ runtime: BrainBarRuntime,
        dbPath: String,
        collector: StatsCollector
    ) {
        let database = BrainDatabase(
            path: dbPath,
            openConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        if !database.isOpen {
            database.reopenIfNeeded()
        }
        if !database.isOpen, !FileManager.default.fileExists(atPath: dbPath) {
            let bootstrapDatabase = BrainDatabase(path: dbPath)
            bootstrapDatabase.close()
            database.reopenIfNeeded()
        }
        if !database.isOpen {
            NSLog(
                "[BrainBar] Read-only database open failed at %@: %@",
                dbPath,
                String(describing: database.lastOpenError)
            )
        }

        runtime.install(
            collector: collector,
            database: database,
            databasePath: dbPath
        )
    }

    static func daemonPIDFromFile(_ path: String) -> pid_t? {
        guard let pid = LiveDaemonPIDResolver.pidFromFile(path),
              LiveDaemonPIDResolver().isDaemon(pid) else { return nil }
        return pid
    }
}
