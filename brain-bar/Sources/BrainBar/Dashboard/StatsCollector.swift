import Darwin
import Foundation
import SQLite3
import SwiftUI

@_silgen_name("notify_register_dispatch")
private func brainbar_notify_register_dispatch(
    _ name: UnsafePointer<CChar>,
    _ outToken: UnsafeMutablePointer<Int32>,
    _ queue: DispatchQueue,
    _ handler: @escaping @convention(block) (Int32) -> Void
) -> UInt32

@_silgen_name("notify_cancel")
private func brainbar_notify_cancel(_ token: Int32) -> UInt32

@MainActor
final class StatsCollector: ObservableObject {
    @Published private(set) var stats: DashboardStats
    @Published private(set) var daemon: DaemonHealthSnapshot?
    @Published private(set) var state: PipelineState

    private let database: BrainDatabase
    private let daemonMonitor: DaemonHealthMonitor
    private let changeObserver: DatabaseChangeObserver

    init(
        dbPath: String,
        daemonMonitor: DaemonHealthMonitor,
        notificationName: String = "com.brainlayer.db.changed"
    ) {
        self.database = BrainDatabase(path: dbPath)
        self.daemonMonitor = daemonMonitor
        self.stats = DashboardStats(
            chunkCount: 0,
            enrichedChunkCount: 0,
            pendingEnrichmentCount: 0,
            enrichmentPercent: 0,
            databaseSizeBytes: 0,
            recentActivityBuckets: Array(repeating: 0, count: 12)
        )
        self.state = .offline
        self.changeObserver = DatabaseChangeObserver(dbPath: dbPath, notificationName: notificationName)
    }

    func start() {
        refresh(force: true)
        changeObserver.start { [weak self] in
            Task { @MainActor [weak self] in
                self?.refresh(force: true)
            }
        }
    }

    func stop() {
        changeObserver.stop()
        database.close()
    }

    func refresh(force: Bool = false) {
        do {
            let nextStats = try database.dashboardStats(activityWindowMinutes: 30, bucketCount: 12)
            let nextDaemon = daemonMonitor.sample()
            stats = nextStats
            daemon = nextDaemon
            state = PipelineState.derive(daemon: nextDaemon, stats: nextStats)
        } catch {
            if force {
                daemon = nil
                state = .offline
            }
        }
    }
}

private final class DatabaseChangeObserver: @unchecked Sendable {
    private let dbPath: String
    private let notificationName: String
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.dashboard-observer", qos: .utility)

    private var db: OpaquePointer?
    private var token: Int32 = 0
    private var lastDataVersion: Int32 = -1
    private var timer: DispatchSourceTimer?
    private var onChange: (@Sendable () -> Void)?

    init(dbPath: String, notificationName: String) {
        self.dbPath = dbPath
        self.notificationName = notificationName
    }

    func start(onChange: @escaping @Sendable () -> Void) {
        self.onChange = onChange
        queue.async { [weak self] in
            self?.startOnQueue()
        }
    }

    func stop() {
        queue.sync {
            if token != 0 {
                _ = brainbar_notify_cancel(token)
                token = 0
            }
            timer?.cancel()
            timer = nil
            if let db {
                sqlite3_close_v2(db)
                self.db = nil
            }
        }
    }

    private func startOnQueue() {
        guard db == nil else { return }

        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_WAL
        guard sqlite3_open_v2(dbPath, &handle, flags, nil) == SQLITE_OK, let handle else {
            return
        }

        db = handle
        lastDataVersion = readDataVersion()
        registerForDarwinNotifications()

        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + .seconds(2), repeating: .seconds(2))
        timer.setEventHandler { [weak self] in
            self?.emitIfChanged()
        }
        timer.resume()
        self.timer = timer
    }

    private func registerForDarwinNotifications() {
        notificationName.withCString { name in
            _ = brainbar_notify_register_dispatch(name, &token, queue) { [weak self] (_: Int32) in
                self?.emitIfChanged()
            }
        }
    }

    private func emitIfChanged() {
        let currentVersion = readDataVersion()
        guard currentVersion != lastDataVersion else { return }
        lastDataVersion = currentVersion
        onChange?()
    }

    private func readDataVersion() -> Int32 {
        guard let db else { return -1 }
        guard sqlite3_get_autocommit(db) != 0 else { return lastDataVersion }

        var stmt: OpaquePointer?
        let rc = sqlite3_prepare_v2(db, "PRAGMA data_version", -1, &stmt, nil)
        guard rc == SQLITE_OK else { return lastDataVersion }
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else { return lastDataVersion }
        return sqlite3_column_int(stmt, 0)
    }
}
