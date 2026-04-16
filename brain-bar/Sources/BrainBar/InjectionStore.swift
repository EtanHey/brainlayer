import CoreFoundation
import Foundation

private func injectionStoreDarwinNotificationCallback(
    center: CFNotificationCenter?,
    observer: UnsafeMutableRawPointer?,
    name: CFNotificationName?,
    object: UnsafeRawPointer?,
    userInfo: CFDictionary?
) {
    guard let observer else { return }
    let store = Unmanaged<InjectionStore>.fromOpaque(observer).takeUnretainedValue()
    Task { @MainActor in
        store.handleDatabaseMutationNotification()
    }
}

@MainActor
final class InjectionStore: ObservableObject {
    @Published private(set) var events: [InjectionEvent] = []

    private let database: BrainDatabase
    private var pollTask: Task<Void, Never>?
    private var isRunning = false
    private var lastDataVersion: Int?
    private var currentSessionID: String?
    private var currentLimit = 50

    init(databasePath: String) throws {
        self.database = BrainDatabase(path: databasePath)
    }

    func start(sessionID: String? = nil, limit: Int = 50) {
        currentSessionID = sessionID
        currentLimit = limit

        guard !isRunning else {
            refresh(force: true)
            return
        }

        isRunning = true
        installDarwinObserver()
        refresh(force: true)

        pollTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(250))
                guard !Task.isCancelled else { break }
                self.refresh(force: false)
            }
        }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil

        if isRunning {
            removeDarwinObserver()
        }

        isRunning = false
    }

    deinit {
        pollTask?.cancel()
        if isRunning {
            let center = CFNotificationCenterGetDarwinNotifyCenter()
            CFNotificationCenterRemoveObserver(
                center,
                Unmanaged.passUnretained(self).toOpaque(),
                CFNotificationName(BrainDatabase.dashboardDidChangeNotification as CFString),
                nil
            )
        }
        database.close()
    }

    func expandedConversation(chunkID: String, before: Int = 3, after: Int = 3) throws -> BrainDatabase.ExpandedConversation {
        try database.expandedConversation(id: chunkID, before: before, after: after)
    }

    fileprivate func handleDatabaseMutationNotification() {
        refresh(force: false)
    }

    private func refresh(force: Bool) {
        do {
            let currentDataVersion = try database.dataVersion()
            if force || currentDataVersion != lastDataVersion {
                events = try database.listInjectionEvents(
                    sessionID: currentSessionID,
                    limit: currentLimit
                )
                lastDataVersion = currentDataVersion
            }
        } catch {
            NSLog("[BrainBar] InjectionStore refresh failed: %@", String(describing: error))
        }
    }

    private func installDarwinObserver() {
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterAddObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque(),
            injectionStoreDarwinNotificationCallback,
            BrainDatabase.dashboardDidChangeNotification as CFString,
            nil,
            .deliverImmediately
        )
    }

    private func removeDarwinObserver() {
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterRemoveObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque(),
            CFNotificationName(BrainDatabase.dashboardDidChangeNotification as CFString),
            nil
        )
    }
}
