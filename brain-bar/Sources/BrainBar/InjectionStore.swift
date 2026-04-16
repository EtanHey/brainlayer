import CoreFoundation
import Foundation

final class InjectionStoreObserverBox: @unchecked Sendable {
    weak var store: InjectionStore?
}

enum InjectionStoreDarwinObserver {
    static func scheduleRefresh(observerBox: InjectionStoreObserverBox) {
        Task { @MainActor [weak store = observerBox.store] in
            store?.handleDatabaseMutationNotification()
        }
    }
}

private func injectionStoreDarwinNotificationCallback(
    center: CFNotificationCenter?,
    observer: UnsafeMutableRawPointer?,
    name: CFNotificationName?,
    object: UnsafeRawPointer?,
    userInfo: CFDictionary?
) {
    guard let observer else { return }
    let observerBox = Unmanaged<InjectionStoreObserverBox>.fromOpaque(observer).takeUnretainedValue()
    InjectionStoreDarwinObserver.scheduleRefresh(observerBox: observerBox)
}

@MainActor
final class InjectionStore: ObservableObject {
    @Published private(set) var events: [InjectionEvent] = []

    private let database: BrainDatabase
    private let observerBox: InjectionStoreObserverBox
    private var pollTask: Task<Void, Never>?
    private var isRunning = false
    private var lastDataVersion: Int?
    private var currentSessionID: String?
    private var currentLimit = 50

    init(databasePath: String) throws {
        self.database = BrainDatabase(path: databasePath)
        self.observerBox = InjectionStoreObserverBox()
        guard database.isOpen else {
            throw BrainDatabase.DBError.notOpen
        }
        observerBox.store = self
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
            Self.removeDarwinObserver(observerBox: observerBox)
        }
        database.close()
    }

    var observerBoxForTesting: InjectionStoreObserverBox { observerBox }

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
        Self.addDarwinObserver(observerBox: observerBox)
    }

    private func removeDarwinObserver() {
        Self.removeDarwinObserver(observerBox: observerBox)
    }

    nonisolated private static func addDarwinObserver(observerBox: InjectionStoreObserverBox) {
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterAddObserver(
            center,
            Unmanaged.passUnretained(observerBox).toOpaque(),
            injectionStoreDarwinNotificationCallback,
            BrainDatabase.dashboardDidChangeNotification as CFString,
            nil,
            .deliverImmediately
        )
    }

    nonisolated private static func removeDarwinObserver(observerBox: InjectionStoreObserverBox) {
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterRemoveObserver(
            center,
            Unmanaged.passUnretained(observerBox).toOpaque(),
            CFNotificationName(BrainDatabase.dashboardDidChangeNotification as CFString),
            nil
        )
    }
}
