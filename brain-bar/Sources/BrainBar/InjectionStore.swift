import CoreFoundation
import Foundation

protocol InjectionEventReading: AnyObject {
    func dataVersion() throws -> Int
    func listInjectionEvents(sessionID: String?, limit: Int) throws -> [InjectionEvent]
    func expandedConversation(
        chunkID: String,
        before: Int,
        after: Int
    ) throws -> BrainDatabase.ExpandedConversation
    func close()
}

extension BrainDatabase: InjectionEventReading {
    func expandedConversation(
        chunkID: String,
        before: Int,
        after: Int
    ) throws -> BrainDatabase.ExpandedConversation {
        try expandedConversation(id: chunkID, before: before, after: after)
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
    let store = Unmanaged<InjectionStore>.fromOpaque(observer).takeUnretainedValue()
    Task { @MainActor in
        store.handleDatabaseMutationNotification()
    }
}

@MainActor
final class InjectionStore: ObservableObject {
    @Published private(set) var events: [InjectionEvent] = []
    @Published private(set) var degradationState: DegradationState = .healthy

    private enum RecoveryPhase: Equatable {
        case healthy
        case degraded(reason: String)
        case probing(reason: String)

        var reason: String? {
            switch self {
            case .healthy:
                return nil
            case .degraded(let reason), .probing(let reason):
                return reason
            }
        }
    }

    private let reader: InjectionEventReading
    private let pollInterval: Duration
    private let refreshDebounceInterval: TimeInterval
    private var pollTask: Task<Void, Never>?
    private var pendingRefreshTask: Task<Void, Never>?
    private var isRunning = false
    private var isActive = false
    private var observerInstalled = false
    private var lastDataVersion: Int?
    private var currentSessionID: String?
    private var currentLimit = 50
    private var recoveryPhase: RecoveryPhase = .healthy

    init(databasePath: String) throws {
        self.reader = BrainDatabase(
            path: databasePath,
            openConfiguration: BrainDatabase.OpenConfiguration(readOnly: true)
        )
        self.pollInterval = .milliseconds(750)
        self.refreshDebounceInterval = 0.15
    }

    init(
        reader: InjectionEventReading,
        pollInterval: Duration = .milliseconds(750),
        refreshDebounceInterval: TimeInterval = 0.15
    ) {
        self.reader = reader
        self.pollInterval = pollInterval
        self.refreshDebounceInterval = refreshDebounceInterval
    }

    func start(sessionID: String? = nil, limit: Int = 50, active: Bool = true) {
        let parametersChanged = currentSessionID != sessionID || currentLimit != limit
        currentSessionID = sessionID
        currentLimit = limit

        guard !isRunning else {
            setActive(active)
            if parametersChanged {
                scheduleRefresh(force: true, immediate: true)
            }
            return
        }

        isRunning = true
        setActive(active)
    }

    func stop() {
        pendingRefreshTask?.cancel()
        pendingRefreshTask = nil
        pollTask?.cancel()
        pollTask = nil

        if observerInstalled {
            removeDarwinObserver()
        }

        isActive = false
        isRunning = false
        reader.close()
    }

    func setActive(_ active: Bool) {
        guard isRunning else { return }
        guard isActive != active else { return }

        isActive = active
        if active {
            installDarwinObserver()
            scheduleRefresh(force: events.isEmpty, immediate: true)
            startPolling()
        } else {
            pendingRefreshTask?.cancel()
            pendingRefreshTask = nil
            pollTask?.cancel()
            pollTask = nil
            removeDarwinObserver()
        }
    }

    deinit {
        // The Darwin observer is registered with `Unmanaged.passUnretained(self)`.
        // If the store is released without an explicit `stop()` call (e.g. a
        // test fixture that skips teardown, or a mid-refactor owner swap),
        // the CF center will happily keep firing the callback on freed memory
        // and crash on the next dashboard mutation notification. Always
        // remove — CFNotificationCenterRemoveObserver is a no-op on a
        // never-registered observer, so this is safe regardless of state.
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterRemoveObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque(),
            CFNotificationName(BrainDatabase.dashboardDidChangeNotification as CFString),
            nil
        )
    }

    func expandedConversation(chunkID: String, before: Int = 3, after: Int = 3) throws -> BrainDatabase.ExpandedConversation {
        try reader.expandedConversation(chunkID: chunkID, before: before, after: after)
    }

    fileprivate func handleDatabaseMutationNotification() {
        scheduleRefresh(force: false)
    }

    func refreshForTesting(force: Bool) {
        refresh(force: force)
    }

    func scheduleRefreshForTesting(force: Bool) {
        scheduleRefresh(force: force)
    }

    private func startPolling() {
        guard pollTask == nil else { return }
        let interval = pollInterval
        pollTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                try? await Task.sleep(for: interval)
                guard !Task.isCancelled else { break }
                await MainActor.run {
                    guard self.isActive else { return }
                    self.scheduleRefresh(force: false, immediate: true)
                }
            }
        }
    }

    private func scheduleRefresh(force: Bool, immediate: Bool = false) {
        guard isActive else { return }
        pendingRefreshTask?.cancel()
        pendingRefreshTask = nil

        if immediate || refreshDebounceInterval <= 0 {
            refresh(force: force)
            return
        }

        let delay = refreshDebounceInterval
        pendingRefreshTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .seconds(delay))
            } catch {
                return
            }
            await MainActor.run {
                guard let self, self.isActive else { return }
                self.pendingRefreshTask = nil
                self.refresh(force: force)
            }
        }
    }

    private func refresh(force: Bool) {
        do {
            let currentDataVersion = try reader.dataVersion()
            let shouldProbeEvents: Bool
            switch recoveryPhase {
            case .healthy, .degraded:
                shouldProbeEvents = false
            case .probing:
                shouldProbeEvents = true
            }

            if force || currentDataVersion != lastDataVersion || shouldProbeEvents {
                events = try reader.listInjectionEvents(
                    sessionID: currentSessionID,
                    limit: currentLimit
                )
                lastDataVersion = currentDataVersion
                recoveryPhase = .healthy
                degradationState = .healthy
            } else if let reason = recoveryPhase.reason {
                // A clean data_version read is not enough to prove the
                // injections read path recovered. Keep the public badge
                // degraded and let the next poll probe listInjectionEvents
                // even if SQLite's data version is still unchanged.
                recoveryPhase = .probing(reason: reason)
                degradationState = .degraded(reason: reason)
            }
        } catch {
            NSLog("[BrainBar] InjectionStore refresh failed: %@", String(describing: error))
            let reason = String(describing: error)
            recoveryPhase = .degraded(reason: reason)
            degradationState = .degraded(reason: reason)
        }
    }

    private func installDarwinObserver() {
        guard !observerInstalled else { return }
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterAddObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque(),
            injectionStoreDarwinNotificationCallback,
            BrainDatabase.dashboardDidChangeNotification as CFString,
            nil,
            .deliverImmediately
        )
        observerInstalled = true
    }

    private func removeDarwinObserver() {
        guard observerInstalled else { return }
        let center = CFNotificationCenterGetDarwinNotifyCenter()
        CFNotificationCenterRemoveObserver(
            center,
            Unmanaged.passUnretained(self).toOpaque(),
            CFNotificationName(BrainDatabase.dashboardDidChangeNotification as CFString),
            nil
        )
        observerInstalled = false
    }
}
