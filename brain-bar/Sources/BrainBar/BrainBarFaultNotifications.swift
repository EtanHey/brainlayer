import Combine
import Foundation
@preconcurrency import UserNotifications

@MainActor
protocol BrainBarFaultNotificationCenter: AnyObject {
    func authorize() async throws -> Bool
    func post(id: String, body: String) async throws
}

/// Only the app launch path constructs this adapter. Tests inject their own center.
@MainActor
final class BrainBarSystemNotificationCenter: NSObject, BrainBarFaultNotificationCenter, UNUserNotificationCenterDelegate {
    private let center: UNUserNotificationCenter
    private let onClick: @MainActor () -> Void

    init(onClick: @escaping @MainActor () -> Void) {
        center = UNUserNotificationCenter.current()
        self.onClick = onClick
        super.init()
        center.delegate = self
    }

    func authorize() async throws -> Bool {
        let status: UNAuthorizationStatus = await withCheckedContinuation { continuation in
            center.getNotificationSettings { settings in
                continuation.resume(returning: settings.authorizationStatus)
            }
        }
        switch status {
        case .notDetermined: return try await center.requestAuthorization(options: [.alert])
        case .authorized, .provisional: return true
        case .denied: return false
        @unknown default: return false
        }
    }

    func post(id: String, body: String) async throws {
        let content = UNMutableNotificationContent()
        content.title = "BrainLayer"
        content.body = body
        // macOS supplies the posting app's bundle icon. No sound is requested.
        try await center.add(UNNotificationRequest(identifier: id, content: content, trigger: nil))
    }

    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse, withCompletionHandler completionHandler: @escaping () -> Void) {
        Task { @MainActor [weak self] in self?.onClick() }
        completionHandler()
    }

    nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter,
        willPresent notification: UNNotification, withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        completionHandler([.banner, .list])
    }
}

/// A cancelled menu is an ordinary dismissal, never a request to open Settings.
@MainActor
enum BrainBarNotificationClick {
    static func perform(hasButton: Bool, popup: () -> Bool, settings: () -> Void) {
        if hasButton { _ = popup() } else { settings() }
    }
}

@MainActor
final class BrainBarFaultNotifications {
    struct Input: Sendable {
        var capture: BrainBarMenuCapture
        /// nil is unreadable/not sampled; an empty map positively confirms no job alerts.
        var jobAlerts: [String: String]? = nil
    }

    private static let deliveredKey = "BrainBarNotificationDeliveredCauses.v2"
    private static let recentKey = "BrainBarNotificationRecentCauses.v1"
    private let center: any BrainBarFaultNotificationCenter
    private let defaults: UserDefaults
    private let log: (String) -> Void
    private let clock: () -> Date
    private var now = Date()
    private var recent: [String: [String: Double]]
    private var savedDelivered: [String: [String]]
    private var savedRecent: [String: [String: Double]]
    private var errorLoggedAt: [String: Date] = [:]
    private var delivered: [String: Set<String>]
    private var current: [String: BrainBarMenuPresenter.NotificationFault] = [:]
    private var generations: [String: Int] = [:]
    private var postingID: String?
    private var delivering = false
    private var updates: AnyCancellable?
    private var observationID = UUID()

    init(center: any BrainBarFaultNotificationCenter, defaults: UserDefaults,
         clock: @escaping () -> Date = Date.init,
         log: @escaping (String) -> Void = { NSLog("[BrainBar] %@", $0) }) {
        self.center = center
        self.defaults = defaults
        self.log = log
        self.clock = clock
        let saved: [String: [String]] = defaults.dictionary(forKey: Self.deliveredKey) as? [String: [String]] ?? [:]
        savedDelivered = saved
        let timestamps = defaults.dictionary(forKey: Self.recentKey) as? [String: [String: Double]] ?? [:]
        savedRecent = timestamps
        recent = Dictionary(uniqueKeysWithValues: timestamps.keys.sorted().prefix(64).map { id in
            (id, Dictionary(uniqueKeysWithValues: (timestamps[id] ?? [:]).keys.sorted().prefix(64).map {
                ($0, timestamps[id]![$0]!)
            }))
        })
        delivered = Dictionary(uniqueKeysWithValues: saved.keys.sorted().prefix(64).map {
            ($0, Set((saved[$0] ?? []).prefix(64)))
        })
    }

    /// Owns the production notification subscription independently of menu rendering/preferences.
    /// All 26 model properties may publish in one sample; capture/derive runs at most once per second.
    func observe(changes: AnyPublisher<Void, Never>, snapshot: @escaping @MainActor () -> Input?) {
        observationID = UUID()
        let id: UUID = observationID
        updates = changes.prepend(())
            .receive(on: RunLoop.main)
            .throttle(for: RunLoop.SchedulerTimeType.Stride.seconds(1), scheduler: RunLoop.main, latest: true)
            .sink { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, self.observationID == id, let input: Input = snapshot() else { return }
                    await self.update(input.capture, jobAlerts: input.jobAlerts)
                }
            }
    }

    func stopObserving() {
        updates = nil
        observationID = UUID()
    }

    private func persist() {
        let saved: [String: [String]] = Dictionary(uniqueKeysWithValues: delivered.keys.sorted().prefix(64).map {
            ($0, Array((delivered[$0] ?? []).sorted().prefix(64)))
        })
        let times = Dictionary(uniqueKeysWithValues: recent.keys.sorted().prefix(64).map { id in
            (id, Dictionary(uniqueKeysWithValues: recent[id]!.keys.sorted().prefix(64).map { ($0, recent[id]![$0]!) }))
        })
        if saved != savedDelivered {
            defaults.set(saved, forKey: Self.deliveredKey)
            savedDelivered = saved
        }
        if times != savedRecent {
            defaults.set(times, forKey: Self.recentKey)
            savedRecent = times
        }
    }

    private static func pair(_ jobID: String) -> (thing: String, cause: String)? {
        switch jobID {
        case "job:drive-consent": return ("safe", "check:drive_auth")
        case "job:backup-daily": return ("safe", "check:backup_local")
        case "job:jsonl-backup": return ("safe", "check:transcripts")
        case "job:maintenance-light", "job:maintenance-full", "job:maintenance-burn":
            return ("upkeep", "app:maintenance")
        default: return nil
        }
    }

    private func record(_ id: String, _ causes: Set<String>, at timestamp: Double? = nil) {
        let added = causes.subtracting(delivered[id] ?? [])
        guard !added.isEmpty else { return }
        delivered[id, default: []].formUnion(added)
        for cause in added { recent[id, default: [:]][cause] = timestamp ?? now.timeIntervalSince1970 }
    }

    /// Alias receipts refer to a successful banner for this same active incident, never a pending add.
    private func acknowledgeAliases() {
        let jobs = current.values.filter { $0.id.hasPrefix("job:") }.sorted { $0.id < $1.id }
        for job in jobs {
            if let posted = jobs.first(where: {
                $0.id != job.id && $0.title == job.title && $0.causes.isSubset(of: delivered[$0.id] ?? [])
            }) {
                record(job.id, job.causes, at: recent[posted.id]?.values.min())
            } else if let pair = Self.pair(job.id), let thing = current[pair.thing],
                      thing.pairingCauses.contains(pair.cause),
                      thing.causes.isSubset(of: delivered[thing.id] ?? []) {
                record(job.id, job.causes, at: recent[thing.id]?[pair.cause] ?? recent[thing.id]?["attention"])
            }
            if let pair = Self.pair(job.id), current[pair.thing]?.pairingCauses.contains(pair.cause) == true,
               job.causes.isSubset(of: delivered[job.id] ?? []) {
                record(pair.thing, ["attention", pair.cause], at: recent[job.id]?.values.min())
            }
        }
    }

    private func pendingCauses(_ fault: BrainBarMenuPresenter.NotificationFault) -> Set<String> {
        if let pair = Self.pair(fault.id), current[pair.thing]?.pairingCauses.contains(pair.cause) == true { return [] }
        let cooling = Set((recent[fault.id] ?? [:]).keys)
        return fault.causes.subtracting(delivered[fault.id] ?? []).subtracting(cooling)
    }

    private func logFailure(_ kind: String, _ error: Error) {
        if let last = errorLoggedAt[kind], now.timeIntervalSince(last) < 60 { return }
        errorLoggedAt[kind] = now
        log("Fault notification \(kind) failed: \(type(of: error))")
    }

    func update(_ capture: BrainBarMenuCapture, jobAlerts: [String: String]? = nil) async {
        now = clock()
        recent = recent.compactMapValues { times in
            let live = times.filter { now.timeIntervalSince1970 - $0.value >= 0 && now.timeIntervalSince1970 - $0.value < 600 }
            return live.isEmpty ? nil : live
        }
        let faults = BrainBarMenuPresenter.notificationFaults(capture)
        var next = Dictionary(uniqueKeysWithValues: faults.map { ($0.id, $0) })
        if let jobAlerts {
            for (key, message) in jobAlerts {
                guard key != "brainbar-activation",
                      !(key == "jsonl-retention" && message.hasPrefix("Drive prune interrupted")) else { continue }
                let id = "job:" + key
                // R31: a paired job cannot promote its amber check back into a fault banner.
                if let pair = Self.pair(id), capture.verdict?.checks.contains(where: {
                    $0.state.uiKind == .notice && pair.cause == BrainBarMenuPresenter.checkNotificationCause($0.name)
                }) == true, !faults.contains(where: { $0.pairingCauses.contains(pair.cause) }) { continue }
                let cause = key == "drive-consent" ? "message:" + message : id
                next[id] = .init(id: id, title: message, causes: [cause])
            }
        }
        var knownIDs = Set(delivered.keys).union(current.keys)
        if let postingID { knownIDs.insert(postingID) }
        for id in knownIDs {
            let measured = id.hasPrefix("job:") ? jobAlerts != nil : BrainBarMenuPresenter.notificationThingIsMeasured(id, capture)
            guard measured else { continue }
            if let fault = next[id] {
                if let previous = delivered[id] { delivered[id] = previous.intersection(fault.causes) }
            } else {
                delivered.removeValue(forKey: id)
                generations[id, default: 0] += 1
            }
        }
        current = next
        acknowledgeAliases()
        persist()
        guard !delivering, current.values.contains(where: { !pendingCauses($0).isEmpty }) else { return }
        delivering = true
        defer { delivering = false }
        do {
            let authorized = try await center.authorize()
            let state = authorized ? "authorized" : "denied"
            if defaults.string(forKey: "BrainBarNotificationAuthorizationState") != state {
                if !authorized { log("Notifications are not authorized; fault notices remain in the menu.") }
                defaults.set(state, forKey: "BrainBarNotificationAuthorizationState")
            }
            guard authorized else { return }
            var failed: Set<String> = []
            while true {
                acknowledgeAliases()
                persist()
                guard let id = current.keys.sorted().first(where: {
                    !failed.contains($0) && !pendingCauses(current[$0]!).isEmpty
                }), let fault = current[id] else { break }
                let newCauses = pendingCauses(fault)
                let generation = generations[id, default: 0]
                postingID = id
                do {
                    try await center.post(id: id, body: fault.title(for: newCauses))
                    // A recovery while add was in flight must not suppress the next episode.
                    if generations[id, default: 0] == generation {
                        record(id, fault.causes, at: clock().timeIntervalSince1970)
                        persist()
                    }
                } catch {
                    failed.insert(id)
                    logFailure("delivery", error)
                }
                postingID = nil
            }
        } catch {
            logFailure("authorization", error)
        }
    }
}

extension BrainBarMenuPresenter {
    static func appNotificationCause(_ reason: String) -> String { "app:" + reason }
    static func checkNotificationCause(_ name: String) -> String { "check:" + name }
}
