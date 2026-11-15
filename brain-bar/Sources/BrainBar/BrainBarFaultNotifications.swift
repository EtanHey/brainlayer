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
    private let center: any BrainBarFaultNotificationCenter
    private let defaults: UserDefaults
    private let log: (String) -> Void
    private var delivered: [String: Set<String>]
    private var current: [String: BrainBarMenuPresenter.NotificationFault] = [:]
    private var generations: [String: Int] = [:]
    private var postingID: String?
    private var delivering = false
    private var updates: AnyCancellable?
    private var observationID = UUID()

    init(center: any BrainBarFaultNotificationCenter, defaults: UserDefaults,
         log: @escaping (String) -> Void = { NSLog("[BrainBar] %@", $0) }) {
        self.center = center
        self.defaults = defaults
        self.log = log
        let saved: [String: [String]] = defaults.dictionary(forKey: Self.deliveredKey) as? [String: [String]] ?? [:]
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
        defaults.set(saved, forKey: Self.deliveredKey)
    }

    func update(_ capture: BrainBarMenuCapture, jobAlerts: [String: String]? = nil) async {
        let faults: [BrainBarMenuPresenter.NotificationFault] = BrainBarMenuPresenter.notificationFaults(capture)
        var next: [String: BrainBarMenuPresenter.NotificationFault] = Dictionary(uniqueKeysWithValues: faults.map { ($0.id, $0) })
        if let jobAlerts {
            for (key, message) in jobAlerts {
                let id: String = "job:" + key
                next[id] = .init(id: id, title: message, causes: ["job:" + key])
            }
        }
        let complete: Bool = BrainBarMenuPresenter.notificationCaptureIsComplete(capture)
        var knownIDs: Set<String> = Set(delivered.keys).union(current.keys)
        if let postingID { knownIDs.insert(postingID) }
        for id in knownIDs {
            let measured: Bool = id.hasPrefix("job:") ? jobAlerts != nil : complete
            guard measured else { continue }
            if let fault: BrainBarMenuPresenter.NotificationFault = next[id] {
                // Losing a positively absent cause never posts; its later return is a new worsening.
                if let previous: Set<String> = delivered[id] { delivered[id] = previous.intersection(fault.causes) }
            } else {
                delivered.removeValue(forKey: id)
                generations[id, default: 0] += 1
            }
        }
        current = next
        persist()
        guard !delivering, needsDelivery else { return }
        delivering = true
        defer { delivering = false }
        do {
            let authorized: Bool = try await center.authorize()
            let state: String = authorized ? "authorized" : "denied"
            if defaults.string(forKey: "BrainBarNotificationAuthorizationState") != state {
                if !authorized { log("Notifications are not authorized; fault notices remain in the menu.") }
                defaults.set(state, forKey: "BrainBarNotificationAuthorizationState")
            }
            guard authorized else { return }
            var failed: Set<String> = []
            while let id: String = current.keys.sorted().first(where: {
                !failed.contains($0) && !(current[$0]!.causes.subtracting(delivered[$0] ?? [])).isEmpty
            }), let fault: BrainBarMenuPresenter.NotificationFault = current[id] {
                let generation: Int = generations[id, default: 0]
                postingID = id
                defer { postingID = nil }
                do {
                    try await center.post(id: id, body: fault.title)
                    // A recovery while add was in flight must not suppress the next episode.
                    if generations[id, default: 0] == generation {
                        delivered[id, default: []].formUnion(fault.causes)
                        persist()
                    }
                } catch {
                    failed.insert(id)
                    log("Fault notification delivery failed: \(type(of: error))")
                }
            }
        } catch {
            log("Fault notification authorization failed: \(type(of: error))")
        }
    }

    private var needsDelivery: Bool {
        current.values.contains { !$0.causes.subtracting(delivered[$0.id] ?? []).isEmpty }
    }
}
