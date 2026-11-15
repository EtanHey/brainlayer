import Foundation
import UserNotifications

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
        let settings = await center.notificationSettings()
        switch settings.authorizationStatus {
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

/// Main-actor serialization also covers updates arriving while the authorization sheet is open.
@MainActor
final class BrainBarFaultNotifications {
    private static let idsKey = "BrainBarNotificationFaultIDs.v1"
    private static let appIDsKey = "BrainBarNotificationAppFaultIDs.v1"
    private let center: any BrainBarFaultNotificationCenter
    private let defaults: UserDefaults
    private let log: (String) -> Void
    private var active: Set<String>
    private var appIDs: Set<String>
    private var pending: [String: String] = [:]
    private var delivering = false
    private var denied = false

    init(center: any BrainBarFaultNotificationCenter, defaults: UserDefaults,
         log: @escaping (String) -> Void = { NSLog("[BrainBar] %@", $0) }) {
        self.center = center
        self.defaults = defaults
        self.log = log
        active = Set((defaults.stringArray(forKey: Self.idsKey) ?? []).prefix(64))
        appIDs = Set((defaults.stringArray(forKey: Self.appIDsKey) ?? []).prefix(64))
    }

    func update(_ capture: BrainBarMenuCapture) async {
        let faults = BrainBarMenuPresenter.notificationFaults(capture)
        let fresh = capture.verdict != nil && (0...900).contains(capture.verdictAwakeAgeS)
        var appCapture = capture
        appCapture.verdict = nil
        let currentAppIDs = Set(BrainBarMenuPresenter.notificationFaults(appCapture).map(\.id))
        let ids = Set(faults.map(\.id))
        // An unreadable producer cannot prove recovery. App-owned facts can still resolve and rearm.
        let next = fresh ? ids : active.subtracting(appIDs).union(ids)
        pending = pending.filter { ids.contains($0.key) }
        for fault in faults where !active.contains(fault.id) { pending[fault.id] = fault.title }
        active = next
        appIDs = currentAppIDs
        defaults.set(active.sorted().prefix(64).map { $0 }, forKey: Self.idsKey)
        defaults.set(appIDs.sorted().prefix(64).map { $0 }, forKey: Self.appIDsKey)
        guard !pending.isEmpty, !delivering, !denied else { return }
        delivering = true
        defer { delivering = false }
        do {
            guard try await center.authorize() else {
                denied = true
                pending.removeAll()
                if !defaults.bool(forKey: "BrainBarNotificationDenialLogged") {
                    log("Notifications are not authorized; fault notices remain in the menu.")
                    defaults.set(true, forKey: "BrainBarNotificationDenialLogged")
                }
                return
            }
            while let id = pending.keys.sorted().first, let body = pending.removeValue(forKey: id) {
                try await center.post(id: id, body: body)
            }
        } catch {
            pending.removeAll()
            log("Fault notification delivery failed: \(type(of: error))")
        }
    }
}
