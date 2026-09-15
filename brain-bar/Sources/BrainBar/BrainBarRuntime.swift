import Foundation

struct BrainBarDevPreviewConfiguration: Equatable {
    let branch: String
    let commitShortSHA: String
    let isDirty: Bool

    var shortSHA: String {
        commitShortSHA + (isDirty ? "-dirty" : "")
    }

    var windowTitle: String {
        "DEV · \(branch) · \(shortSHA)"
    }

    static func isPreviewProcess(infoDictionary: [String: Any]? = Bundle.main.infoDictionary) -> Bool {
        infoDictionary?["BrainBarDevPreview"] as? Bool == true
    }

    static func hasSafePreviewIdentity(bundleIdentifier: String?) -> Bool {
        bundleIdentifier?.hasPrefix("com.brainlayer.brainbar.dev.") == true
    }

    static func resolve(infoDictionary: [String: Any]? = Bundle.main.infoDictionary) -> Self? {
        guard isPreviewProcess(infoDictionary: infoDictionary),
              let branch = infoDictionary?["BrainBarDevBranch"] as? String,
              !branch.isEmpty,
              let commit = infoDictionary?["GitCommit"] as? String,
              !commit.isEmpty
        else {
            return nil
        }
        let describe = infoDictionary?["GitDescribe"] as? String ?? ""
        return Self(
            branch: branch,
            commitShortSHA: String(commit.prefix(8)),
            isDirty: describe.hasSuffix("-dirty")
        )
    }
}

enum BrainBarDuplicateInstanceAction: Equatable {
    case replaceExistingPreview
    case continueRestartHandoff
    case terminateNewInstance

    static func resolve(isDevPreview: Bool, restartHandoffMatches: Bool) -> Self {
        if isDevPreview {
            return .replaceExistingPreview
        }
        return restartHandoffMatches ? .continueRestartHandoff : .terminateNewInstance
    }
}

enum BrainBarPreviewReplacement {
    static func replaceExisting(
        terminate: () -> Bool,
        isTerminated: () -> Bool,
        timeout: TimeInterval = 2,
        now: () -> Date = Date.init,
        pumpRunLoop: (Date) -> Void
    ) -> Bool {
        guard terminate() || isTerminated() else { return false }
        let deadline = now().addingTimeInterval(timeout)
        while !isTerminated() && now() < deadline {
            pumpRunLoop(now().addingTimeInterval(0.02))
        }
        return isTerminated()
    }
}

@MainActor
final class BrainBarRuntime: ObservableObject {
    let launchMode: BrainBarLaunchMode
    let hotkeyStatus = HotkeyRouteStatus()
    let windowCoordinator: BrainBarWindowCoordinator

    @Published private(set) var collector: StatsCollector?
    @Published private(set) var database: BrainDatabase?
    private(set) var databasePath: String?
    @Published private(set) var requestedQuickAction: BrainBarQuickAction?

    var onToggleRequested: (() -> Void)?
    var onSearchRequested: (() -> Void)?
    var onQuickCaptureRequested: (() -> Void)?

    init(
        launchMode: BrainBarLaunchMode = BrainBarLaunchMode.resolve(),
        windowCoordinator: BrainBarWindowCoordinator = BrainBarWindowCoordinator()
    ) {
        self.launchMode = launchMode
        self.windowCoordinator = windowCoordinator
    }

    func install(
        collector: StatsCollector,
        database: BrainDatabase?,
        databasePath: String? = nil
    ) {
        self.collector = collector
        self.database = database
        self.databasePath = databasePath
    }

    func handleToggleRequest() {
        onToggleRequested?()
    }

    func showSearchPanel() {
        onSearchRequested?()
    }

    func showQuickCapturePanel() {
        onQuickCaptureRequested?()
    }

    func presentQuickAction(_ action: BrainBarQuickAction) {
        requestedQuickAction = action
    }

    func clearQuickActionRequest() {
        requestedQuickAction = nil
    }
}
