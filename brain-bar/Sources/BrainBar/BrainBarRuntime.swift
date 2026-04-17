import Foundation

@MainActor
final class BrainBarRuntime: ObservableObject {
    let launchMode: BrainBarLaunchMode
    let hotkeyStatus = HotkeyRouteStatus()
    let windowCoordinator: BrainBarWindowCoordinator

    @Published private(set) var collector: StatsCollector?
    @Published private(set) var injectionStore: InjectionStore?
    @Published private(set) var database: BrainDatabase?
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
        injectionStore: InjectionStore?,
        database: BrainDatabase
    ) {
        self.collector = collector
        self.injectionStore = injectionStore
        self.database = database
    }

    func handleToggleRequest() {
        if launchMode == .menuBarWindow, windowCoordinator.toggleVisibility() {
            return
        }
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
