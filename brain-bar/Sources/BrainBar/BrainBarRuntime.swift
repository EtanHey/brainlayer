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

    private var injectionStoreFactory: (() -> InjectionStore?)?

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
        database: BrainDatabase?,
        injectionStoreFactory: (() -> InjectionStore?)? = nil
    ) {
        self.collector = collector
        self.injectionStore = injectionStore
        self.database = database
        self.injectionStoreFactory = injectionStoreFactory
    }

    func ensureInjectionStore() {
        guard injectionStore == nil else { return }
        injectionStore = injectionStoreFactory?()
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
