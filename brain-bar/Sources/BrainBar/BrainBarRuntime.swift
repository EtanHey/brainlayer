import Foundation

@MainActor
final class BrainBarRuntime: ObservableObject {
    let launchMode: BrainBarLaunchMode
    let hotkeyStatus = HotkeyRouteStatus()
    let windowCoordinator: BrainBarWindowCoordinator

    @Published private(set) var collector: StatsCollector?
    @Published private(set) var database: BrainDatabase?
    private(set) var databasePath: String?

    var onToggleRequested: (() -> Void)?

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

}
