import Foundation

@MainActor
final class BrainBarRuntime: ObservableObject {
    let launchMode: BrainBarLaunchMode
    let hotkeyStatus = HotkeyRouteStatus()
    let windowCoordinator: BrainBarWindowCoordinator
    /// Google Drive access for backups. It has no runner until AppDelegate installs the live one,
    /// so tests and previews never run the CLI.
    let driveAuth = BrainBarDriveAuthModel(runner: nil)
    let maintenanceRetry = BrainBarMaintenanceRetryModel(runner: nil)

    @Published private(set) var collector: StatsCollector?
    @Published private(set) var database: BrainDatabase?
    private(set) var databasePath: String?

    #if BRAINBAR_UI
    /// The app's one status model; it reads the Drive auth state at each publication.
    @Published var jobsModel: BrainBarSettingsViewModel? {
        didSet {
            jobsModel?.attachDriveAuth(driveAuth)
            if let collector { jobsModel?.attachVectorCoverage(collector) }
        }
    }
    #endif

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
        #if BRAINBAR_UI
        jobsModel?.attachVectorCoverage(collector)
        #endif
        self.database = database
        self.databasePath = databasePath
    }

    func handleToggleRequest() {
        onToggleRequested?()
    }

}
