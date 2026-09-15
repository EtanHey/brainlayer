import XCTest
@testable import BrainBar

@MainActor
final class BrainBarRuntimeWiringTests: XCTestCase {
    private var tempDBPath: String!

    override func setUp() async throws {
        try await super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-runtime-wiring-\(UUID().uuidString).db"
    }

    override func tearDown() async throws {
        for suffix in ["", "-wal", "-shm"] {
            try? FileManager.default.removeItem(atPath: tempDBPath + suffix)
        }
        tempDBPath = nil
        try await super.tearDown()
    }

    func testRuntimeStartsWithDatabaseAndCollectorNil() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        XCTAssertNil(runtime.database)
        XCTAssertNil(runtime.collector)
    }

    func testDevPreviewConfigurationRequiresExplicitBundleStamp() {
        XCTAssertTrue(
            BrainBarDevPreviewConfiguration.isPreviewProcess(
                infoDictionary: ["BrainBarDevPreview": true]
            )
        )
        XCTAssertNil(BrainBarDevPreviewConfiguration.resolve(infoDictionary: [:]))
        XCTAssertFalse(
            BrainBarDevPreviewConfiguration.hasSafePreviewIdentity(
                bundleIdentifier: "com.brainlayer.brainbar"
            )
        )
        XCTAssertTrue(
            BrainBarDevPreviewConfiguration.hasSafePreviewIdentity(
                bundleIdentifier: "com.brainlayer.brainbar.dev.wt-example-deadbeef"
            )
        )
        XCTAssertNil(
            BrainBarDevPreviewConfiguration.resolve(infoDictionary: [
                "BrainBarDevPreview": false,
                "BrainBarDevBranch": "wt/example",
                "GitCommit": "1234567890abcdef",
            ])
        )
    }

    func testDevPreviewConfigurationBuildsSelfIdentifyingTitle() throws {
        let configuration = try XCTUnwrap(
            BrainBarDevPreviewConfiguration.resolve(infoDictionary: [
                "BrainBarDevPreview": true,
                "BrainBarDevBranch": "wt/badge-state-contract",
                "GitCommit": "17e1ec370add5693ce2e1cb3b555947503c5a000",
                "GitDescribe": "v1.5.29-6-g17e1ec37",
            ])
        )

        XCTAssertEqual(configuration.branch, "wt/badge-state-contract")
        XCTAssertEqual(configuration.shortSHA, "17e1ec37")
        XCTAssertEqual(configuration.windowTitle, "DEV · wt/badge-state-contract · 17e1ec37")
    }

    func testDirtyDevPreviewTitleDoesNotClaimCleanCommitContents() throws {
        let configuration = try XCTUnwrap(
            BrainBarDevPreviewConfiguration.resolve(infoDictionary: [
                "BrainBarDevPreview": true,
                "BrainBarDevBranch": "wt/dev-previews",
                "GitCommit": "9441742ead8dc2964b43c702be8905926efc9e54",
                "GitDescribe": "v1.5.29-6-g9441742e-dirty",
            ])
        )

        XCTAssertEqual(configuration.windowTitle, "DEV · wt/dev-previews · 9441742e-dirty")
    }

    func testDuplicateInstancePolicyReplacesOnlyDevPreviews() {
        XCTAssertEqual(
            BrainBarDuplicateInstanceAction.resolve(
                isDevPreview: true,
                restartHandoffMatches: false
            ),
            .replaceExistingPreview
        )
        XCTAssertEqual(
            BrainBarDuplicateInstanceAction.resolve(
                isDevPreview: false,
                restartHandoffMatches: true
            ),
            .continueRestartHandoff
        )
        XCTAssertEqual(
            BrainBarDuplicateInstanceAction.resolve(
                isDevPreview: false,
                restartHandoffMatches: false
            ),
            .terminateNewInstance
        )
    }

    func testLaunchDecisionRefusesDevStampWithProductionBundleIdentifier() throws {
        let configuration = try XCTUnwrap(
            BrainBarDevPreviewConfiguration.resolve(infoDictionary: [
                "BrainBarDevPreview": true,
                "BrainBarDevBranch": "wt/example",
                "GitCommit": "1234567890abcdef",
            ])
        )

        XCTAssertEqual(
            BrainBarLaunchDecision.resolve(
                isDevPreview: true,
                previewConfiguration: configuration,
                bundleIdentifier: "com.brainlayer.brainbar"
            ),
            .refuse
        )
        XCTAssertEqual(
            BrainBarLaunchDecision.resolve(
                isDevPreview: true,
                previewConfiguration: configuration,
                bundleIdentifier: "com.brainlayer.brainbar.dev.wt-example-deadbeef"
            ),
            .replaceExistingPreview
        )
        XCTAssertEqual(
            BrainBarLaunchDecision.resolve(
                isDevPreview: false,
                previewConfiguration: nil,
                bundleIdentifier: "com.brainlayer.brainbar"
            ),
            .production
        )
    }

    func testPreviewReplacementFailsClosedWhenTerminateRequestIsRefused() {
        let replaced = BrainBarPreviewReplacement.replaceExisting(
            terminate: { false },
            isTerminated: { false },
            pumpRunLoop: { _ in XCTFail("must not wait after a refused terminate request") }
        )

        XCTAssertFalse(replaced)
    }

    func testPreviewReplacementFailsClosedWhenExistingProcessOutlivesTimeout() {
        var clock = Date(timeIntervalSince1970: 0)
        let replaced = BrainBarPreviewReplacement.replaceExisting(
            terminate: { true },
            isTerminated: { false },
            timeout: 0.05,
            now: { clock },
            pumpRunLoop: { date in clock = date }
        )

        XCTAssertFalse(replaced)
    }

    func testPreviewReplacementContinuesOnlyAfterExistingProcessTerminates() {
        var terminated = false
        let replaced = BrainBarPreviewReplacement.replaceExisting(
            terminate: { true },
            isTerminated: { terminated },
            pumpRunLoop: { _ in terminated = true }
        )

        XCTAssertTrue(replaced)
    }

    func testDevPreviewStampHardBlocksServerStart() {
        let socketPath = NSTemporaryDirectory() + "brainbar-dev-preview-\(UUID().uuidString).sock"
        let rejection = expectation(description: "server start rejected")
        let server = BrainBarServer(
            socketPath: socketPath,
            dbPath: tempDBPath,
            enableHybridSearchHelper: false,
            processInfoDictionary: ["BrainBarDevPreview": true]
        )
        server.onStartRejected = { reason in
            XCTAssertEqual(reason, "DEV preview processes are UI-only")
            rejection.fulfill()
        }

        server.start()

        wait(for: [rejection], timeout: 0.1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: socketPath))
    }

    func testWireRuntimePopulatesDatabaseWithoutOpeningAnExtraWriter() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let collector = BrainBarAppSupport.makeStatsCollector(
            dbPath: tempDBPath,
            targetPID: ProcessInfo.processInfo.processIdentifier,
            brainBusEvents: nil
        )
        defer { collector.stop() }

        BrainBarAppSupport.wireRuntime(runtime, dbPath: tempDBPath, collector: collector)
        XCTAssertNotNil(
            runtime.database,
            "Regression guard: BrainBarApp must not pass nil database to runtime.install — "
            + "the UI gates 'Warming memory…' / QuickCaptureViewModel on database != nil. "
            + "See PR #312 (FastAPI daemon removal) — UI process must open SQLite directly."
        )
        XCTAssertNotNil(runtime.collector)
    }

    func testWireRuntimeOpensReadonlyHandleEvenWhenDBFileMissing() {
        // Fresh-install scenario: brainlayer.db doesn't exist yet. wireRuntime
        // must bootstrap the file before installing the read-only BrainDatabase.
        // Otherwise the runtime carries a closed handle and search/graph stay
        // broken until app restart.
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: tempDBPath),
            "Test precondition: DB file must NOT exist at start"
        )

        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let collector = BrainBarAppSupport.makeStatsCollector(
            dbPath: tempDBPath,
            targetPID: ProcessInfo.processInfo.processIdentifier,
            brainBusEvents: nil
        )
        defer { collector.stop() }

        BrainBarAppSupport.wireRuntime(runtime, dbPath: tempDBPath, collector: collector)
        XCTAssertNotNil(runtime.database)
        XCTAssertTrue(
            runtime.database?.isOpen == true,
            "Regression guard for fresh installs: wireRuntime must create the DB "
            + "before installing the read-only BrainDatabase handle."
        )
    }

    func testDevPreviewRuntimeNeverCreatesMissingDatabase() {
        XCTAssertFalse(FileManager.default.fileExists(atPath: tempDBPath))
        let runtime = BrainBarRuntime()
        let collector = BrainBarAppSupport.makeUIStatsCollector(
            dbPath: tempDBPath,
            brainBusEvents: nil,
            daemonPIDProvider: { 0 }
        )
        defer { collector.stop() }

        BrainBarAppSupport.wireRuntime(
            runtime,
            dbPath: tempDBPath,
            collector: collector,
            bootstrapMissingDatabase: false
        )

        XCTAssertFalse(FileManager.default.fileExists(atPath: tempDBPath))
        XCTAssertFalse(runtime.database?.isOpen == true)
    }
}
