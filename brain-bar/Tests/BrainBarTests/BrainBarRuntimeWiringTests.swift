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

    func testRuntimeStartsWithDatabaseAndInjectionStoreNil() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        XCTAssertNil(runtime.database)
        XCTAssertNil(runtime.injectionStore)
        XCTAssertNil(runtime.collector)
    }

    func testWireRuntimePopulatesDatabaseAndLazilyLoadsInjectionStore() {
        let runtime = BrainBarRuntime(launchMode: .menuItemDaemon)
        let collector = BrainBarAppSupport.makeStatsCollector(
            dbPath: tempDBPath,
            targetPID: ProcessInfo.processInfo.processIdentifier,
            brainBusEvents: nil
        )
        defer { collector.stop() }

        BrainBarAppSupport.wireRuntime(runtime, dbPath: tempDBPath, collector: collector)
        defer { runtime.injectionStore?.stop() }

        XCTAssertNotNil(
            runtime.database,
            "Regression guard: BrainBarApp must not pass nil database to runtime.install — "
            + "the UI gates 'Warming memory…' / QuickCaptureViewModel on database != nil. "
            + "See PR #312 (FastAPI daemon removal) — UI process must open SQLite directly."
        )
        XCTAssertNil(
            runtime.injectionStore,
            "Dashboard startup should not eagerly open InjectionStore; it owns an extra writable SQLite handle."
        )
        runtime.ensureInjectionStore()
        XCTAssertNotNil(
            runtime.injectionStore,
            "Regression guard: BrainBarApp must provide a lazy InjectionStore factory so the Injections tab can load on demand."
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
        defer { runtime.injectionStore?.stop() }

        XCTAssertNotNil(runtime.database)
        XCTAssertTrue(
            runtime.database?.isOpen == true,
            "Regression guard for fresh installs: wireRuntime must create the DB "
            + "before installing the read-only BrainDatabase handle."
        )
    }
}
