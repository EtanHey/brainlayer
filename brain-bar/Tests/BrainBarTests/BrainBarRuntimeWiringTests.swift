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
        let runtime = BrainBarRuntime(launchMode: .menuBarWindow)
        XCTAssertNil(runtime.database)
        XCTAssertNil(runtime.injectionStore)
        XCTAssertNil(runtime.collector)
    }

    func testWireRuntimePopulatesDatabaseAndInjectionStore() {
        let runtime = BrainBarRuntime(launchMode: .menuBarWindow)
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
        XCTAssertNotNil(
            runtime.injectionStore,
            "Regression guard: BrainBarApp must wire InjectionStore — "
            + "the Injections tab placeholder shows when injectionStore == nil."
        )
        XCTAssertNotNil(runtime.collector)
    }

    func testWireRuntimeOpensReadonlyHandleEvenWhenDBFileMissing() {
        // Fresh-install scenario: brainlayer.db doesn't exist yet. InjectionStore
        // must create the file first; the read-only BrainDatabase must then open
        // successfully. Otherwise the runtime installs a permanently-closed handle
        // that satisfies `database != nil` but breaks search/graph downstream.
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: tempDBPath),
            "Test precondition: DB file must NOT exist at start"
        )

        let runtime = BrainBarRuntime(launchMode: .menuBarWindow)
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
            "Regression guard for Cursor Bugbot + ChatGPT-Codex flag (PR #314 review): "
            + "on fresh install, wireRuntime must order InjectionStore (creates file) "
            + "BEFORE the read-only BrainDatabase open — otherwise the runtime carries "
            + "a closed handle and search/graph stay broken until app restart."
        )
    }
}
