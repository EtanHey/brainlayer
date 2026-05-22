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
}
