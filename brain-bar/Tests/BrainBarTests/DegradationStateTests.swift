import XCTest
@testable import BrainBar

@MainActor
final class KGDegradationStateTests: XCTestCase {
    var db: BrainDatabase!
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-kg-degraded-\(UUID().uuidString).db"
        db = BrainDatabase(path: tempDBPath)
    }

    override func tearDown() {
        db.close()
        for suffix in ["", "-wal", "-shm"] {
            try? FileManager.default.removeItem(atPath: tempDBPath + suffix)
        }
        super.tearDown()
    }

    // Regression guard: per Etan-mandate 2026-05-22 "WITHOUT DEGRATION!", a KG
    // read-path failure must surface a user-visible degradation state, not
    // silently blank the canvas. Before this guard, loadGraph()'s catch block
    // set nodes=[]/edges=[] and returned false with NO published flag — the UI
    // had no way to differentiate "empty graph" from "queries failing."
    func testLoadGraphFailureSetsDegradationState() async {
        db.close() // Force subsequent queries to throw — simulates fatal
                   // ReadOnly state.

        let vm = KGViewModel(database: db)
        XCTAssertEqual(vm.degradationState, .healthy, "Initial state is healthy")

        let success = await vm.loadGraph()

        XCTAssertFalse(success, "loadGraph should report failure on closed DB")
        XCTAssertTrue(
            vm.degradationState.isDegraded,
            "loadGraph failure MUST set degradationState to .degraded — see Etan-mandate 2026-05-22 (no silent blank states)."
        )
        XCTAssertNotNil(vm.degradationState.reason)
    }

    func testLoadGraphSuccessKeepsDegradationStateHealthy() async throws {
        // Match the seeding pattern of existing KGViewModelTests — a single
        // entity isn't surfaced by fetchKGEntities without participating in
        // at least one relation.
        try db.insertEntity(id: "a", type: "person", name: "Alice")
        try db.insertEntity(id: "b", type: "project", name: "BrainLayer")
        try db.insertRelation(sourceId: "a", targetId: "b", relationType: "builds")

        let vm = KGViewModel(database: db)
        let success = await vm.loadGraph()

        XCTAssertTrue(success)
        XCTAssertEqual(vm.degradationState, .healthy)
        XCTAssertEqual(vm.nodes.count, 2)
        XCTAssertEqual(vm.edges.count, 1)
    }
}

@MainActor
final class InjectionStoreDegradationStateTests: XCTestCase {
    var tempDBPath: String!

    override func setUp() {
        super.setUp()
        tempDBPath = NSTemporaryDirectory() + "brainbar-inj-degraded-\(UUID().uuidString).db"
    }

    override func tearDown() {
        for suffix in ["", "-wal", "-shm"] {
            try? FileManager.default.removeItem(atPath: tempDBPath + suffix)
        }
        super.tearDown()
    }

    // Regression guard: per Etan-mandate 2026-05-22 the injections panel must
    // not silently blank when InjectionStore.refresh hits a transient
    // ReadOnly / busy / locked error. Before this guard, refresh's catch only
    // NSLog'd the error and the UI had no signal that data was stale.
    func testInjectionStoreStartsHealthy() throws {
        let store = try InjectionStore(databasePath: tempDBPath)
        defer { store.stop() }

        XCTAssertEqual(store.degradationState, .healthy)
    }
}

@MainActor
final class DegradationStateTypeTests: XCTestCase {
    func testHealthyIsNotDegraded() {
        let state: DegradationState = .healthy
        XCTAssertFalse(state.isDegraded)
        XCTAssertNil(state.reason)
    }

    func testDegradedExposesReason() {
        let state: DegradationState = .degraded(reason: "ReadOnly")
        XCTAssertTrue(state.isDegraded)
        XCTAssertEqual(state.reason, "ReadOnly")
    }

    func testDegradationStateEquatable() {
        XCTAssertEqual(DegradationState.healthy, .healthy)
        XCTAssertEqual(DegradationState.degraded(reason: "x"), .degraded(reason: "x"))
        XCTAssertNotEqual(DegradationState.healthy, .degraded(reason: "x"))
        XCTAssertNotEqual(
            DegradationState.degraded(reason: "x"),
            DegradationState.degraded(reason: "y")
        )
    }
}
