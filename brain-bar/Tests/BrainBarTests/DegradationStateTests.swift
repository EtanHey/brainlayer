import XCTest
@testable import BrainBar

@MainActor
final class KGDegradationStateTests: XCTestCase {
    deinit {}

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

    func testLoadGraphUntilSuccessfulRecoversAfterInitialDegradation() async {
        // Regression guard: Cursor Bugbot PR #315 flagged that KGCanvasView
        // only performed one load. A first failed read must be retried so the
        // graph badge can clear after a later successful graph query.
        let reader = ScriptedKnowledgeGraphReader(results: [
            .failure(ScriptedKnowledgeGraphError.fetchFailed),
            .failure(ScriptedKnowledgeGraphError.fetchFailed),
            .success((
                entities: [
                    BrainDatabase.KGEntityRow(
                        id: "a",
                        name: "Alice",
                        entityType: "person",
                        description: nil,
                        importance: 7
                    ),
                    BrainDatabase.KGEntityRow(
                        id: "b",
                        name: "BrainLayer",
                        entityType: "project",
                        description: nil,
                        importance: 8
                    ),
                ],
                relations: [
                    BrainDatabase.KGRelationRow(
                        id: "r1",
                        sourceId: "a",
                        targetId: "b",
                        relationType: "builds"
                    ),
                ]
            )),
        ])
        let vm = KGViewModel(graphReader: reader)
        var retrySleeps = 0

        let success = await vm.loadGraphUntilSuccessful(retryDelay: .milliseconds(1)) { _ in
            retrySleeps += 1
        }

        XCTAssertTrue(success)
        XCTAssertEqual(vm.degradationState, .healthy)
        XCTAssertEqual(vm.nodes.count, 2)
        XCTAssertEqual(vm.edges.count, 1)
        XCTAssertEqual(retrySleeps, 1)
    }

    func testLoadGraphRepeatedlyMarksLaterFailureAfterInitialSuccess() async {
        // Regression guard: Cursor Bugbot PR #315 flagged that the graph tab
        // stopped checking the read path after the first successful load.
        let reader = ScriptedKnowledgeGraphReader(results: [
            .success((
                entities: [
                    BrainDatabase.KGEntityRow(
                        id: "a",
                        name: "Alice",
                        entityType: "person",
                        description: nil,
                        importance: 7
                    ),
                ],
                relations: []
            )),
            .failure(ScriptedKnowledgeGraphError.fetchFailed),
            .failure(ScriptedKnowledgeGraphError.fetchFailed),
        ])
        let vm = KGViewModel(graphReader: reader)
        var sleepCount = 0

        let loadedOnce = await vm.loadGraphRepeatedly(
            refreshDelay: .milliseconds(1),
            retryDelay: .milliseconds(1)
        ) { _ in
            sleepCount += 1
            if sleepCount == 2 {
                throw CancellationError()
            }
        }

        XCTAssertTrue(loadedOnce)
        XCTAssertTrue(vm.degradationState.isDegraded)
        XCTAssertEqual(vm.degradationState.reason, "fetchFailed")
    }

    func testLoadGraphMarksDegradedWhenRetrySleepIsCancelledAfterReadFailure() async {
        // Regression guard: Cursor Bugbot PR #315 flagged cancellation between
        // failed graph-read attempts as a path that hid the degraded badge.
        let reader = ScriptedKnowledgeGraphReader(results: [
            .failure(ScriptedKnowledgeGraphError.fetchFailed),
        ])
        let vm = KGViewModel(graphReader: reader)

        let success = await vm.loadGraph { _ in
            throw CancellationError()
        }

        XCTAssertFalse(success)
        XCTAssertTrue(vm.degradationState.isDegraded)
        XCTAssertEqual(vm.degradationState.reason, "fetchFailed")
    }
}

@MainActor
final class InjectionStoreDegradationStateTests: XCTestCase {
    deinit {}

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
    deinit {}

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

private enum ScriptedKnowledgeGraphError: Error {
    case fetchFailed
}

// ScriptedKnowledgeGraphReader is @unchecked Sendable because the mutable
// results and activeRelations buffers are test-confined and only accessed via
// fetchKGEntities/fetchKGRelations from @MainActor tests.
private final class ScriptedKnowledgeGraphReader: KnowledgeGraphReading, @unchecked Sendable {
    deinit {}

    typealias GraphResult = (
        entities: [BrainDatabase.KGEntityRow],
        relations: [BrainDatabase.KGRelationRow]
    )

    private var results: [Result<GraphResult, Error>]
    private var activeRelations: [BrainDatabase.KGRelationRow] = []

    init(results: [Result<GraphResult, Error>]) {
        self.results = results
    }

    func fetchKGEntities(limit: Int) throws -> [BrainDatabase.KGEntityRow] {
        guard !results.isEmpty else {
            throw ScriptedKnowledgeGraphError.fetchFailed
        }
        switch results.removeFirst() {
        case .success(let graph):
            activeRelations = graph.relations
            return graph.entities
        case .failure(let error):
            throw error
        }
    }

    func fetchKGRelations(limit: Int) throws -> [BrainDatabase.KGRelationRow] {
        activeRelations
    }
}
