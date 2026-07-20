import XCTest
@testable import BrainBar

final class BrainBarTruthFoundationTests: XCTestCase {
    deinit {}

    func testMetricContractsNameClockAndCardinality() {
        XCTAssertEqual(
            DashboardMetricContract.chunkRows,
            MetricContract(clock: .sourceTime, cardinality: .chunkRows)
        )
        XCTAssertEqual(
            DashboardMetricContract.agentOriginChunks,
            MetricContract(clock: .sourceTime, cardinality: .chunkRows)
        )
        XCTAssertEqual(
            DashboardMetricContract.watcherIngestedChunks,
            MetricContract(clock: .ingestTime, cardinality: .distinctChunkIDs)
        )
    }

    func testWatcherFlowStateUsesOnlyTypedLiveEvidence() {
        XCTAssertEqual(
            WatcherFlowState.derive(
                process: .running(pid: 42),
                recentDistinctChunkCount: 1,
                recentFlowReadable: true,
                pendingWorkCount: 0
            ),
            .flowing
        )
        XCTAssertEqual(
            WatcherFlowState.derive(
                process: .running(pid: 42),
                recentDistinctChunkCount: 0,
                recentFlowReadable: true,
                pendingWorkCount: 2
            ),
            .stalled
        )
        XCTAssertEqual(
            WatcherFlowState.derive(
                process: .running(pid: 42),
                recentDistinctChunkCount: 0,
                recentFlowReadable: true,
                pendingWorkCount: 0
            ),
            .runningNoRecentFlow
        )
        XCTAssertEqual(
            WatcherFlowState.derive(
                process: .absent,
                recentDistinctChunkCount: 10,
                recentFlowReadable: true,
                pendingWorkCount: 10
            ),
            .offline
        )
        XCTAssertEqual(
            WatcherFlowState.derive(
                process: .running(pid: 42),
                recentDistinctChunkCount: 0,
                recentFlowReadable: false,
                pendingWorkCount: 0
            ),
            .runningFlowUnverified
        )
        XCTAssertEqual(
            WatcherFlowState.derive(
                process: .failure("launchctl timed out"),
                recentDistinctChunkCount: 0,
                recentFlowReadable: false,
                pendingWorkCount: 0
            ),
            .unknown
        )
    }

    func testSnapshotFreshnessStrictlyCrossesAfterSixtySeconds() {
        let success = Date(timeIntervalSince1970: 1_000)

        XCTAssertEqual(
            SnapshotFreshnessState.derive(
                lastSuccessAt: nil,
                isRefreshing: true,
                lastFetchError: nil,
                now: success,
                snapshotFreshnessThreshold: 60
            ),
            .loading
        )
        XCTAssertEqual(
            SnapshotFreshnessState.derive(
                lastSuccessAt: success,
                isRefreshing: false,
                lastFetchError: nil,
                now: success.addingTimeInterval(60),
                snapshotFreshnessThreshold: 60
            ),
            .live(ageSeconds: 60)
        )
        XCTAssertEqual(
            SnapshotFreshnessState.derive(
                lastSuccessAt: success,
                isRefreshing: false,
                lastFetchError: nil,
                now: success.addingTimeInterval(60.001),
                snapshotFreshnessThreshold: 60
            ),
            .stale(ageSeconds: 60)
        )
        XCTAssertEqual(
            SnapshotFreshnessState.derive(
                lastSuccessAt: success,
                isRefreshing: false,
                lastFetchError: "read failed",
                now: success.addingTimeInterval(15),
                snapshotFreshnessThreshold: 60
            ),
            .error(message: "read failed", lastSuccessAgeSeconds: 15)
        )
    }

    func testLaunchctlWatcherProbeDistinguishesRunningAbsentAndFailure() {
        let running = LaunchctlWatcherProcessProbe(
            commandRunner: { _ in
                LaunchctlWatcherProcessProbe.CommandResult(
                    terminationStatus: 0,
                    output: "state = running\npid = 4242\n"
                )
            },
            uidProvider: { 501 }
        )
        XCTAssertEqual(running.sample(), .running(pid: 4242))

        let absent = LaunchctlWatcherProcessProbe(
            commandRunner: { _ in
                LaunchctlWatcherProcessProbe.CommandResult(
                    terminationStatus: 113,
                    output: "Could not find service com.brainlayer.watch"
                )
            },
            uidProvider: { 501 }
        )
        XCTAssertEqual(absent.sample(), .absent)

        let failed = LaunchctlWatcherProcessProbe(
            commandRunner: { _ in
                LaunchctlWatcherProcessProbe.CommandResult(
                    terminationStatus: 1,
                    output: "operation timed out"
                )
            },
            uidProvider: { 501 }
        )
        XCTAssertEqual(failed.sample(), .failure("operation timed out"))
    }
}
