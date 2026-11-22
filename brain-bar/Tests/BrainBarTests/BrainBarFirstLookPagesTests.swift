import SwiftUI
import XCTest
@testable import BrainBar

@MainActor final class BrainBarFirstLookPagesTests: XCTestCase {
    private final class DesignDefaults: UserDefaults, @unchecked Sendable {
        var choice: String?
        init(choice: String? = nil) {
            self.choice = choice
            super.init(suiteName: "BrainBarFirstLookPages." + UUID().uuidString)!
        }
        override func string(forKey key: String) -> String? { key == BrainBarDesign.key ? choice : nil }
    }

    func testNewIsDefaultAndStoredClassicStillWins() {
        let defaults = DesignDefaults()
        XCTAssertEqual(BrainBarDesign.current(in: defaults), .new)
        XCTAssertNil(defaults.choice, "reading does not store a preference")
        defaults.choice = "classic"
        XCTAssertEqual(BrainBarDesign.current(in: defaults), .classic)
    }

    func testActivityScaleLabelsUseTheirOwnPeaks() {
        let wave = NDWave(saves: [1, 7, 2], reads: [31, 4, 3])
        let labels = NDActivitySection.scaleLabels(wave, minutes: 3)
        XCTAssertEqual(labels.saves, "up to 7 / min")
        XCTAssertEqual(labels.reads, "up to 31 / min")
        XCTAssertNotEqual(labels.saves, labels.reads)
        var sliding = wave
        sliding.outgoing = .init(saves: 17, reads: 43)
        XCTAssertEqual(NDActivitySection.scaleLabels(sliding, minutes: 3).saves, "up to 17 / min")
        XCTAssertEqual(NDActivitySection.scaleLabels(sliding, minutes: 3).reads, "up to 43 / min")
        let wide = NDWave(saves: Array(repeating: 7, count: 60), reads: Array(repeating: 31, count: 60))
        XCTAssertEqual(NDActivitySection.scaleLabels(wide, minutes: 180).saves, "up to 7 / 3 min")
        XCTAssertEqual(NDActivitySection.scaleLabels(wide, minutes: 1440).reads, "up to 31 / 24 min")
    }

    private func find<T>(_ type: T.Type, in value: Any) -> T? {
        if let result = value as? T { return result }
        for child in Mirror(reflecting: value).children {
            if let result = find(type, in: child.value) { return result }
        }
        return nil
    }

    func testRangeSelectionFetchesWithinThrottleAndAppliesUnderPointer() async throws {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        let now = store.inputs.capture.now
        store.waveNow = { now }
        store.fetchWaveBuckets = { minutes, count in
            .init(activityWindowMinutes: minutes, bucketCount: count, allWriteBuckets: [minutes],
                  agentWriteBuckets: [minutes], watcherWriteBuckets: [count], enrichmentBuckets: [],
                  watcherFlowReadability: .readable)
        }
        store.setWindowActive(true)
        await store.waitForWaveFetchForTesting()
        XCTAssertEqual(store.inputs.heroWaves[.hour]?.saves, [60])
        store.holdChart(true)
        store.range = .day
        await store.waitForWaveFetchForTesting()
        XCTAssertEqual(store.inputs.heroWaves[.day]?.saves, [1440])
        store.range = .threeHours
        await store.waitForWaveFetchForTesting()
        XCTAssertEqual(store.inputs.heroWaves[.threeHours]?.saves, [180])
        store.waveNow = { now.addingTimeInterval(16) }
        store.fetchWaveBuckets = { minutes, count in
            .init(activityWindowMinutes: minutes, bucketCount: count, allWriteBuckets: [999],
                  agentWriteBuckets: [999], watcherWriteBuckets: [999], enrichmentBuckets: [],
                  watcherFlowReadability: .readable)
        }
        store.refreshWaves()
        await store.waitForWaveFetchForTesting()
        XCTAssertEqual(store.inputs.menuWave?.saves, [999])
        XCTAssertEqual(store.inputs.heroWaves[.threeHours]?.saves, [180], "periodic updates still wait under the pointer")
        store.holdChart(false)
        XCTAssertEqual(store.inputs.heroWaves[.threeHours]?.saves, [999])
        XCTAssertEqual(store.inputs.heroWaves[.day]?.saves, [1440], "releasing the pointer must not restore a held old range")
        store.setWindowActive(false)
    }

    func testUndatedAndEventConversationRowsPreserveGoodDots() throws {
        let now = BrainBarDesignFixture.inputs(.calm).capture.now
        let at = ISO8601DateFormatter().string(from: now.addingTimeInterval(-3600))
        let lines = ["{\"uploaded\":true}", "{\"event\":\"scan\",\"attempted_at\":\"\(at)\"}",
                     "{\"attempted_at\":\"\(at)\",\"uploaded\":true,\"verified\":true}"]
        let parsed = try XCTUnwrap(NDBackupHistoryReader.parseConversations(.init(lines: lines.map { Substring($0) }, unreadable: false), now: now))
        XCTAssertEqual(parsed.attempts?.map(\.ok), [true])
        XCTAssertEqual(parsed.skipped, 0)
    }

    func testUndatedAndEventDatabaseRowsPreserveGoodDots() throws {
        let now = BrainBarDesignFixture.inputs(.calm).capture.now
        let at = ISO8601DateFormatter().string(from: now.addingTimeInterval(-3600))
        let lines = ["{\"backup_log_provenance\":\"real\"}",
                     "{\"event\":\"scan\",\"attempted_at\":\"\(at)\",\"backup_log_provenance\":\"real\"}",
                     "{\"attempted_at\":\"\(at)\",\"backup_log_provenance\":\"real\",\"uploaded\":true,\"verified\":true}"]
        let parsed = try XCTUnwrap(NDBackupHistoryReader.parseDatabase(.init(lines: lines.map { Substring($0) }, unreadable: false), now: now))
        XCTAssertEqual(parsed.drive?.map(\.ok), [true])
        XCTAssertEqual(parsed.skipped, 0)
    }

    func testDamagedConversationRowHasUnknownLegend() throws {
        var inputs = BrainBarDesignFixture.inputs(.calm)
        let at = ISO8601DateFormatter().string(from: inputs.capture.now)
        let parsed = try XCTUnwrap(NDBackupHistoryReader.parseConversations(.init(lines: [Substring("{\"attempted_at\":\"\(at)\",\"uploaded\":\"bad\"}")], unreadable: false), now: inputs.capture.now))
        XCTAssertNil(parsed.attempts)
        inputs.backupHistory = .read(.init(local: [], drive: [], conversations: parsed.attempts, integrity: []))
        let store = NDStore(fixture: inputs)
        XCTAssertTrue(NDBackupsPage.legend(NDBackupsPage(store: store).lanes(store.model)).contains("Record couldn't be read"))
    }

    func testNextRunRingsNameTheirClockTime() {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        let lanes = NDBackupsPage(store: store).lanes(store.model)
        let next = lanes.flatMap(\.events).filter { $0.kind == .next }
        XCTAssertFalse(next.isEmpty)
        for event in next {
            let at = store.inputs.capture.now.addingTimeInterval(event.hours * 3600)
            XCTAssertEqual(event.help, "Next run " + NDR1.hhmm(at, store.inputs.capture))
        }
    }

    func testOlderLabelAndBaselineDoNotOverlap() {
        let line = NDRail.baselineFrame(width: 200, olderLabelWidth: 36)
        XCTAssertGreaterThan(line.minX, 36)
        XCTAssertEqual(line.maxX, 200)
    }

    func testDottedRailFollowsScheduleRatherThanJobName() throws {
        for interval: TimeInterval? in [nil, 300, 3599, 3600, 86400] {
            var inputs = BrainBarDesignFixture.inputs(.calm)
            inputs.jobs = [.init(job: .walCheckpoint, enabled: true, continuous: false,
                                lastRun: inputs.capture.now.addingTimeInterval(-22 * 3600),
                                nextRun: inputs.capture.now.addingTimeInterval(24 * 60), loaded: true,
                                lastOutcome: .ok, scheduleInterval: interval)]
            let store = NDStore(fixture: inputs)
            let lane = try XCTUnwrap(NDJobsPage(store: store).lanes(store.model).first)
            let rail = try XCTUnwrap(find(NDRail.self, in: NDJobLaneView(store: store, lane: lane).body))
            let frequent = interval.map { $0 > 0 && $0 < 3600 } ?? false
            XCTAssertEqual(rail.dotted, frequent, "interval \(String(describing: interval))")
            XCTAssertEqual(rail.events.isEmpty, frequent)
        }
    }

    func testScheduleEvidenceUsesCalendarAndSupervisorIntervals() {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(secondsFromGMT: 0)!
        let now = BrainBarDesignFixture.inputs(.calm).capture.now
        let provider = BrainLayerLaunchdStatusProvider(commandRunner: { command in
            if command.first == "/bin/launchctl" { return .init(terminationStatus: 0, output: "runs = 1") }
            return .init(terminationStatus: 0, output: """
            {"source":"supervisor","runtime":{"configured":true,"paused":false,"starts":1,"failures":0,
            "next_run_at":1234,"interval":300}}
            """)
        }, uidProvider: { 501 }, runRecordDate: { _ in nil }, schedule: { _ in nil }, calendar: calendar,
        now: { now }, ownership: { .supervisor }, calendarHistory: { .empty },
        calendarSchedule: { DateComponents(hour: 4, minute: 0) }, calendarScheduleProblem: { nil },
        calendarLogsDirectory: { URL(fileURLWithPath: "/invented-calendar") }, calendarIsDisabled: { _ in false })
        let observations = provider.sampleActivity()
        XCTAssertEqual(observations[.walCheckpoint]?.scheduleInterval, 86400)
        XCTAssertEqual(observations[.drain]?.scheduleInterval, 300)
    }
}
