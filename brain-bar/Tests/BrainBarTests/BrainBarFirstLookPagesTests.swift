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
        XCTAssertEqual(NDActivitySection.headerHeight, 22, "scale captions must not increase the Activity header")
        let wave = NDWave(saves: [1, 7, 2], reads: [31, 4, 3])
        let labels = NDActivitySection.scaleLabels(wave, minutes: 3)
        XCTAssertEqual(labels.saves, "up to 7 per min")
        XCTAssertEqual(labels.reads, "up to 31 per min")
        XCTAssertNotEqual(labels.saves, labels.reads)
        var sliding = wave
        sliding.outgoing = .init(saves: 17, reads: 43)
        XCTAssertEqual(NDActivitySection.scaleLabels(sliding, minutes: 3).saves, "up to 17 per min")
        XCTAssertEqual(NDActivitySection.scaleLabels(sliding, minutes: 3).reads, "up to 43 per min")
        let wide = NDWave(saves: Array(repeating: 7, count: 60), reads: Array(repeating: 31, count: 60))
        XCTAssertEqual(NDActivitySection.scaleLabels(wide, minutes: 180).saves, "up to 7 per 3 min")
        XCTAssertEqual(NDActivitySection.scaleLabels(wide, minutes: 1440).reads, "up to 31 per 24 min")
        let zero = NDActivitySection.scaleLabels(NDWave(saves: [0], reads: [0]), minutes: 1)
        XCTAssertNil(zero.saves)
        XCTAssertNil(zero.reads)
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
        let old = ISO8601DateFormatter().string(from: now.addingTimeInterval(-NDBackupHistoryReader.windowS - 3600))
        let lines = ["{\"uploaded\":true,\"ts\":\"\(old)\"}", "{\"event\":\"scan\",\"attempted_at\":\"\(at)\"}",
                     "{\"attempted_at\":\"\(at)\",\"uploaded\":true,\"verified\":true}"]
        let parsed = try XCTUnwrap(NDBackupHistoryReader.parseConversations(.init(lines: lines.map { Substring($0) }, unreadable: false), now: now))
        XCTAssertEqual(parsed.attempts?.map(\.ok), [true])
        XCTAssertEqual(parsed.skipped, 0)
    }

    func testUndatedAndEventDatabaseRowsPreserveGoodDots() throws {
        let now = BrainBarDesignFixture.inputs(.calm).capture.now
        let at = ISO8601DateFormatter().string(from: now.addingTimeInterval(-3600))
        let old = ISO8601DateFormatter().string(from: now.addingTimeInterval(-NDBackupHistoryReader.windowS - 3600))
        let lines = ["{\"backup_log_provenance\":\"real\",\"ts\":\"\(old)\"}",
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

    func testBothJobHoversUseTheSameCadenceAsTheirRails() {
        let at = ISO8601DateFormatter().date(from: "2026-11-23T04:00:00Z")!
        let capture = BrainBarMenuCapture(now: at.addingTimeInterval(-3600), timeZone: TimeZone(secondsFromGMT: 0)!)
        for job in [BrainLayerLaunchdJob.drain, .walCheckpoint] {
            var input = NDJobInput(job: job, enabled: true, continuous: false, lastRun: nil, nextRun: at,
                                   loaded: true, scheduleInterval: 300)
            XCTAssertTrue(input.runsFrequently)
            XCTAssertTrue(NDJobLaneView.jobHelp(input, job: job, capture: capture).hasSuffix("Every few minutes."))
            input.scheduleInterval = 86400
            input.scheduleCalendar = DateComponents(hour: 4, minute: 0)
            XCTAssertFalse(input.runsFrequently)
            let daily = NDJobLaneView.jobHelp(input, job: job, capture: capture)
            XCTAssertTrue(daily.hasSuffix("Runs daily at 04:00."))
            XCTAssertFalse(daily.contains("Every few minutes"))
            input.scheduleInterval = 82800
            XCTAssertTrue(NDJobLaneView.jobHelp(input, job: job, capture: capture).hasSuffix("Runs daily at 04:00."),
                          "a daylight-saving transition does not turn a calendar job into a 23-hour timer")
        }
    }

    func testRetentionErrorAndUnprovenUndatedRowsAreNotSkipped() throws {
        let now = BrainBarDesignFixture.inputs(.calm).capture.now
        let error = #"{"event":"local_retention_error","error":"invented inventory unavailable"}"#
        let retention = NDBackupHistoryReader.Log(lines: [Substring(error)], unreadable: false)
        let conversation = try XCTUnwrap(NDBackupHistoryReader.parseConversations(retention, now: now))
        XCTAssertNil(conversation.attempts)
        XCTAssertEqual(conversation.skipped, 1)
        let database = try XCTUnwrap(NDBackupHistoryReader.parseDatabase(retention, now: now))
        XCTAssertNil(database.local)
        XCTAssertEqual(database.skipped, 1)
        let undated = NDBackupHistoryReader.Log(lines: [Substring(#"{"uploaded":true,"verified":true}"#)], unreadable: false)
        XCTAssertNil(NDBackupHistoryReader.parseConversations(undated, now: now)?.attempts)
    }

    @MainActor private final class RangeGate {
        var heroRequests: [Int] = []
        var active = 0
        var maximumActive = 0
        private var dayStarted = false
        private var onStart: CheckedContinuation<Void, Never>?
        private var onRelease: CheckedContinuation<Void, Never>?

        func waitForDay() async {
            if dayStarted { return }
            await withCheckedContinuation { onStart = $0 }
        }

        func releaseDay() { onRelease?.resume(); onRelease = nil }

        func fetch(_ minutes: Int, _ count: Int) async -> BrainDatabase.PipelineWindowBuckets? {
            if count == 60 {
                heroRequests.append(minutes)
                active += 1
                maximumActive = max(maximumActive, active)
                if minutes == 1440, !dayStarted {
                    dayStarted = true
                    onStart?.resume(); onStart = nil
                    await withCheckedContinuation { onRelease = $0 }
                }
                active -= 1
            }
            return .init(activityWindowMinutes: minutes, bucketCount: count, allWriteBuckets: [minutes],
                         agentWriteBuckets: [minutes], watcherWriteBuckets: [count], enrichmentBuckets: [],
                         watcherFlowReadability: .readable)
        }
    }

    func testLatestRangeWaitsForTheActiveQueryAndDropsIntermediateQueries() async {
        let gate = RangeGate()
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        let now = store.inputs.capture.now
        store.waveNow = { now }
        store.fetchWaveBuckets = { minutes, count in await gate.fetch(minutes, count) }
        store.setWindowActive(true)
        await store.waitForWaveFetchForTesting()
        store.holdChart(true)
        store.range = .day
        await gate.waitForDay()
        store.range = .hour
        store.range = .day
        store.range = .threeHours
        store.waveNow = { now.addingTimeInterval(16) }
        store.refreshWaves()
        gate.releaseDay()
        await store.waitForWaveFetchForTesting()
        XCTAssertEqual(gate.heroRequests, [60, 1440, 180])
        XCTAssertEqual(gate.maximumActive, 1)
        XCTAssertEqual(store.inputs.heroWaves[.threeHours]?.saves, [180])
        XCTAssertEqual(store.inputs.heroWaves[.threeHours]?.endsAt, now.addingTimeInterval(16))
        XCTAssertNotEqual(store.inputs.heroWaves[.day]?.saves, [1440], "the superseded result is discarded")
        store.holdChart(false)
        store.setWindowActive(false)
    }
}
