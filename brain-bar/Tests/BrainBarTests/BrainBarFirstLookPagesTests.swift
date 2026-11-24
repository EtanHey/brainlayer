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

    func testNewIsTheOnlyDesignDespiteStoredClassic() {
        let defaults = DesignDefaults()
        XCTAssertEqual(BrainBarDesign.current(in: defaults), .new)
        XCTAssertNil(defaults.choice, "reading does not store a preference")
        defaults.choice = "classic"
        XCTAssertEqual(BrainBarDesign.current(in: defaults), .new)
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

    func testOlderLabelAndBaselineDoNotOverlap() throws {
        let line = NDRail.baselineFrame(width: 200)
        let label = NDRail.olderLabelFrame(width: 36)
        XCTAssertLessThan(label.maxX, line.minX)
        XCTAssertEqual(line.minX, 0)
        XCTAssertEqual(line.maxX, 200)
        var inputs = BrainBarDesignFixture.inputs(.nightlyFailing)
        inputs.jobs = [.init(job: .maintenanceNightly, enabled: true, continuous: false,
                            lastRun: inputs.capture.now.addingTimeInterval(-11 * 3600), nextRun: nil,
                            loaded: true, lastOutcome: .failed)]
        let store = NDStore(fixture: inputs)
        let lane = try XCTUnwrap(NDJobsPage(store: store).lanes(store.model).first)
        XCTAssertEqual(lane.streak, 3)
        XCTAssertNotNil(lane.olderLabel)
        let dot = try XCTUnwrap(lane.events.first { $0.kind == .fail && $0.hours == -35 })
        let dotX = NDRail.timePosition(dot.hours, width: 200, past: NDJobsPage.past, ahead: NDJobsPage.ahead)
        XCTAssertEqual(dotX, 200 / 48, accuracy: 0.01, "time coordinates stay on the shared axis")
        XCTAssertLessThan(label.maxX, dotX - 3.5, "older-count slot clears the left-edge failure dot")
    }

    func testEveryStoredDesignResolvesToNewWithoutRewritingTheValue() {
        for raw in ["invalid", "", "classic", "new"] {
            let defaults = DesignDefaults(choice: raw)
            XCTAssertEqual(BrainBarDesign.current(in: defaults), .new)
            XCTAssertEqual(defaults.choice, raw)
        }
    }

    func testInstalledCalendarRetainsMonthlyConstraints() throws {
        let directory = brainBarTestTemporaryDirectory().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let label = "invented.monthly"
        let capture = BrainBarDesignFixture.inputs(.calm).capture
        for fields in [["Day": 1, "Hour": 4, "Minute": 0],
                       ["Month": 2, "Day": 1, "Weekday": 1, "Hour": 4, "Minute": 0]] {
            let plist: [String: Any] = ["StartCalendarInterval": fields]
            let data = try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
            try data.write(to: directory.appendingPathComponent(label + ".plist"))
            let schedule = try XCTUnwrap(BrainLayerLaunchdStatusProvider.readSchedule(label: label, in: directory))
            XCTAssertEqual(schedule.month, fields["Month"])
            XCTAssertEqual(schedule.day, 1)
            XCTAssertEqual(schedule.weekday, fields["Weekday"].map { ($0 % 7) + 1 })
            XCTAssertEqual(schedule.hour, 4)
            for job in [BrainLayerLaunchdJob.drain, .walCheckpoint] {
                let input = NDJobInput(job: job, enabled: true, continuous: false, lastRun: nil,
                                       nextRun: capture.now.addingTimeInterval(3600), loaded: true,
                                       scheduleInterval: 86400, scheduleCalendar: schedule)
                let help = NDJobLaneView.jobHelp(input, job: job, capture: capture)
                XCTAssertFalse(help.contains("Runs daily"), help)
                XCTAssertTrue(help.contains("Runs every"), help)
            }
        }
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
            for calendar in [DateComponents(minute: 0), DateComponents(day: 1, hour: 4), DateComponents(hour: 4, weekday: 2)] {
                input.scheduleCalendar = calendar
                input.scheduleInterval = 3600
                let help = NDJobLaneView.jobHelp(input, job: job, capture: capture)
                XCTAssertTrue(help.hasSuffix("Runs every 1\u{00A0}h."), help)
                XCTAssertFalse(help.contains("Runs daily"), help)
            }
        }
    }

    func testRetentionErrorIsADatedFailureWithoutMakingConversationHistoryUnknown() throws {
        let now = BrainBarDesignFixture.inputs(.calm).capture.now
        let at = now.addingTimeInterval(-3600)
        let date = ISO8601DateFormatter().string(from: at)
        let error = #"{"event":"local_retention_error","error":"invented inventory unavailable"}"#
        let upload = "{\"attempted_at\":\"\(date)\",\"uploaded\":true,\"verified\":true}"
        let retention = NDBackupHistoryReader.Log(lines: [Substring(error), Substring(upload)], unreadable: false)
        let conversation = try XCTUnwrap(NDBackupHistoryReader.parseConversations(retention, now: now))
        let attempts = try XCTUnwrap(conversation.attempts)
        XCTAssertEqual(attempts.count, 2)
        XCTAssertEqual(attempts.filter { !$0.ok }.count, 1)
        XCTAssertEqual(attempts.first?.at, at)
        XCTAssertEqual(attempts.first?.isErrorEvent, true)
        XCTAssertEqual(conversation.skipped, 0)
    }

    func testDatabaseRetentionErrorIsNotSkipped() throws {
        let now = BrainBarDesignFixture.inputs(.calm).capture.now
        let error = #"{"event":"local_retention_error","error":"invented inventory unavailable"}"#
        let retention = NDBackupHistoryReader.Log(lines: [Substring(error)], unreadable: false)
        let database = try XCTUnwrap(NDBackupHistoryReader.parseDatabase(retention, now: now))
        XCTAssertNil(database.local)
        XCTAssertEqual(database.skipped, 1)
    }

    func testUnprovenUndatedRowsAreNotSkipped() {
        let now = BrainBarDesignFixture.inputs(.calm).capture.now
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
