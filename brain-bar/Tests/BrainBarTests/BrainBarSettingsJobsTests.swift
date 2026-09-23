import Foundation
import XCTest
@testable import BrainBar

final class BrainBarSettingsJobsTests: XCTestCase {
    func testHumanGroupsOwnOnlyTheApprovedJobs() {
        XCTAssertEqual(BrainLayerLaunchdJobGroup.allCases.map(\.title), ["Ingest", "Backups", "Maintenance"])
        XCTAssertEqual(BrainLayerLaunchdJobGroup.ingest.jobs, [.watch, .index])
        XCTAssertEqual(BrainLayerLaunchdJobGroup.backups.jobs, [.backupDaily, .jsonlBackup])
        XCTAssertEqual(
            BrainLayerLaunchdJobGroup.maintenance.jobs,
            [.maintenanceNightly, .maintenanceWeekly]
        )
        XCTAssertEqual(
            BrainLayerLaunchdJobGroup.advancedJobs,
            [.walCheckpoint, .repairFTS, .decay, .drain, .hotlane]
        )
    }

    func testEveryJobLabelMatchesShippedPlist() throws {
        for job in BrainLayerLaunchdJob.allCases {
            let plist = try sourceFile("../scripts/launchd/\(job.launchdLabel).plist")
            XCTAssertTrue(plist.contains("<string>\(job.launchdLabel)</string>"), "\(job)")
        }
        XCTAssertEqual(BrainLayerLaunchdJob.hotlane.launchdLabel, "com.brainlayer.hotlane-brainbar")
    }

    func testReceiptAndScheduleReadersUseTheirOwnFiles() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let runDate = Date(timeIntervalSince1970: 1_784_465_400)
        try "1784465400\n".write(
            to: directory.appendingPathComponent("com.brainlayer.index.started"), atomically: true, encoding: .utf8
        )
        try """
        <?xml version="1.0" encoding="UTF-8"?>
        <plist version="1.0"><dict><key>StartCalendarInterval</key><dict>
        <key>Hour</key><integer>3</integer><key>Minute</key><integer>15</integer>
        </dict></dict></plist>
        """.write(to: directory.appendingPathComponent("com.brainlayer.index.plist"), atomically: true, encoding: .utf8)
        XCTAssertEqual(BrainLayerLaunchdStatusProvider.readRunRecord(.index, in: directory), runDate)
        XCTAssertEqual(BrainLayerLaunchdStatusProvider.readSchedule(.index, in: directory)?.hour, 3)
        XCTAssertNil(BrainLayerLaunchdStatusProvider.readRunRecord(.maintenanceNightly, in: directory))
    }

    func testGroupStatusUsesEveryMembersFixedLaunchdAndLogState() {
        let lastWatch = Date(timeIntervalSince1970: 1_784_462_400)
        let lastIndex = Date(timeIntervalSince1970: 1_784_458_800)
        let nextIndex = Date(timeIntervalSince1970: 1_784_520_900)
        let observations: [BrainLayerLaunchdJob: BrainLayerLaunchdJobObservation] = [
            .watch: .init(
                loadState: .running,
                runs: 4,
                lastExitCode: 0,
                lastRunAt: lastWatch,
                nextRunAt: nil,
                isContinuous: true
            ),
            .index: .init(
                loadState: .loaded,
                runs: 3,
                lastExitCode: 0,
                lastRunAt: lastIndex,
                nextRunAt: nextIndex,
                isContinuous: false
            ),
        ]

        let status = BrainLayerLaunchdJobGroup.ingest.status(
            settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: observations,
            formatDate: DashboardMetricFormatter.jobDateTimeString
        )

        XCTAssertEqual(status.health, .healthy)
        XCTAssertNil(status.attentionReason)
        XCTAssertEqual(
            status.lastRunText,
            "Watcher \(DashboardMetricFormatter.jobDateTimeString(lastWatch)) · " +
                "Index \(DashboardMetricFormatter.jobDateTimeString(lastIndex))"
        )
        XCTAssertEqual(
            status.nextRunText,
            "Watcher Continuous · Index \(DashboardMetricFormatter.jobDateTimeString(nextIndex))"
        )
        XCTAssertTrue(status.lastRunText.contains(", "))
        XCTAssertTrue(status.nextRunText.contains(", "))

        var failed = observations
        failed[.index] = .init(
            loadState: .loaded,
            runs: 4,
            lastExitCode: 1,
            lastRunAt: lastIndex,
            nextRunAt: nextIndex,
            isContinuous: false
        )
        XCTAssertEqual(
            BrainLayerLaunchdJobGroup.ingest.status(
                settings: BrainLayerConfig.defaultConfig.launchdJobs,
                observations: failed,
                formatDate: DashboardMetricFormatter.jobDateTimeString
            ).health,
            .unhealthy
        )
        XCTAssertEqual(
            BrainLayerLaunchdJobGroup.ingest.status(
                settings: BrainLayerConfig.defaultConfig.launchdJobs,
                observations: failed,
                formatDate: DashboardMetricFormatter.jobDateTimeString
            ).attentionReason,
            "Index last run exited 1 at \(DashboardMetricFormatter.jobDateTimeString(lastIndex))."
        )

        failed[.index] = .init(loadState: .loaded, runs: 0, lastExitCode: nil,
                                lastRunAt: lastIndex, nextRunAt: nextIndex, isContinuous: false)
        XCTAssertEqual(BrainLayerLaunchdJobGroup.ingest.status(
            settings: BrainLayerConfig.defaultConfig.launchdJobs,
            observations: failed, formatDate: DashboardMetricFormatter.jobDateTimeString
        ).health, .awaitingRun)
    }

    func testLaunchdProviderReadsRunReceiptAndInstalledPlistRatherThanLogMtime() throws {
        let now = Date(timeIntervalSince1970: 1_784_466_000)
        let runDate = Date(timeIntervalSince1970: 1_784_465_400)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(secondsFromGMT: 0))
        let output = """
        state = not running
        runs = 23
        last exit code = 0
        """
        let weeklyOutput = """
        state = not running
        runs = 2
        last exit code = 0
        """
        let provider = BrainLayerLaunchdStatusProvider(
            commandRunner: { command in
                if command.last?.hasSuffix("com.brainlayer.index") == true {
                    return BrainLayerLaunchdCommandResult(terminationStatus: 0, output: output)
                }
                if command.last?.hasSuffix("com.brainlayer.maintenance-weekly") == true {
                    return BrainLayerLaunchdCommandResult(terminationStatus: 0, output: weeklyOutput)
                }
                return BrainLayerLaunchdCommandResult(terminationStatus: 113, output: "Could not find service")
            },
            uidProvider: { 501 },
            runRecordDate: { job in job == .index ? runDate : nil },
            schedule: { job in
                job == .index ? DateComponents(hour: 3, minute: 15) :
                    job == .maintenanceWeekly ? DateComponents(hour: 4, minute: 0, weekday: 1) : nil
            },
            calendar: calendar,
            now: { now }
        )

        let observation = try XCTUnwrap(provider.sampleActivity()[.index])
        XCTAssertEqual(observation.loadState, .loaded)
        XCTAssertEqual(observation.runs, 23)
        XCTAssertEqual(observation.lastExitCode, 0)
        XCTAssertEqual(observation.lastRunAt, runDate)
        let next = try XCTUnwrap(observation.nextRunAt)
        XCTAssertEqual(calendar.component(.hour, from: next), 3)
        XCTAssertEqual(calendar.component(.minute, from: next), 15)
        XCTAssertGreaterThan(next, now)
        let weeklyNext = try XCTUnwrap(provider.sampleActivity()[.maintenanceWeekly]?.nextRunAt)
        XCTAssertEqual(calendar.component(.weekday, from: weeklyNext), 1)
    }

    func testSettingsPresentationHidesAdvancedByDefaultAndContainsNoEnrichmentSection() throws {
        XCTAssertFalse(BrainBarSettingsPresentation.defaultAdvancedExpanded)
        XCTAssertEqual(BrainBarSettingsPresentation.visibleAdvancedJobs(isExpanded: false), [])
        XCTAssertEqual(
            BrainBarSettingsPresentation.visibleAdvancedJobs(isExpanded: true),
            BrainLayerLaunchdJobGroup.advancedJobs
        )

        let settings = try sourceFile("Sources/BrainBar/BrainBarSettingsView.swift")
        let body = try sourceSlice(from: "var body: some View", throughBefore: "private var header", in: settings)
        XCTAssertFalse(body.contains("Enrichment"))
        XCTAssertFalse(settings.contains("private var enrichmentControls"))
    }

    private func sourceSlice(from start: String, throughBefore end: String, in source: String) throws -> String {
        let startRange = try XCTUnwrap(source.range(of: start))
        let endRange = try XCTUnwrap(source.range(of: end, range: startRange.upperBound..<source.endIndex))
        return String(source[startRange.lowerBound..<endRange.lowerBound])
    }

    private func sourceFile(_ relativePath: String) throws -> String {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: packageRoot.appendingPathComponent(relativePath), encoding: .utf8)
    }
}
