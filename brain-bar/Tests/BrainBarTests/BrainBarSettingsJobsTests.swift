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
            formatDate: DashboardMetricFormatter.shortAbsoluteTimeString
        )

        XCTAssertEqual(status.health, .healthy)
        XCTAssertEqual(
            status.lastRunText,
            "Watcher \(DashboardMetricFormatter.shortAbsoluteTimeString(lastWatch)) · Index \(DashboardMetricFormatter.shortAbsoluteTimeString(lastIndex))"
        )
        XCTAssertEqual(
            status.nextRunText,
            "Watcher Continuous · Index \(DashboardMetricFormatter.shortAbsoluteTimeString(nextIndex))"
        )

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
                formatDate: DashboardMetricFormatter.shortAbsoluteTimeString
            ).health,
            .unhealthy
        )
    }

    func testLaunchdProviderReadsExitScheduleAndLastRunFromJobLogs() throws {
        let now = Date(timeIntervalSince1970: 1_784_466_000)
        let logDate = Date(timeIntervalSince1970: 1_784_465_400)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = try XCTUnwrap(TimeZone(secondsFromGMT: 0))
        let output = """
        state = not running
        stdout path = /tmp/index.out.log
        stderr path = /tmp/index.err.log
        runs = 23
        last exit code = 0
                    "Minute" => 15
                    "Hour" => 3
        """
        let provider = BrainLayerLaunchdStatusProvider(
            commandRunner: { command in
                command.last?.hasSuffix("com.brainlayer.index") == true
                    ? BrainLayerLaunchdCommandResult(terminationStatus: 0, output: output)
                    : BrainLayerLaunchdCommandResult(terminationStatus: 113, output: "Could not find service")
            },
            uidProvider: { 501 },
            fileModificationDate: { url in
                url.path == "/tmp/index.err.log" ? logDate : nil
            },
            calendar: calendar,
            now: { now }
        )

        let observation = try XCTUnwrap(provider.sampleActivity()[.index])
        XCTAssertEqual(observation.loadState, .loaded)
        XCTAssertEqual(observation.runs, 23)
        XCTAssertEqual(observation.lastExitCode, 0)
        XCTAssertEqual(observation.lastRunAt, logDate)
        let next = try XCTUnwrap(observation.nextRunAt)
        XCTAssertEqual(calendar.component(.hour, from: next), 3)
        XCTAssertEqual(calendar.component(.minute, from: next), 15)
        XCTAssertGreaterThan(next, now)
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
