import AppKit
import XCTest
@testable import BrainBar

@MainActor
final class BadgeCadenceSupervisorTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_792_598_400)
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("badge-cadence-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: root)
    }

    // Explicit paths avoid process-wide env changes and reads of installed job files.
    // BRAINLAYER_SUPERVISOR_JOBS_PATH is the new reader override required by this contract.
    private var environment: [String: String] {
        [
            "BRAINLAYER_HEALTH_CHECK_PLIST_PATH": root.appendingPathComponent("health-check.plist").path,
            "BRAINLAYER_SUPERVISOR_JOBS_PATH": root.appendingPathComponent("jobs.json").path,
        ]
    }

    func testSupervisorCadenceIsKnownWhenLaunchAgentPlistIsAbsent() throws {
        try writeSupervisor(interval: 300)
        XCTAssertFalse(FileManager.default.fileExists(atPath: environment["BRAINLAYER_HEALTH_CHECK_PLIST_PATH"]!))
        let cadence = ObservabilityReader.healthCheckCadence(environment: environment)
        XCTAssertEqual(cadence.interval, 300)
        XCTAssertNil(cadence.assumption, "An installed supervisor interval is known, not an assumption.")
    }

    func testSupervisorCadenceUsesConfiguredIntervalRatherThanHardcodedDefault() throws {
        try writeSupervisor(interval: 420)
        let cadence = ObservabilityReader.healthCheckCadence(environment: environment)
        XCTAssertEqual(cadence.interval, 420)
        XCTAssertNil(cadence.assumption)
    }

    func testFreshBadgeWithLaunchAgentCadenceHasNoAttentionOrRedDot() throws {
        try writeLaunchAgent(interval: 420)
        let cadence = ObservabilityReader.healthCheckCadence(environment: environment)
        XCTAssertEqual(cadence.interval, 420)
        XCTAssertNil(cadence.assumption)
        // Older than the default's window, still fresh for the configured interval.
        try assertNoAttentionOrRedDot(readBadge(age: 650, cadence: cadence))
    }

    func testFreshBadgeWithSupervisorCadenceHasNoAttentionOrRedDot() throws {
        try writeSupervisor(interval: 420)
        let cadence = ObservabilityReader.healthCheckCadence(environment: environment)
        try assertNoAttentionOrRedDot(readBadge(age: 650, cadence: cadence))
    }

    func testFreshBadgeWithTrulyUnknownCadenceIsAtMostShortNeutralInfo() throws {
        let cadence = ObservabilityReader.healthCheckCadence(environment: environment)
        XCTAssertEqual(cadence.interval, 300)
        XCTAssertNotNil(cadence.assumption)
        let badge = try readBadge(age: 599, cadence: cadence)
        try assertNoAttentionOrRedDot(badge)
        XCTAssertLessThanOrEqual(badge.reason.count, 60)
        let title = BrainBarStatusPopoverController.statusLineTitle(for: badge)
        XCTAssertLessThanOrEqual(title.count, 60)
        XCTAssertFalse(title.contains(try XCTUnwrap(cadence.assumption)))
    }

    func testStaleBadgeBeyondTwiceCadenceStillNeedsAttentionAndRedDot() throws {
        for source in ["launch-agent", "supervisor", "unknown"] {
            for path in environment.values where FileManager.default.fileExists(atPath: path) {
                try FileManager.default.removeItem(atPath: path)
            }
            if source == "launch-agent" { try writeLaunchAgent(interval: 420) }
            if source == "supervisor" { try writeSupervisor(interval: 420) }
            let cadence = ObservabilityReader.healthCheckCadence(environment: environment)
            let expectedInterval: TimeInterval = source == "unknown" ? 300 : 420
            let badge = try readBadge(age: expectedInterval * 2 + 1, cadence: cadence)
            XCTAssertTrue(badge.badgeOn, source)
            XCTAssertTrue(badge.reason.localizedCaseInsensitiveContains("stale"), "\(source): \(badge.reason)")
            let title = BrainBarStatusPopoverController.statusLineTitle(for: badge)
            XCTAssertTrue(title.hasPrefix("Needs attention: "), source)
            XCTAssertLessThanOrEqual(title.count, 60, source)
            XCTAssertFalse(title.contains("assuming"), source)
            XCTAssertGreaterThan(try redPixelCount(for: badge), 0, source)
        }
    }

    func testBadgeAtExactlyTwiceKnownCadenceIsStillFresh() throws {
        try assertNoAttentionOrRedDot(readBadge(age: 600, cadence: .known(300)))
    }

    func testAttentionMenuLineIsAtMostSixtyCharactersInBothMenuPaths() throws {
        let message = "Health check failed; inspect its log for the detailed cause and retry after resolving the failure."
        let badge = try readBadge(age: 1, cadence: .known(300), activeMessage: message)
        XCTAssertTrue(badge.badgeOn, "Real producer alerts must remain visible.")
        let title = BrainBarStatusPopoverController.statusLineTitle(for: badge)
        XCTAssertTrue(title.hasPrefix("Needs attention: "))
        XCTAssertLessThanOrEqual(title.count, 60)
        XCTAssertEqual(BrainBarStatusPopoverController.menuRowTitles(for: badge).first, title)
        XCTAssertFalse(title.contains("cadence unknown"))
        XCTAssertFalse(title.contains("assuming"))
    }

    private func writeSupervisor(interval: Int) throws {
        let jobs: [String: Any] = [
            "com.brainlayer.health-check": [
                "enabled": true,
                "plist": [
                    "Label": "com.brainlayer.health-check",
                    "ProgramArguments": ["/usr/bin/true"],
                    "StartInterval": interval,
                ],
            ],
            // An unrelated interval must not win by dictionary order or recursive search.
            "com.brainlayer.drain": ["enabled": true, "plist": ["StartInterval": 15]],
        ]
        try JSONSerialization.data(withJSONObject: jobs).write(to: root.appendingPathComponent("jobs.json"))
    }

    private func writeLaunchAgent(interval: Int) throws {
        let data = try PropertyListSerialization.data(
            fromPropertyList: ["Label": "com.brainlayer.health-check", "StartInterval": interval],
            format: .xml, options: 0
        )
        try data.write(to: root.appendingPathComponent("health-check.plist"))
    }

    private func readBadge(
        age: TimeInterval,
        cadence: ObservabilityCadence,
        activeMessage: String? = nil
    ) throws -> BadgeStatePresentation {
        let active: [[String: String]] = activeMessage.map {
            [["code": "health_check_failed", "severity": "critical", "message": $0]]
        } ?? []
        let document: [String: Any] = [
            "schema_version": 1,
            "generated_at": ISO8601DateFormatter().string(from: now.addingTimeInterval(-age)),
            "alerts": [
                "state": "measured", "reason": "", "inputs": [],
                "badge_on": !active.isEmpty, "active": active, "suppressed": [],
            ],
        ]
        let url = root.appendingPathComponent("badge-state.json")
        try JSONSerialization.data(withJSONObject: document).write(to: url)
        return BadgeStateReader.read(url: url, now: now, cadence: cadence)
    }

    private func assertNoAttentionOrRedDot(
        _ badge: BadgeStatePresentation,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        XCTAssertFalse(badge.badgeOn, badge.reason, file: file, line: line)
        let title = BrainBarStatusPopoverController.statusLineTitle(for: badge)
        XCTAssertFalse(title.hasPrefix("Needs attention:"), title, file: file, line: line)
        XCTAssertEqual(BrainBarStatusPopoverController.menuRowTitles(for: badge).first, title, file: file, line: line)
        XCTAssertEqual(try redPixelCount(for: badge), 0, "The production icon factory must draw no red dot.", file: file, line: line)
    }

    private func redPixelCount(for badge: BadgeStatePresentation) throws -> Int {
        let stats = DashboardStats(
            chunkCount: 1, enrichedChunkCount: 1, pendingEnrichmentCount: 0,
            enrichmentPercent: 100, enrichmentRatePerMinute: 0, databaseSizeBytes: 0,
            recentActivityBuckets: [0], recentAgentWriteBuckets: [0],
            recentWatcherWriteBuckets: [0], recentEnrichmentBuckets: [0]
        )
        let image = BrainBarStatusPopoverController.statusIconImage(stats: stats, badgeOn: badge.badgeOn)
        let bitmap = try XCTUnwrap(NSBitmapImageRep(data: XCTUnwrap(image.tiffRepresentation)))
        var count = 0
        for y in 0..<bitmap.pixelsHigh {
            for x in 0..<bitmap.pixelsWide {
                let color = try XCTUnwrap(bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB))
                if color.redComponent > 0.7 && color.greenComponent < 0.4 && color.blueComponent < 0.4 && color.alphaComponent > 0.5 {
                    count += 1
                }
            }
        }
        return count
    }
}
