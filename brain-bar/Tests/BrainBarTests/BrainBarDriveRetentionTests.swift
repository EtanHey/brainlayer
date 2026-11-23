import AppKit
import SwiftUI
import Vision
import XCTest
@testable import BrainBar

@MainActor final class BrainBarDriveRetentionTests: XCTestCase {
    let referenceDate = Date(timeIntervalSince1970: 1_795_435_200) // 23 Nov 2026, noon UTC
    func testRoundTwoPresentationRegressions() throws {
        let evidence = try fixture { json in
            json["rule"] = ["daily_keep": 45, "monthly_keep": 100, "per_machine": true]
        }
        XCTAssertNotNil(BrainBarDriveRetentionPresentation(evidence: evidence).rule)
        XCTAssertEqual(BrainBarDriveRetentionPresentation.date("2026-11-23", relativeTo: referenceDate), "Nov 23")
        XCTAssertEqual(BrainBarDriveRetentionPresentation.date("2027-11-23", relativeTo: referenceDate), "Nov 23, 2027")
        let broken = try fixture { $0["config_error"] = "Fixture rule cannot be read" }
        XCTAssertTrue(BrainBarDriveRetentionPresentation(evidence: broken).atRest.contains("Fixture rule cannot be read"))
    }

    func testRoundTwoTimestampUsesLocalDay() {
        // An explicit zone: the result must not depend on the machine running the test (CI is UTC).
        let plusTwo = TimeZone(secondsFromGMT: 7200)!
        XCTAssertEqual(BrainBarDriveRetentionPresentation.date("2026-11-22T23:30:00Z", relativeTo: referenceDate, timeZone: plusTwo), "Nov 23")
        XCTAssertEqual(BrainBarDriveRetentionPresentation.date("2026-11-22T23:30:00.123Z", relativeTo: referenceDate, timeZone: plusTwo), "Nov 23")
        // The producer writes datetime.isoformat(): six fractional digits and a +00:00 offset.
        XCTAssertEqual(BrainBarDriveRetentionPresentation.date("2026-11-22T23:30:00.123456+00:00", relativeTo: referenceDate, timeZone: plusTwo), "Nov 23")
        XCTAssertEqual(BrainBarDriveRetentionPresentation.date("2026-11-22T23:30:00Z", relativeTo: referenceDate, timeZone: TimeZone(secondsFromGMT: 0)!), "Nov 22")
        XCTAssertEqual(BrainBarDriveRetentionPresentation.date("2024-08-31", relativeTo: referenceDate, timeZone: plusTwo), "Aug 31, 2024")
    }

    func testRoundTwoDrawnKeepsAndUnverified() throws {
        let evidence = try fixture { json in
            var machines = json["machines"] as! [[String: Any]]
            machines[0]["kept_by_rule"] = ["count": 8, "bytes": 8_000_000_000, "oldest": "2026-08-31", "newest": "2026-11-23"]
            machines[0]["unverified"] = ["count": 2, "bytes": 2_000_000_000, "oldest": "2026-06-01", "newest": "2026-11-22"]
            json["machines"] = machines
        }
        let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence, referenceDate: referenceDate), size: NSSize(width: 452, height: 720), appearance: .darkAqua)
        let text = try words(rep)
        XCTAssertTrue(text.contains("Keeps 8 copies, 8 GB, newest Nov 23"), text)
        XCTAssertTrue(text.contains("Not verified, kept for now: 2 copies, 2 GB"), text)
        // Kept lines first, the decision line last (round-2 review R2-1).
        let keeps = try XCTUnwrap(text.range(of: "Keeps 8 copies")), monthly = try XCTUnwrap(text.range(of: "Monthly:"))
        let unverified = try XCTUnwrap(text.range(of: "Not verified")), remove = try XCTUnwrap(text.range(of: "Would remove"))
        XCTAssertTrue(keeps.lowerBound < monthly.lowerBound && monthly.lowerBound < unverified.lowerBound
                      && unverified.lowerBound < remove.lowerBound, text)
        XCTAssertFalse(text.contains("2026"), text)
        XCTAssertFalse(text.contains("Keeps: 10 copies, 10 GB;"), text)
        XCTAssertEqual(text.components(separatedBy: "Not verified, kept for now:").count - 1, 1)
    }

    func testOptionalCohortsFallBackWithoutDateRange() throws {
        let evidence = try fixture { json in
            var machines = json["machines"] as! [[String: Any]]
            for index in machines.indices {
                machines[index].removeValue(forKey: "kept_by_rule")
                machines[index].removeValue(forKey: "unverified")
            }
            json["machines"] = machines
        }
        XCTAssertEqual(BrainBarDriveRetentionPresentation.keeps(evidence.machines[0]), "Keeps 10 copies, 10 GB")
        let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence, referenceDate: referenceDate), size: NSSize(width: 452, height: 670), appearance: .darkAqua)
        let text = try words(rep)
        XCTAssertTrue(text.contains("Keeps 10 copies, 10 GB"), text)
        XCTAssertFalse(text.contains(", newest"), text)
        XCTAssertFalse(text.contains("Not verified, kept for now"), text)
    }

    /// Without kept_by_rule, Keeps falls back to `kept`, which already includes the unverified copies; drawing the
    /// unverified line beside it would count them twice (round-2 review R2-4).
    func testUnverifiedAndPinnedLinesNeedKeptByRule() throws {
        let evidence = try fixture { json in
            var machines = json["machines"] as! [[String: Any]]
            machines[0].removeValue(forKey: "kept_by_rule")
            machines[0]["unverified"] = ["count": 2, "bytes": 2_000_000_000, "oldest": "2026-06-01", "newest": "2026-11-22"]
            machines[0]["pinned"] = ["count": 1, "bytes": 1_000_000_000, "oldest": "2026-11-15", "newest": "2026-11-15"]
            json["machines"] = machines
        }
        let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence, referenceDate: referenceDate), size: NSSize(width: 452, height: 720), appearance: .darkAqua)
        let text = try words(rep)
        XCTAssertTrue(text.contains("Keeps 10 copies, 10 GB"), text)
        XCTAssertFalse(text.contains("Not verified, kept for now"), text)
        XCTAssertFalse(text.contains("Pinned, kept for now"), text)
    }

    func testLongWrappedErrorAndMissingCaptionDrawCompletely() throws {
        let reason = "Fixture inventory cannot be measured because the daily backup has no readable size. The fixture account is connected, but this inventory entry has no byte count, so its removal size cannot be calculated safely. Wait for a fresh fixture inventory before comparing copy counts and sizes. Reconnect the fixture account if another measurement reports the same missing size. Final fixture detail."
        let evidence = try fixture { json in
            var machines = json["machines"] as! [[String: Any]]
            machines[0]["state"] = "unmeasurable"
            machines[0]["error"] = reason
            json["machines"] = [machines[0]]
        }
        let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence, referenceDate: referenceDate), size: NSSize(width: 452, height: 670), appearance: .darkAqua)
        let text = try words(rep)
        XCTAssertTrue(text.contains("Couldn't measure:"), text)
        XCTAssertTrue(text.contains("Final fixture detail."), text)
        let missing = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: nil, referenceDate: referenceDate), size: NSSize(width: 452, height: 400), appearance: .darkAqua)
        XCTAssertEqual(try words(missing).components(separatedBy: "Not measured yet").count - 1, 1)
        try save(rep, "sheet-long-error-dark")
    }

    func testRoundTwoThreeLineMachineUsesIntrinsicHeight() throws {
        let one = try fixture { json in
            var machines = json["machines"] as! [[String: Any]]
            machines[1]["monthly_representatives"] = []
            json["machines"] = [machines[1]]
        }
        let two = try fixture { json in
            var machines = json["machines"] as! [[String: Any]]
            machines[0] = machines[1]
            json["machines"] = machines
        }
        func height(_ evidence: BrainBarDriveRetention) -> CGFloat {
            let host = NSHostingView(rootView: BrainBarDriveRetentionSheet(evidence: evidence, referenceDate: referenceDate))
            host.frame = NSRect(x: 0, y: 0, width: 452, height: 800)
            host.layoutSubtreeIfNeeded()
            return host.fittingSize.height
        }
        let addedHeight = height(two) - height(one)
        print("Three-line Mac added drawn height: \(addedHeight) pt")
        XCTAssertGreaterThan(addedHeight, 50)
        XCTAssertLessThan(addedHeight, 95, "A three-line Mac needs only its drawn content and padding")
        let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: one, referenceDate: referenceDate), size: NSSize(width: 452, height: 670), appearance: .darkAqua)
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        try VNImageRequestHandler(cgImage: try XCTUnwrap(rep.cgImage)).perform([request])
        let remove = try XCTUnwrap(request.results?.first { $0.topCandidates(1).first?.string.contains("Would remove:") == true })
        let total = try XCTUnwrap(request.results?.first { $0.topCandidates(1).first?.string.contains("The first cleanup") == true })
        let drawnGap = (remove.boundingBox.minY - total.boundingBox.maxY) * rep.size.height
        print("Three-line Mac drawn gap to cleanup caption: \(drawnGap) pt")
        XCTAssertLessThan(drawnGap, 36, "No empty band after the three-line Mac in a tall host")
        let many = try fixture { $0["machines"] = Array(repeating: ["machine_id": "Fixture Mac", "state": "not_run"], count: 6) }
        let more = try fixture { $0["machines"] = Array(repeating: ["machine_id": "Fixture Mac", "state": "not_run"], count: 12) }
        XCTAssertEqual(height(many), height(more), accuracy: 1, "Overflow scrolls within the same capped card")
        XCTAssertGreaterThan(height(many), height(one) + 150)
        try save(rep, "sheet-three-line-dark")
    }

    func testRoundTwoConfigurationErrorDrawsAtRestInBothDesigns() async throws {
        let evidence = try fixture { $0["config_error"] = "Fixture rule cannot be read" }
        let model = BrainBarDriveAuthModel(runner: ValidDriveRunner()); await model.refreshStatus()
        for (name, appearance) in NDRenderHarness.appearances {
            var inputs = BrainBarDesignFixture.inputs(.calm); inputs.driveRetention = evidence
            let new = try renderSettings(NDStore(fixture: inputs), appearance: appearance)
            let classic = try NDRenderHarness.host(BrainBarDriveAuthCard(model: model, retention: evidence).padding(20).frame(width: 480, height: 170, alignment: .leading).background(Color.brainBarBackgroundBase).foregroundStyle(Color.brainBarTextPrimary), size: NSSize(width: 480, height: 170), appearance: appearance)
            for (design, rep) in [("new", new), ("classic", classic)] {
                let text = try words(rep)
                XCTAssertTrue(text.contains("Fixture rule cannot be read"), text)
                try save(rep, "\(design)-settings-config-error-\(name)")
            }
        }
    }
    func testOptionalRetentionSurvivesBackupDecoding() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let json = """
        {"state":"measured","reason":"","inputs":[],"drive_retention":{
          "rule":{"daily_keep":7,"monthly_keep":3,"per_machine":true,"description":"Fixture rule"},
          "writes_enabled":false,"config_error":null,"machines":[]}}
        """
        let backups = try decoder.decode(ObservabilityDocument.Backups.self, from: Data(json.utf8))
        let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(backups)) as! [String: Any]
        XCTAssertNotNil(encoded["driveRetention"], "The optional producer section must reach Settings")
    }

    func testAbsentSectionStillDecodes() throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let backups = try decoder.decode(ObservabilityDocument.Backups.self, from: Data(#"{"state":"measured","reason":"","inputs":[]}"#.utf8))
        XCTAssertEqual(backups.state, "measured")
    }

    func testSettingsDrawsNotReportedWithoutInventingRule() throws {
        let store = NDStore(fixture: BrainBarDesignFixture.inputs(.calm))
        for (name, appearance) in NDRenderHarness.appearances {
            let rep = try renderSettings(store, appearance: appearance)
            let text = try words(rep)
            XCTAssertTrue(text.contains("rule is not reported yet"), text)
            XCTAssertFalse(text.contains("7 daily"), text)
            try save(rep, "new-settings-missing-\(name)")
        }
    }

    struct ValidDriveRunner: BrainLayerCLIRunning {
        func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult? {
            .init(terminationStatus: 0, stdout: #"{"state":"valid"}"#)
        }
    }

    func testClassicDriveDrawsNotReported() async throws {
        let model = BrainBarDriveAuthModel(runner: ValidDriveRunner())
        await model.refreshStatus()
        for (name, appearance) in NDRenderHarness.appearances {
            let rep = try NDRenderHarness.host(BrainBarDriveAuthCard(model: model).padding(20).frame(width: 480, height: 170, alignment: .leading).background(Color.brainBarBackgroundBase).foregroundStyle(Color.brainBarTextPrimary),
                                                size: NSSize(width: 480, height: 170), appearance: appearance)
            let text = try words(rep)
            XCTAssertTrue(text.contains("rule is not reported yet"), text)
            try save(rep, "classic-settings-missing-\(name)")
        }
    }

    func words(_ rep: NSBitmapImageRep) throws -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        try VNImageRequestHandler(cgImage: try XCTUnwrap(rep.cgImage)).perform([request])
        return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ")
    }

    func settings(_ store: NDStore) -> some View {
        store.refreshSettingsFileSummary()
        let defaults = UserDefaults(suiteName: "brainbar.retention-test.\(UUID().uuidString)")!
        defaults.register(defaults: [BrainBarDesign.key: BrainBarDesign.new.rawValue])
        return NDSettingsView(store: store).defaultAppStorage(defaults)
            .overlay(alignment: .top) {
                Text(BrainBarMenuPresenter.SurfaceSay.settingsTitle).font(NDType.section).foregroundStyle(ND.t2)
                    .frame(height: NDSettingsWindowController.titleBar)
            }
    }

    func renderSettings(_ store: NDStore, appearance: NSAppearance.Name) throws -> NSBitmapImageRep {
        let view = settings(store)
        let host = NSHostingView(rootView: view)
        host.frame = NSRect(x: 0, y: 0, width: 480, height: 420)
        host.layoutSubtreeIfNeeded()
        return try NDRenderHarness.host(view, size: NSSize(width: 480, height: ceil(host.fittingSize.height)), appearance: appearance)
    }

    func fixture(_ change: (inout [String: Any]) -> Void = { _ in }) throws -> BrainBarDriveRetention {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "drive-retention-invented", withExtension: "json", subdirectory: "Fixtures"))
        var json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        change(&json)
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(BrainBarDriveRetention.self, from: JSONSerialization.data(withJSONObject: json))
    }

    func testTwoMeasuredMacsAndReadOnlyControlsDraw() async throws {
        let evidence = try fixture(), p = BrainBarDriveRetentionPresentation(evidence: try fixture())
        XCTAssertEqual(p.total?.count, 30); XCTAssertEqual(p.total?.bytes, 30_000_000_000)
        XCTAssertFalse(evidence.writesEnabled); XCTAssertFalse(p.canEnableCleanup)
        XCTAssertEqual(p.atRest, "7 daily and 3 monthly copies per Mac. Cleanup is off.")
        let model = BrainBarDriveAuthModel(runner: ValidDriveRunner()); await model.refreshStatus()
        for (name, appearance) in NDRenderHarness.appearances {
            var inputs = BrainBarDesignFixture.inputs(.calm); inputs.driveRetention = evidence
            let store = NDStore(fixture: inputs)
            let new = try renderSettings(store, appearance: appearance)
            let classic = try NDRenderHarness.host(BrainBarDriveAuthCard(model: model, retention: evidence).padding(20).frame(width: 480, height: 170, alignment: .leading).background(Color.brainBarBackgroundBase).foregroundStyle(Color.brainBarTextPrimary), size: NSSize(width: 480, height: 170), appearance: appearance)
            for (design, rep) in [("new", new), ("classic", classic)] {
                let drawn = try words(rep)
                XCTAssertTrue(drawn.contains(p.atRest), drawn)
                try save(rep, "\(design)-settings-measured-\(name)")
            }
            let sheet = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence, referenceDate: referenceDate), size: NSSize(width: 452, height: 670), appearance: appearance)
            let text = try words(sheet)
            for expected in ["Fixture Mac A", "Fixture Mac B", "20 copies", "30 copies", "Keeps 8", "Would remove:", "30 GB", "previous full months", "Cleanup is off", "Done"] {
                XCTAssertTrue(text.contains(expected), "Missing \(expected): \(text)")
            }
            XCTAssertFalse(text.contains("Across all Macs")); XCTAssertFalse(text.contains("Turn on cleanup"))
            try save(sheet, "sheet-measured-\(name)")
            let missing = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: nil, referenceDate: referenceDate), size: NSSize(width: 452, height: 400), appearance: appearance)
            XCTAssertTrue(try words(missing).contains("rule is not reported yet"))
            try save(missing, "sheet-missing-\(name)")
        }
    }

    func testStaleUnmeasurableAndNullNeverInventZero() throws {
        for state in ["stale", "unmeasurable", "not_run", "measured"] {
            let evidence = try fixture { json in
                var machines = json["machines"] as! [[String: Any]]
                machines[1]["state"] = state; machines[1]["would_remove"] = NSNull()
                if state != "measured" && state != "not_run" { machines[1]["error"] = "Fixture inventory unavailable" }
                json["machines"] = machines
            }
            let p = BrainBarDriveRetentionPresentation(evidence: evidence)
            XCTAssertNil(p.total); XCTAssertEqual(p.firstCleanup, "Not measured yet")
            for (name, appearance) in NDRenderHarness.appearances {
                let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence, referenceDate: referenceDate), size: NSSize(width: 452, height: 670), appearance: appearance)
                let text = try words(rep)
                XCTAssertTrue(text.contains(state == "measured" || state == "not_run" ? "Not measured yet" : "Fixture inventory unavailable"), text)
                XCTAssertNil(text.range(of: #"\b0 copies\b"#, options: .regularExpression), text)
                if state == "stale" { XCTAssertTrue(text.contains("Out of date"), text) }
                if state == "unmeasurable" { XCTAssertTrue(text.contains("Couldn't measure"), text) }
                try save(rep, "sheet-\(state == "measured" ? "null-summary" : state)-\(name)")
            }
        }
    }

    func testConfigurationErrorAndUnexpectedWritesStayOff() throws {
        let evidence = try fixture { $0["config_error"] = "Fixture rule cannot be read"; $0["writes_enabled"] = true }
        let p = BrainBarDriveRetentionPresentation(evidence: evidence)
        XCTAssertNil(p.rule); XCTAssertNil(p.total); XCTAssertFalse(p.canEnableCleanup)
        XCTAssertTrue(p.atRest.contains("Fixture rule cannot be read"))
        let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence, referenceDate: referenceDate), size: NSSize(width: 452, height: 600), appearance: .darkAqua)
        XCTAssertTrue(try words(rep).contains("Fixture rule cannot be read"))
        let confirm = try NDRenderHarness.host(BrainBarDriveRetentionConfirmation(presentation: .init(evidence: try fixture()), cancel: {})
            .padding(20).frame(width: 452, height: 160).background(ND.bg).foregroundStyle(ND.t1)
            .environment(\.ndPalette, .window), size: NSSize(width: 452, height: 160), appearance: .darkAqua)
        let text = try words(confirm)
        XCTAssertTrue(text.contains("Move 30 copies"), text); XCTAssertTrue(text.contains("Cancel"), text)
        try save(confirm, "confirmation-unreachable-dark")
    }

    func save(_ rep: NSBitmapImageRep, _ name: String) throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_RETENTION_PROOF"] else { return }
        let directory = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try NDRenderHarness.write(rep, name: name, in: directory)
    }
}

extension BrainBarDriveRetentionTests {
    /// A weekly-pinned copy is part of `kept`; with the producer's `pinned` summary it gets its own line, between the
    /// other kept lines and Would remove, and zero pinned copies draw nothing.
    func testPinnedLineDrawnOnlyWhenCountAboveZero() throws {
        let evidence = try fixture { json in
            var machines = json["machines"] as! [[String: Any]]
            machines[0]["kept_by_rule"] = ["count": 7, "bytes": 7_000_000_000, "oldest": "2026-08-31", "newest": "2026-11-23"]
            machines[0]["unverified"] = ["count": 2, "bytes": 2_000_000_000, "oldest": "2026-06-01", "newest": "2026-11-22"]
            machines[0]["pinned"] = ["count": 1, "bytes": 1_000_000_000, "oldest": "2026-11-15", "newest": "2026-11-15"]
            machines[1]["kept_by_rule"] = ["count": 10, "bytes": 10_000_000_000, "oldest": "2026-08-31", "newest": "2026-11-23"]
            machines[1]["pinned"] = ["count": 0, "bytes": 0, "oldest": NSNull(), "newest": NSNull()]
            json["machines"] = machines
        }
        let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence, referenceDate: referenceDate), size: NSSize(width: 452, height: 760), appearance: .darkAqua)
        let text = try words(rep)
        XCTAssertTrue(text.contains("Pinned, kept for now: 1 copy, 1 GB"), text)
        XCTAssertEqual(text.components(separatedBy: "Pinned, kept for now:").count - 1, 1, text)
        let pinned = try XCTUnwrap(text.range(of: "Pinned")), remove = try XCTUnwrap(text.range(of: "Would remove"))
        XCTAssertTrue(pinned.lowerBound < remove.lowerBound, text)
        try save(rep, "sheet-pinned-dark")
    }
}
