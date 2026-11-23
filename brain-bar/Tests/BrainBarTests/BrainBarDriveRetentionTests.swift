import AppKit
import SwiftUI
import Vision
import XCTest
@testable import BrainBar

@MainActor final class BrainBarDriveRetentionTests: XCTestCase {
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
            let rep = try NDRenderHarness.host(settings(store), size: NSSize(width: 480, height: 420), appearance: appearance)
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
        let defaults = UserDefaults(suiteName: "brainbar.retention-test.\(UUID().uuidString)")!
        defaults.register(defaults: [BrainBarDesign.key: BrainBarDesign.new.rawValue])
        return NDSettingsView(store: store).defaultAppStorage(defaults)
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
            let new = try NDRenderHarness.host(settings(store), size: NSSize(width: 480, height: 420), appearance: appearance)
            let classic = try NDRenderHarness.host(BrainBarDriveAuthCard(model: model, retention: evidence).padding(20).frame(width: 480, height: 170, alignment: .leading).background(Color.brainBarBackgroundBase).foregroundStyle(Color.brainBarTextPrimary), size: NSSize(width: 480, height: 170), appearance: appearance)
            for (design, rep) in [("new", new), ("classic", classic)] {
                let drawn = try words(rep)
                XCTAssertTrue(drawn.contains(p.atRest), drawn)
                try save(rep, "\(design)-settings-measured-\(name)")
            }
            let sheet = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence), size: NSSize(width: 452, height: 670), appearance: appearance)
            let text = try words(sheet)
            for expected in ["Fixture Mac A", "Fixture Mac B", "20 copies", "30 copies", "Keeps:", "Would remove:", "30 GB", "previous full months", "Cleanup is off", "Done"] {
                XCTAssertTrue(text.contains(expected), "Missing \(expected): \(text)")
            }
            XCTAssertFalse(text.contains("Across all Macs")); XCTAssertFalse(text.contains("Turn on cleanup"))
            try save(sheet, "sheet-measured-\(name)")
            let missing = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: nil), size: NSSize(width: 452, height: 400), appearance: appearance)
            XCTAssertTrue(try words(missing).contains("rule is not reported yet"))
            try save(missing, "sheet-missing-\(name)")
        }
    }

    func testStaleUnmeasurableAndNullNeverInventZero() throws {
        for state in ["stale", "unmeasurable", "not_run", "measured"] {
            let evidence = try fixture { json in
                var machines = json["machines"] as! [[String: Any]]
                machines[1]["state"] = state; machines[1]["would_remove"] = NSNull()
                if state != "measured" { machines[1]["error"] = "Fixture inventory unavailable" }
                json["machines"] = machines
            }
            let p = BrainBarDriveRetentionPresentation(evidence: evidence)
            XCTAssertNil(p.total); XCTAssertEqual(p.firstCleanup, "Not measured yet")
            let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence), size: NSSize(width: 452, height: 670), appearance: .darkAqua)
            let text = try words(rep)
            XCTAssertTrue(text.contains(state == "measured" ? "Not measured yet" : "Fixture inventory unavailable"), text)
            XCTAssertNil(text.range(of: #"\b0 copies\b"#, options: .regularExpression), text)
            if state == "stale" { XCTAssertTrue(text.contains("Stale"), text) }
            try save(rep, "sheet-\(state)-dark")
        }
    }

    func testConfigurationErrorAndUnexpectedWritesStayOff() throws {
        let evidence = try fixture { $0["config_error"] = "Fixture rule cannot be read"; $0["writes_enabled"] = true }
        let p = BrainBarDriveRetentionPresentation(evidence: evidence)
        XCTAssertNil(p.rule); XCTAssertNil(p.total); XCTAssertFalse(p.canEnableCleanup)
        XCTAssertTrue(p.atRest.contains("not reported yet"))
        let rep = try NDRenderHarness.host(BrainBarDriveRetentionSheet(evidence: evidence), size: NSSize(width: 452, height: 600), appearance: .darkAqua)
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
