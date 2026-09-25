// SecretScrubberGoldenTests.swift — BrainBar's Swift scrubber must agree with the
// Python scrubber, case for case.
//
// Both suites assert the SAME file, tests/fixtures/secret_scrub/golden.json:
// tests/test_secret_scrub_golden.py for Python, this file for Swift. A rule change
// in one implementation without the other fails one of the two. Every token in the
// fixture is synthetic (zeros or an alphabet walk).

import Foundation
import XCTest
@testable import BrainBar

final class SecretScrubberGoldenTests: XCTestCase {
    private struct GoldenCase: Decodable {
        let name: String
        let input: String
        let expectedText: String
        let expectedProviders: [String]
        let expectedQuarantineCount: Int

        enum CodingKeys: String, CodingKey {
            case name
            case input
            case expectedText = "expected_text"
            case expectedProviders = "expected_providers"
            case expectedQuarantineCount = "expected_quarantine_count"
        }
    }

    private struct GoldenFile: Decodable {
        let cases: [GoldenCase]
    }

    private func goldenCases() throws -> [GoldenCase] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("tests/fixtures/secret_scrub/golden.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(GoldenFile.self, from: data).cases.map { golden in
            GoldenCase(
                name: golden.name,
                input: Self.expand(golden.input),
                expectedText: Self.expand(golden.expectedText),
                expectedProviders: golden.expectedProviders,
                expectedQuarantineCount: golden.expectedQuarantineCount
            )
        }
    }

    // The fixture writes any run of 16+ identical characters as {{c*N}}, so no
    // token-shaped literal is committed. tests/test_secret_scrub_golden.py
    // expands it the same way.
    private static func expand(_ text: String) -> String {
        let pattern = try! NSRegularExpression(pattern: #"\{\{([A-Za-z0-9])\*(\d+)\}\}"#)
        let source = text as NSString
        var output = ""
        var cursor = 0
        for match in pattern.matches(in: text, range: NSRange(location: 0, length: source.length)) {
            output += source.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            let character = source.substring(with: match.range(at: 1))
            let count = Int(source.substring(with: match.range(at: 2))) ?? 0
            output += String(repeating: character, count: count)
            cursor = match.range.location + match.range.length
        }
        output += source.substring(from: cursor)
        return output
    }

    func testSwiftScrubberMatchesEveryGoldenCase() throws {
        let cases = try goldenCases()
        XCTAssertGreaterThan(cases.count, 30)
        for golden in cases {
            let result = SecretScrubber.scrub(golden.input)
            XCTAssertEqual(result.text, golden.expectedText, "text differs for \(golden.name)")
            XCTAssertEqual(result.providers, golden.expectedProviders, "providers differ for \(golden.name)")
            XCTAssertEqual(
                result.quarantineCount,
                golden.expectedQuarantineCount,
                "quarantine count differs for \(golden.name)"
            )
        }
    }

    func testGoldenFixtureCoversEverySwiftProviderFamily() throws {
        let covered = Set(try goldenCases().flatMap(\.expectedProviders))
        let families = Set(SecretScrubber.providerNames).union(["assignment"])
        XCTAssertTrue(families.isSubset(of: covered), "golden misses \(families.subtracting(covered).sorted())")
    }

    func testRescrubbingIsIdempotent() throws {
        for golden in try goldenCases() {
            let once = SecretScrubber.scrub(golden.input).text
            XCTAssertEqual(SecretScrubber.scrub(once).text, once, "re-scrub changed \(golden.name)")
        }
    }

    // The store path runs this on every brain_store, so it must stay fast on the
    // same adversarial shapes the Python suite pins (#960, #961).
    func testAdversarialLabelRunIsFast() {
        let text = String(String(repeating: "key-", count: 16_384).prefix(65_536))
        let started = Date()
        _ = SecretScrubber.scrub(text)
        XCTAssertLessThan(Date().timeIntervalSince(started), 2.0)
    }

    func testThousandsOfFindingsAreFast() {
        let value = "aB3dE5gH7jK9mN1pQ2rS4tU6vW8xY0zC"
        let line = "\"api_key\":\"\(value)\","
        let text = String(repeating: line, count: (1_048_576 / line.utf16.count) + 1)
        let started = Date()
        let result = SecretScrubber.scrub(text)
        XCTAssertGreaterThan(result.providers.count, 20_000)
        XCTAssertLessThan(Date().timeIntervalSince(started), 5.0)
    }
}
