import AppKit
import XCTest
@testable import BrainBar

@MainActor final class BrainBarMenuBuilderTests: XCTestCase {
    static func wordLiterals(in source: String) -> [String] {
        let code = source.split(separator: "\n", omittingEmptySubsequences: false)
            .map { line in line.range(of: "//").map { String(line[..<$0.lowerBound]) } ?? String(line) }.joined(separator: "\n")
        let literal = try! NSRegularExpression(pattern: #""((?:[^"\\\n]|\\.)*)""#)
        let word = try! NSRegularExpression(pattern: "[A-Za-z]{2,}")
        return literal.matches(in: code, range: NSRange(code.startIndex..., in: code)).compactMap { match in
            let text = String(code[Range(match.range(at: 1), in: code)!])
            let hasWord = word.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
            let symbol = code[..<Range(match.range, in: code)!.lowerBound].hasSuffix("systemName: ")
                && NSImage(systemSymbolName: text, accessibilityDescription: nil) != nil
            return hasWord && !symbol ? text : nil
        }
    }

    func testTheWordCheckCatchesAWord() {
        XCTAssertEqual(Self.wordLiterals(in: #"Text("All good"); Image(systemName: "checkmark"); Text("checkmark"); key("q") // "Quit""#),
                       ["All good", "checkmark"])
    }
}
