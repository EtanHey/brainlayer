import AppKit
import Vision
import XCTest
@testable import BrainBar

/// Menu PR 1: the built `NSMenu` for each of the seventeen contract cases, against the cases file itself (not the
/// presenter's output), with no window on screen.
@MainActor final class BrainBarMenuBuilderTests: XCTestCase {
    typealias P = BrainBarMenuPresentation

    private enum Item: Equatable {
        case separator
        case hosted(words: String, action: BrainBarMenuAction?)
        case standard(title: String, key: String, modifiers: UInt, action: BrainBarMenuAction?, alternate: Bool)
    }

    private func cases() throws -> [(id: String, presentation: P, expected: P)] {
        try BrainBarMenuPresentationTests.loadCases().map { c in
            (c["id"] as! String, BrainBarMenuPresenter.present(try BrainBarMenuCases.capture(c["inputs"] as! [String: Any])),
             BrainBarMenuPresentationTests.expected(c))
        }
    }

    /// What the cases file says the menu holds, item by item.
    private func expectedItems(_ p: P) -> [Item] {
        p.rows.flatMap { row -> [Item] in
            let divider: [Item] = row.dividerBefore ? [.separator] : []
            switch row.kind {
            case .separator: return [.separator]
            case .standardItem:
                let command = NSEvent.ModifierFlags.command.rawValue, option = NSEvent.ModifierFlags.option.rawValue
                let (key, action): (String, BrainBarMenuAction) = row.keyEquivalent == "⌘," ? (",", .settings) : ("q", .quit)
                return [.standard(title: row.strings[0], key: key, modifiers: command, action: action, alternate: false)]
                    + (row.optionAlternate.map { [.standard(title: $0, key: key, modifiers: command | option, action: .restart, alternate: true)] } ?? [])
            default:
                return divider + [.hosted(words: (row.strings + [row.action?.label].compactMap { $0 }).joined(separator: " "),
                                          action: row.action.map { .row($0.kind) })]
            }
        }
    }

    private func items(_ menu: NSMenu) -> [Item] {
        menu.items.map { item in
            if item.isSeparatorItem { return .separator }
            let action = item.isEnabled ? item.representedObject as? BrainBarMenuAction : nil
            if let host = item.view as? BrainBarMenuRowHost { return .hosted(words: host.accessibilityLabel() ?? "", action: action) }
            return .standard(title: item.title, key: item.keyEquivalent, modifiers: item.keyEquivalentModifierMask.rawValue,
                             action: action, alternate: item.isAlternate)
        }
    }

    func testEveryCaseBuildsTheCasesFilesItemsInOrder() throws {
        let all = try cases()
        XCTAssertEqual(all.count, 17)
        for (id, presentation, expected) in all {
            let builder = BrainBarMenuBuilder(accessibility: .init())
            let menu = builder.build(presentation)
            XCTAssertEqual(items(menu), expectedItems(expected), id)
            for item in menu.items where item.view != nil {
                XCTAssertEqual(item.view?.frame.width, BrainBarMenuMetrics.width, "\(id): one width for every row and form")
                if item.isEnabled { XCTAssertGreaterThanOrEqual(item.view?.frame.height ?? 0, 24, "\(id): an action row is a full target") }
            }
        }
    }

    /// The words the menu actually draws, read back by OCR from the off-screen 2x render, in order: every string of
    /// every row and every button label, then the standing items (the key glyphs are left out; OCR misreads ⌘). Drawn
    /// with Increase Contrast, which changes no geometry (below) and spares OCR the grey secondary text.
    func testTheDrawnMenuReadsTheCasesFilesWordsInOrder() throws {
        func letters(_ s: String) -> String { String(s.lowercased().unicodeScalars.filter(CharacterSet.alphanumerics.contains).map(Character.init)) }
        let metrics = BrainBarMenuRenderHarness.measured()
        for (id, presentation, expected) in try cases() {
            let builder = BrainBarMenuBuilder(accessibility: .init(increaseContrast: true))
            let rep = try BrainBarMenuRenderHarness.snapshot(builder.build(presentation), metrics: metrics,
                                                             appearance: .accessibilityHighContrastAqua, optionHeld: false)
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = false
            try VNImageRequestHandler(cgImage: XCTUnwrap(rep.cgImage), options: [:]).perform([request])
            let read = letters((request.results ?? []).sorted { $0.boundingBox.maxY > $1.boundingBox.maxY }
                .compactMap { $0.topCandidates(1).first?.string }.joined())
            let hosted = expected.rows.filter { ![.separator, .standardItem].contains($0.kind) }
                .flatMap { $0.strings + [$0.action?.label].compactMap { $0 } }
            let standing = expected.rows.filter { $0.kind == .standardItem }.flatMap(\.strings)
            XCTAssertTrue(read.hasPrefix(letters(hosted.joined())), "\(id): drew \(read)")
            var rest = Substring(read.dropFirst(letters(hosted.joined()).count))
            for title in standing.map(letters) {
                guard let found = rest.range(of: title) else { XCTFail("\(id): \(title) not drawn after the rows"); break }
                rest = rest[found.upperBound...]
            }
        }
    }

    /// Nothing is wired: every item only records what it asks for.
    func testEveryActionOnlyRecords() throws {
        let s12 = try cases().first { $0.id == "S12" }!.presentation
        let recorder = BrainBarMenuRecordedActions()
        let builder = BrainBarMenuBuilder(actions: recorder, accessibility: .init())  // the menu holds its builder weakly
        let menu = builder.build(s12)
        for index in menu.items.indices where menu.items[index].isEnabled { menu.performActionForItem(at: index) }
        XCTAssertEqual(recorder.calls, [.row(.copyReport), .row(.reconnectDrive), .row(.copyReport), .settings, .quit, .restart])
    }

    func testOnlyAKeyboardHighlightFocusesThatRowsButton() throws {
        let s12 = try cases().first { $0.id == "S12" }!.presentation
        let builder = BrainBarMenuBuilder(accessibility: .init())
        let menu = builder.build(s12)
        let rows = menu.items.filter { $0.view != nil && $0.isEnabled }
        func focused() -> [Bool] { rows.map { builder.focusState(of: $0)?.keyboardFocused ?? false } }
        builder.highlight(rows[1], byKeyboard: true)
        XCTAssertEqual(focused(), [false, true, false])
        builder.highlight(rows[1], byKeyboard: false)
        XCTAssertEqual(focused(), [false, false, false], "a hover never shows the focus ring")
        builder.highlight(nil, byKeyboard: true)
        XCTAssertEqual(focused(), [false, false, false])
    }

    /// Reduce Motion and Increase Contrast build every case, and change no row's size: nothing shifts.
    func testAccessibilityVariantsBuildWithTheSameGeometry() throws {
        for (id, presentation, _) in try cases() {
            func heights(_ a: BrainBarMenuAccessibility) -> [CGFloat] {
                BrainBarMenuBuilder(accessibility: a).build(presentation).items.compactMap { $0.view?.frame.height }
            }
            let base = heights(.init())
            for variant in [BrainBarMenuAccessibility(reduceMotion: true), .init(increaseContrast: true), .init(reduceMotion: true, increaseContrast: true)] {
                XCTAssertEqual(heights(variant), base, "\(id) \(variant)")
            }
        }
    }

    // MARK: no words in the new view files

    static let viewFiles = ["Menu/BrainBarMenuRows.swift", "Menu/BrainBarMenuBuilder.swift"]

    /// A string literal with a word in it (two letters in a row) is allowed only when it is an SF Symbol name.
    static func wordLiterals(in source: String) -> [String] {
        let code = source.split(separator: "\n", omittingEmptySubsequences: false)
            .map { line in line.range(of: "//").map { String(line[..<$0.lowerBound]) } ?? String(line) }.joined(separator: "\n")
        let literal = try! NSRegularExpression(pattern: #""((?:[^"\\\n]|\\.)*)""#)
        let word = try! NSRegularExpression(pattern: "[A-Za-z]{2,}")
        return literal.matches(in: code, range: NSRange(code.startIndex..., in: code)).compactMap { match in
            let text = String(code[Range(match.range(at: 1), in: code)!])
            let hasWord = word.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
            return hasWord && NSImage(systemSymbolName: text, accessibilityDescription: nil) == nil ? text : nil
        }
    }

    func testTheWordCheckCatchesAWord() {
        XCTAssertEqual(Self.wordLiterals(in: #"Text("All good"); Image(systemName: "checkmark"); key("q") // "Quit""#), ["All good"])
    }

    func testNoViewFileWritesAWord() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("../../Sources/BrainBar")
        for file in Self.viewFiles {
            let source = try String(contentsOf: root.appendingPathComponent(file), encoding: .utf8)
            XCTAssertEqual(Self.wordLiterals(in: source), [], file)
        }
    }

    /// The harness reads the cases beside the sources; it must be the same file the tests bundle.
    func testTheHarnessReadsTheBundledCases() throws {
        XCTAssertEqual(try BrainBarMenuCases.digest(of: BrainBarMenuCases.sourceURL()),
                       try BrainBarMenuCases.digest(of: BrainBarMenuPresentationTests.casesURL()))
    }
}
