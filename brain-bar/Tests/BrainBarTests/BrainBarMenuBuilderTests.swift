import AppKit
import SwiftUI
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
            (row.dividerBefore ? [.separator] : []) + { () -> [Item] in
            switch row.kind {
            case .separator: return [.separator]
            case .standardItem:
                let command = NSEvent.ModifierFlags.command.rawValue, option = NSEvent.ModifierFlags.option.rawValue
                let (key, action): (String, BrainBarMenuAction) = row.keyEquivalent == "⌘," ? (",", .settings) : ("q", .quit)
                return [.standard(title: row.strings[0], key: key, modifiers: command, action: action, alternate: false)]
                    + (row.optionAlternate.map { [.standard(title: $0, key: key, modifiers: command | option, action: .restart, alternate: true)] } ?? [])
            default: return [.hosted(words: row.strings.joined(separator: " "), action: row.action.map { .row($0.kind) })]
            }
            }()
        }
    }

    private func items(_ menu: NSMenu) -> [Item] {
        menu.items.map { item in
            if item.isSeparatorItem { return .separator }
            let action = item.isEnabled ? item.representedObject as? BrainBarMenuAction : nil
            if let host = item.view as? BrainBarMenuRowHost {  // one element per row; a row with a button is a group
                XCTAssertTrue(host.isAccessibilityElement())
                XCTAssertEqual(host.accessibilityRole(), action == nil ? .staticText : .group)
                if action != nil {  // the button is named by the value's label, not the drawn (line-breaking) text
                    XCTAssertEqual((host.hosting as? NSHostingView<BrainBarMenuRowView>)?.rootView.row.action.map(BrainBarMenuRowView.accessibilityName), item.title)
                }
                return .hosted(words: host.accessibilityLabel() ?? "", action: action)
            }
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
                if case .row(let kind)? = item.representedObject as? BrainBarMenuAction {  // type-select reaches the action by name
                    XCTAssertEqual(item.title, presentation.rows.compactMap(\.action).first { $0.kind == kind }?.label, id)
                }
            }
        }
    }

    // MARK: what the views draw, read from the off-screen 2x pixels of each row

    private static func letters(_ s: String) -> String {
        String(s.lowercased().unicodeScalars.filter(CharacterSet.alphanumerics.contains).map(Character.init))
    }

    /// The words a row draws between two x positions (points), by OCR, top to bottom. Drawn with Increase Contrast,
    /// which changes no geometry (below). The gutter is left out: its marks are checked by their pixels.
    private static func read(_ rep: NSBitmapImageRep, x: ClosedRange<CGFloat>) throws -> String {
        var image = try XCTUnwrap(rep.cgImage)
        image = try XCTUnwrap(image.cropping(to: CGRect(x: x.lowerBound * 2, y: 0, width: (x.upperBound - x.lowerBound) * 2, height: CGFloat(image.height))))
        // Rows are drawn on transparency; Vision reads that as black, so lay the row on the light material first.
        let context = try XCTUnwrap(CGContext(data: nil, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: 0,
                                              space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.setFillColor(.white)
        context.fill(CGRect(x: 0, y: 0, width: image.width, height: image.height))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        image = try XCTUnwrap(context.makeImage())
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        return letters((request.results ?? []).sorted { $0.boundingBox.maxY > $1.boundingBox.maxY }.compactMap { $0.topCandidates(1).first?.string }.joined())
    }

    /// Pixels drawn (alpha over a quarter) inside a rect given in points.
    /// Opaque near-white pixels inside a rect given in points: a disc's glyph is drawn white on the disc's colour.
    private static func white(_ rep: NSBitmapImageRep, x: ClosedRange<CGFloat>) -> Int {
        var count = 0
        for px in Int(x.lowerBound * 2)..<Int(x.upperBound * 2) {
            for py in 0..<rep.pixelsHigh {
                guard let c = rep.colorAt(x: px, y: py)?.usingColorSpace(.deviceRGB), c.alphaComponent > 0.9 else { continue }
                if min(c.redComponent, c.greenComponent, c.blueComponent) > 0.85 { count += 1 }
            }
        }
        return count
    }

    private static func ink(_ rep: NSBitmapImageRep, x: ClosedRange<CGFloat>) -> Int {
        var count = 0
        for px in Int(x.lowerBound * 2)..<Int(x.upperBound * 2) { for py in 0..<rep.pixelsHigh where (rep.colorAt(x: px, y: py)?.alphaComponent ?? 0) > 0.25 { count += 1 } }
        return count
    }

    private static func render(_ view: NSView) throws -> NSBitmapImageRep {
        view.appearance = NSAppearance(named: .accessibilityHighContrastAqua)
        view.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
        return try BrainBarMenuRenderHarness.rowBitmap(view)
    }

    private static func host(_ row: P.Row) -> NSView {
        BrainBarMenuRowHost(BrainBarMenuRowView(row: row, previous: nil, next: nil, accessibility: .init(increaseContrast: true),
                                                focus: BrainBarMenuRowFocus()) { _ in }, words: row.strings, isAction: row.action != nil)
    }

    /// Every row of a menu draws exactly its fields: its words and button label and nothing else (OCR, standing titles
    /// included), a mark or spinner in the gutter exactly when the row has one (R28b), and a bar when it has a percent.
    private func assertDrawsExactly(_ presentation: P, _ id: String) throws {
        let metrics = BrainBarMenuRenderHarness.measured()
        let builder = BrainBarMenuBuilder(accessibility: .init(increaseContrast: true))
        let menu = builder.build(presentation)
        var hosted = presentation.rows.filter { $0.kind != .separator && !($0.kind == .standardItem && $0.strings.count == 1 && $0.action == nil) }
        for item in menu.items where !item.isSeparatorItem && !item.isAlternate {
            let rep = try Self.render(BrainBarMenuRenderHarness.rowView(item, metrics: metrics))
            guard item.view != nil else {  // a standard row: its title, read left of the key equivalent
                XCTAssertEqual(try Self.read(rep, x: 0...(BrainBarMenuMetrics.width - 60)), Self.letters(item.title), "\(id): \(item.title)")
                continue
            }
            let row = hosted.removeFirst()
            XCTAssertEqual(try Self.read(rep, x: 44...BrainBarMenuMetrics.width), Self.letters((row.strings + [row.action?.label].compactMap { $0 }).joined()), "\(id): \(row.kind)")
            let marked = row.mark != nil || row.spinner || (row.state != nil && row.kind != .headlineDetail)
            XCTAssertEqual(Self.ink(rep, x: 16...42) > 20, marked, "\(id): \(row.kind) gutter")
            if row.mark != nil, row.spinner { XCTAssertGreaterThan(Self.ink(rep, x: 300...324), 20, "\(id): spinner at the line's end") }
            if row.progressPercent != nil {
                var bare = row
                bare.progressPercent = nil
                XCTAssertGreaterThan(Self.ink(rep, x: 52...324) - Self.ink(try Self.render(Self.host(bare)), x: 52...324), 1_000, "\(id): the bar")
            }
        }
        XCTAssertEqual(hosted.count, 0, id)
    }

    func testEveryCaseDrawsExactlyItsFields() throws {
        for (id, presentation, _) in try cases() { try assertDrawsExactly(presentation, id) }
    }

    /// Codex round 1 (B1–B3): fields on kinds that "do not usually have them", standing rows richer than a standard
    /// item, and a button label too long for one line. Nothing is dropped and no action is inferred.
    func testAnyFieldOnAnyKindIsDrawn() throws {
        let long = String(repeating: "Copy diagnostic report ", count: 6)
        let rows: [P.Row] = [
            .init(kind: .statusLine, strings: ["Waiting"], action: .init(kind: .copyReport, label: "Copy report")),
            .init(kind: .headline, strings: ["Halfway"], progressPercent: 50),
            .init(kind: .statusLine, strings: ["Working"], mark: .busy, spinner: true),
            .init(kind: .standardItem, strings: ["Settings…", "Account settings"], keyEquivalent: "⌘,"),
            .init(kind: .standardItem, strings: ["Settings…"], action: .init(kind: .copyReport, label: "Copy report"), keyEquivalent: "⌘,"),
            .init(kind: .needsYou, strings: ["Attention"], action: .init(kind: .copyReport, label: long)),
        ]
        let presentation = P(form: .needs, headline: "", count: 0, icon: .marked, rows: rows)
        try assertDrawsExactly(presentation, "synthetic")
        let builder = BrainBarMenuBuilder(accessibility: .init())
        let items = builder.build(presentation).items
        XCTAssertEqual(items.map { $0.view != nil }, [true, true, true, true, true, true], "rich standing rows are hosted")
        XCTAssertEqual(items.map { $0.representedObject as? BrainBarMenuAction },
                       [.row(.copyReport), nil, nil, nil, .row(.copyReport), .row(.copyReport)], "no action inferred over an explicit one")
        XCTAssertEqual(items[3].keyEquivalent, ",", "the hosted standing row keeps its key equivalent")
    }

    /// Codex round 2 (R2-B1): a mark carries its symbol, not just colour. A disc's glyph is white on the disc, so a disc
    /// without it has no white at all; a bare mark (the checkmark, the state ellipsis) is the glyph itself.
    func testEveryMarkDrawsItsSymbol() throws {
        let metrics = BrainBarMenuRenderHarness.measured()
        var seen = Set<String>()
        for (id, presentation, _) in try cases() {
            let builder = BrainBarMenuBuilder(accessibility: .init(increaseContrast: true))
            let rows = presentation.rows.filter { $0.kind != .separator && !($0.kind == .standardItem && $0.strings.count == 1 && $0.action == nil) }
            for (row, item) in zip(rows, builder.build(presentation).items.filter { $0.view != nil }) {
                let rep = try Self.render(BrainBarMenuRenderHarness.rowView(item, metrics: metrics))
                if let mark = row.mark, mark != .checkmark {
                    XCTAssertGreaterThan(Self.white(rep, x: 16...42), 12, "\(id): the \(mark) mark draws its symbol")
                    seen.insert(mark.rawValue)
                } else if row.mark == .checkmark || (row.mark == nil && !row.spinner && row.state == .cantTell && row.kind != .headlineDetail) {
                    XCTAssertGreaterThan(Self.ink(rep, x: 16...42), 20, "\(id): the bare mark is drawn")
                    seen.insert(row.mark?.rawValue ?? "ellipsis")
                }
            }
        }
        XCTAssertEqual(seen, ["ok", "busy", "wait", "marked", "alert-circle", "checkmark", "ellipsis"])
    }

    /// Codex round 2 (R2-B2): the line-breaking transform keeps every character; it only turns a space into U+00A0.
    func testLineBreakingKeepsEveryCharacter() throws {
        let edges = ["40% done", "11:28 now", "Nov 9 at 03:44", "3 things need you", "is 21", "21 22", "no number here", "",
                     "To Google Drive. Started at 11:28. 40% done, about 12:48."]
        let strings = try cases().flatMap { $0.presentation.rows.flatMap { $0.strings + [$0.action?.label].compactMap { $0 } } }
        for input in strings + edges {
            let output = BrainBarMenuRowView.keepingNumbersWithUnits(input).replacingOccurrences(of: "\u{00A0}", with: " ")
            XCTAssertEqual(Array(output.unicodeScalars), Array(input.unicodeScalars), input)
        }
        XCTAssertEqual(BrainBarMenuRowView.keepingNumbersWithUnits("is 21 hours old."), "is 21\u{00A0}hours old.")
        XCTAssertEqual(BrainBarMenuRowView.keepingNumbersWithUnits("40% done"), "40%\u{00A0}done")
        XCTAssertEqual(BrainBarMenuRowView.accessibilityName(of: .init(kind: .copyReport, label: "Open 2 folders")), "Open 2 folders")
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

    /// A string literal with a word in it (two letters in a row) is allowed only where a symbol is constructed
    /// (`systemName: "…"`) and only when it is an SF Symbol name.
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
