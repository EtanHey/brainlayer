import Foundation

/// Swift port of `src/brainlayer/pipeline/secret_scrub.py`, so BrainBar's store path
/// never writes a raw secret to the chunks row, its FTS copies or the pending queue.
///
/// Parity contract: `tests/fixtures/secret_scrub/golden.json` is asserted by BOTH
/// `tests/test_secret_scrub_golden.py` and `SecretScrubberGoldenTests`. Change a rule
/// here and in Python together, and regenerate the fixture from Python.
///
/// Two deliberate differences from Python, neither of which changes output on the
/// fixture: offsets are UTF-16 code units (Python uses code points), and the scan
/// window is measured in UTF-16 units. Both only differ for astral characters.
enum SecretScrubber {
    struct Redaction: Equatable, Sendable {
        let provider: String
        let start: Int
        let end: Int
    }

    struct Result: Sendable {
        let text: String
        let redactions: [Redaction]
        let quarantineCount: Int

        var providers: [String] { redactions.map(\.provider) }
    }

    private struct Span {
        let start: Int
        let end: Int
        let provider: String
        let order: Int
    }

    static let maxScanUnits = 128 * 1024
    static let windowOverlapUnits = 512
    static let minEntropyTokenLength = 24
    static let entropyThreshold = 4.0

    // Same patterns, same order as _PROVIDER_PATTERNS: on an identical span the
    // earlier provider wins (e.g. anthropic over openai for sk-ant-…).
    private static let providerPatternSources: [(String, String)] = [
        ("anthropic", #"\bsk-ant-[A-Za-z0-9_-]{20,}\b"#),
        ("stripe", #"\b(?:[sr]k_(?:live|test)|whsec)_[A-Za-z0-9]{16,}\b"#),
        ("openai", #"\bsk-(?:proj-|svcacct-|admin-|org-)?[A-Za-z0-9_-]{20,}\b"#),
        ("aws", #"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"#),
        ("github", #"\b(?:gh[opusr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}_[A-Za-z0-9_]{20,})\b"#),
        ("slack", #"\bxox[baprs]-(?:[A-Za-z0-9]+-){1,}[A-Za-z0-9]{16,}\b"#),
        ("google", #"\bAIza[A-Za-z0-9_-]{32,}\b"#),
        ("gitlab", #"\bglpat-[A-Za-z0-9_-]{20,}\b"#),
        ("supabase", #"\b(?:sbp_[A-Za-z0-9]{20,}|sb_secret_[A-Za-z0-9_-]{20,})\b"#),
        ("sendgrid", #"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{32,}\b"#),
        ("groq", #"\bgsk_[A-Za-z0-9]{40,}\b"#),
        ("tailscale", #"\btskey-[a-z]+-[A-Za-z0-9]{6,}-[A-Za-z0-9]{16,}\b"#),
        ("vercel", #"\bvc[kpi]_[A-Za-z0-9]{20,}\b"#),
    ]

    static var providerNames: [String] { providerPatternSources.map(\.0) }

    // ── Python's regex semantics, spelled out (#962 review B1) ──────────────
    // ICU and Python disagree at Unicode edges, and a disagreement here stores a
    // raw secret. So nothing below relies on ICU's own \b, \s or case folding;
    // each is written as the exact set Python uses. The sets were enumerated
    // over every code point with Python's `re` (see the #962 PR body):
    // - Python \w is exactly [\p{L}\p{N}_]. ICU's \b also counts marks,
    //   Join_Control (ZWJ/ZWNJ) and connector punctuation as word characters,
    //   so a token followed by U+0301 did not end on a boundary.
    // - Python \s is these 29 code points; ICU's \s lacks U+001C-U+001F.
    // - Under IGNORECASE, Python's [A-Za-z] also matches U+0130, U+0131,
    //   U+017F and U+212A, the literal i also matches U+0130/U+0131, k matches
    //   U+212A and s matches U+017F. ICU case folding differs (it can also match
    //   "ss" against U+00DF), so the label pattern is case-SENSITIVE with every
    //   variant listed.
    private static let pythonWordClass = #"[\p{L}\p{N}_]"#
    private static let pythonWordBoundary =
        "(?:(?<=\(pythonWordClass))(?!\(pythonWordClass))|(?<!\(pythonWordClass))(?=\(pythonWordClass)))"
    private static let pythonWhitespace =
        #"[\t\n\x{0B}\x{0C}\r\x{1C}-\x{20}\x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]"#
    private static let foldedLetterExtras = #"\x{130}\x{131}\x{17F}\x{212A}"#
    private static let labelChar = "[A-Za-z0-9_.\\-\(foldedLetterExtras)]"
    private static let labelEndChar = "[A-Za-z0-9_\(foldedLetterExtras)]"
    private static let valueChar = "[A-Za-z0-9_./+=:\\-\(foldedLetterExtras)]"

    /// One keyword, case-insensitive the way Python's re.IGNORECASE is.
    private static func pythonCaseless(_ word: String) -> String {
        word.map { letter -> String in
            switch letter {
            case "i": return #"[Ii\x{130}\x{131}]"#
            case "k": return #"[Kk\x{212A}]"#
            case "s": return #"[Ss\x{17F}]"#
            default: return "[\(letter.uppercased())\(letter)]"
            }
        }.joined()
    }

    private static func withPythonWordBoundaries(_ pattern: String) -> String {
        pattern.replacingOccurrences(of: #"\b"#, with: pythonWordBoundary)
    }

    // Constant patterns: a compile failure is a programming error that the golden
    // test catches, so force-try is the honest shape here.
    nonisolated(unsafe) private static let providerPatterns: [(String, NSRegularExpression)] =
        providerPatternSources.map { ($0.0, try! NSRegularExpression(pattern: withPythonWordBoundaries($0.1))) }

    // _SECRET_LABEL_RE, post #960: the label is one whole run of label characters
    // that contains a keyword and ends on a word character; an optional quote after
    // it accepts JSON / dict keys.
    nonisolated(unsafe) private static let labelPattern: NSRegularExpression = {
        let keywords = ["key", "token", "secret", "password", "api", "auth", "access"]
            .map(pythonCaseless).joined(separator: "|")
        let pattern = "(?<!\(labelChar))(?=\(labelChar)*?(?:\(keywords)))"
            + "(?<label>\(labelChar)*\(labelEndChar))(?<labelquote>[\"']?)"
            + "\(pythonWhitespace)*[=:]\(pythonWhitespace)*(?<quote>[\"']?)"
            + "(?<value>\(valueChar){24,})\\k<quote>"
        return try! NSRegularExpression(pattern: pattern)
    }()
    nonisolated(unsafe) private static let tokenPattern = try! NSRegularExpression(
        pattern: withPythonWordBoundaries(#"\b[A-Za-z0-9_./+=:-]{24,}\b"#)
    )
    nonisolated(unsafe) private static let hexFullPattern = try! NSRegularExpression(
        pattern: #"^[0-9a-fA-F]{16,}$"#
    )
    nonisolated(unsafe) private static let uuidFullPattern = try! NSRegularExpression(
        pattern: #"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"#
    )

    static func scrub(_ text: String) -> Result {
        guard !text.isEmpty else { return Result(text: text, redactions: [], quarantineCount: 0) }
        let full = text as NSString
        let windows = scanWindows(length: full.length)

        var spans: [Span] = []
        var order = 0
        for window in windows {
            let slice = full.substring(with: NSRange(location: window.start, length: window.end - window.start)) as NSString
            for (provider, pattern) in providerPatterns {
                for match in pattern.matches(in: slice as String, range: NSRange(location: 0, length: slice.length)) {
                    spans.append(Span(start: window.start + match.range.location,
                                      end: window.start + match.range.location + match.range.length,
                                      provider: provider, order: order))
                    order += 1
                }
            }
        }
        for window in windows {
            let slice = full.substring(with: NSRange(location: window.start, length: window.end - window.start)) as NSString
            for span in assignmentSpans(in: slice, offset: window.start, existing: spans) {
                spans.append(Span(start: span.start, end: span.end, provider: span.provider, order: order))
                order += 1
            }
        }
        let kept = withoutOverlaps(spans)

        var quarantine: [(start: Int, end: Int, order: Int)] = []
        for window in windows {
            let slice = full.substring(with: NSRange(location: window.start, length: window.end - window.start)) as NSString
            for token in quarantinedTokens(in: slice, offset: window.start, redactions: kept) {
                quarantine.append((token.start, token.end, quarantine.count))
            }
        }
        let quarantineCount = withoutDuplicateQuarantine(quarantine)

        return Result(
            text: kept.isEmpty ? text : applyRedactions(full, kept),
            redactions: kept.map { Redaction(provider: $0.provider, start: $0.start, end: $0.end) },
            quarantineCount: quarantineCount
        )
    }

    /// Scrubs text bound for storage and returns the metadata keys the Python
    /// watcher writes: `secret_scrub_redactions` (sorted providers, merged with any
    /// already present) and `secret_scrub_quarantine_count`.
    static func scrubForStorage(
        content: String,
        tags: [String],
        metadata: [String: Any]
    ) -> (content: String, tags: [String], metadata: [String: Any]) {
        let result = scrub(content)
        var providers = Set(result.providers)
        let scrubbedTags = tags.map { tag -> String in
            let tagResult = scrub(tag)
            providers.formUnion(tagResult.providers)
            return tagResult.text
        }
        return (
            result.text,
            scrubbedTags,
            mergeScrubMetadata(metadata, providers: providers, quarantineCount: result.quarantineCount)
        )
    }

    static func mergeScrubMetadata(
        _ metadata: [String: Any],
        providers: Set<String>,
        quarantineCount: Int
    ) -> [String: Any] {
        var merged = metadata
        if !providers.isEmpty {
            let existing = (merged["secret_scrub_redactions"] as? [String]) ?? []
            merged["secret_scrub_redactions"] = Array(providers.union(existing)).sorted()
        }
        if quarantineCount > 0 {
            merged["secret_scrub_quarantine_count"] = quarantineCount
        }
        return merged
    }

    // MARK: - Port of the Python helpers

    private static func scanWindows(length: Int) -> [(start: Int, end: Int)] {
        guard length > maxScanUnits else { return [(0, length)] }
        var windows: [(start: Int, end: Int)] = []
        var start = 0
        while start < length {
            let end = min(length, start + maxScanUnits)
            windows.append((start, end))
            if end == length { break }
            start = max(end - windowOverlapUnits, start + 1)
        }
        return windows
    }

    private static func assignmentSpans(in slice: NSString, offset: Int, existing: [Span]) -> [Span] {
        let index = SpanIndex(existing.map { ($0.start, $0.end) })
        var found: [Span] = []
        var position = 0
        let length = slice.length
        while position <= length,
              let match = labelPattern.firstMatch(
                  in: slice as String,
                  options: [.withTransparentBounds],
                  range: NSRange(location: position, length: length - position)
              ) {
            let valueRange = match.range(withName: "value")
            let rawValue = slice.substring(with: valueRange)
            let value = rstrip(rawValue, ".,;)")
            let valueStart = offset + valueRange.location
            let valueEnd = valueStart + (value as NSString).length
            let quoted = match.range(withName: "labelquote").length > 0

            // A quoted label always resumes at its value (#960 round 1): the old
            // rule never matched it, so it could start its own match inside.
            position = quoted ? valueRange.location : match.range.location + match.range.length
            if index.overlaps(valueStart, valueEnd) || isJoinKeyLike(value)
                || looksLikePathOrURL(value) || !isHighEntropy(value) {
                if !quoted, looksLikePathOrURL(value) {
                    let lastSeparator = max(
                        (rawValue as NSString).range(of: "/", options: .backwards).location.notFoundAsMinusOne,
                        (rawValue as NSString).range(of: "\\", options: .backwards).location.notFoundAsMinusOne
                    )
                    position = valueRange.location + lastSeparator + 1
                }
                continue
            }
            found.append(Span(start: valueStart, end: valueEnd, provider: "assignment", order: 0))
        }
        return found
    }

    private static func quarantinedTokens(
        in slice: NSString,
        offset: Int,
        redactions: [Span]
    ) -> [(start: Int, end: Int)] {
        let index = SpanIndex(redactions.map { ($0.start, $0.end) })
        var out: [(start: Int, end: Int)] = []
        for match in tokenPattern.matches(in: slice as String, range: NSRange(location: 0, length: slice.length)) {
            // Python: value = match.group(0).strip(".,;)"); start = match.start();
            // end = start + len(value). The start is not advanced past stripped
            // leading characters there, so it is not here either.
            let value = strip(slice.substring(with: match.range), ".,;)")
            let start = offset + match.range.location
            let end = start + (value as NSString).length
            if index.overlaps(start, end) { continue }
            if isJoinKeyLike(value) || looksLikePathOrURL(value) { continue }
            if isHighEntropy(value) { out.append((start, end)) }
        }
        return out
    }

    /// Spans sorted by (start, end, insertion order); keep the first of any
    /// overlapping group. No span is empty, so "overlaps a kept span" is exactly
    /// "starts before the furthest kept end" (same argument as #961).
    private static func withoutOverlaps(_ spans: [Span]) -> [Span] {
        let sorted = spans.sorted { ($0.start, $0.end, $0.order) < ($1.start, $1.end, $1.order) }
        var kept: [Span] = []
        var furthestEnd = -1
        for span in sorted where span.start >= furthestEnd {
            kept.append(span)
            furthestEnd = max(furthestEnd, span.end)
        }
        return kept
    }

    private static func withoutDuplicateQuarantine(_ tokens: [(start: Int, end: Int, order: Int)]) -> Int {
        let sorted = tokens.sorted { ($0.start, $0.end, $0.order) < ($1.start, $1.end, $1.order) }
        var count = 0
        var furthestEnd = -1
        for token in sorted where token.start >= furthestEnd {
            count += 1
            furthestEnd = max(furthestEnd, token.end)
        }
        return count
    }

    private static func applyRedactions(_ text: NSString, _ spans: [Span]) -> String {
        var parts: [String] = []
        var cursor = 0
        for span in spans {
            parts.append(text.substring(with: NSRange(location: cursor, length: span.start - cursor)))
            parts.append("[REDACTED:\(span.provider)]")
            cursor = span.end
        }
        parts.append(text.substring(from: cursor))
        return parts.joined()
    }

    private static func isJoinKeyLike(_ value: String) -> Bool {
        let range = NSRange(location: 0, length: (value as NSString).length)
        return uuidFullPattern.firstMatch(in: value, range: range) != nil
            || hexFullPattern.firstMatch(in: value, range: range) != nil
    }

    private static func looksLikePathOrURL(_ value: String) -> Bool {
        value.contains("/") || value.contains("\\") || value.contains("://")
    }

    private static func isHighEntropy(_ value: String) -> Bool {
        let units = Array(value.utf16)
        guard units.count >= minEntropyTokenLength else { return false }
        var counts: [UInt16: Int] = [:]
        for unit in units { counts[unit, default: 0] += 1 }
        let length = Double(units.count)
        let entropy = -counts.values.reduce(0.0) { sum, count in
            let p = Double(count) / length
            return sum + p * log2(p)
        }
        return entropy >= entropyThreshold
    }

    private static func rstrip(_ value: String, _ characters: String) -> String {
        var scalars = Substring(value)
        while let last = scalars.last, characters.contains(last) { scalars.removeLast() }
        return String(scalars)
    }

    private static func strip(_ value: String, _ characters: String) -> String {
        var scalars = Substring(value)
        while let first = scalars.first, characters.contains(first) { scalars.removeFirst() }
        while let last = scalars.last, characters.contains(last) { scalars.removeLast() }
        return String(scalars)
    }

    /// "Does [start, end) overlap any of these spans?" in O(log n): starts sorted,
    /// prefix max of ends, bisect. Same predicate as Python's _SpanIndex.
    private struct SpanIndex {
        private let starts: [Int]
        private let prefixMaxEnd: [Int]

        init(_ spans: [(Int, Int)]) {
            let ordered = spans.sorted { $0.0 < $1.0 }
            starts = ordered.map(\.0)
            var running = Int.min
            prefixMaxEnd = ordered.map { span in
                running = max(running, span.1)
                return running
            }
        }

        func overlaps(_ start: Int, _ end: Int) -> Bool {
            var low = 0
            var high = starts.count
            while low < high {
                let mid = (low + high) / 2
                if starts[mid] < end { low = mid + 1 } else { high = mid }
            }
            return low > 0 && prefixMaxEnd[low - 1] > start
        }
    }
}

private extension Int {
    var notFoundAsMinusOne: Int { self == NSNotFound ? -1 : self }
}
