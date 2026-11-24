import AppKit

/// The report is limited to presentation words and health-check diagnostics, never source contents.
enum BrainBarMenuReport {
    /// `extraLines`: runtime warnings a surface shows beyond the presentation.
    static func text(presentation: BrainBarMenuPresentation, verdict: BrainBarHealthVerdict?,
                     version: String, now: Date, extraLines: [String] = []) -> String {
        let timestamp = ISO8601DateFormatter().string(from: now)
        let words: [String] = presentation.rows.filter { $0.kind != .separator }.flatMap { row -> [String] in
            row.strings + (row.action.map { [$0.label] } ?? [])
        }
        let checks: [String] = verdict?.checks.map { "\($0.name): \($0.state.rawValue) — \($0.note)" } ?? ["Unavailable"]
        // Built in steps: one long `+` chain did not type-check in time on the CI toolchain (Swift 6.2).
        var lines: [String] = ["BrainBar report", "Version: \(version)", "Time: \(timestamp)", "", "Menu:"]
        lines += words
        if !extraLines.isEmpty {
            lines += ["", "Warnings:"]
            lines += extraLines
        }
        lines += ["", "Checks:"]
        lines += checks
        let report: String = lines.joined(separator: "\n") + "\n"
        let scrubbed = SecretScrubber.scrub(report).text
            .replacingOccurrences(of: #"/(Users|home)/[^/\s]+"#, with: "/$1/[user]", options: .regularExpression)
            .replacingOccurrences(of: #"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"#, with: "[email]", options: .regularExpression)
        return scrubUnlabelledTokens(scrubbed)
    }

    /// Reports leave the process: unlike ingestion's quarantine, remove mixed-class opaque tokens.
    private static func scrubUnlabelledTokens(_ text: String) -> String {
        let pattern = try! NSRegularExpression(pattern: #"[A-Za-z0-9_+/=-]{24,}"#)
        var result = text
        for match in pattern.matches(in: text, range: NSRange(text.startIndex..., in: text)).reversed() {
            guard let range = Range(match.range, in: result) else { continue }
            let token = String(result[range])
            let classes = [token.rangeOfCharacter(from: .lowercaseLetters) != nil,
                           token.rangeOfCharacter(from: .uppercaseLetters) != nil,
                           token.rangeOfCharacter(from: .decimalDigits) != nil,
                           token.contains { "_+/=-".contains($0) }].filter { $0 }.count
            if classes >= 2 { result.replaceSubrange(range, with: "[redacted]") }
        }
        return result
    }
}
