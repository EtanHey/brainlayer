import AppKit

/// No registration or cached value: each open re-reads the key, defaulting to the new menu.
enum BrainBarNewMenuPreference {
    static let key = "BrainBarNewMenuEnabled"
    static let defaultValue: Bool = true
    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool {
        defaults.object(forKey: key) == nil ? defaultValue : defaults.bool(forKey: key)
    }
}

/// The report is limited to presentation words and health-check diagnostics, never source contents.
enum BrainBarMenuReport {
    /// `extraLines`: warnings a surface shows beyond the presentation (the new design's runtime warnings); the classic
    /// menu passes none, so its report is unchanged.
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

/// Action targets are injected so tests exercise dispatch without activating windows, auth or process control.
@MainActor final class BrainBarLiveMenuActions: BrainBarMenuActions {
    let settings: () -> Void, storage: () -> Void, quit: () -> Void, restart: () -> Void
    let reconnect: () -> Void, copy: () -> Void, dismiss: () -> Void

    init(settings: @escaping () -> Void, storage: @escaping () -> Void,
         quit: @escaping () -> Void, restart: @escaping () -> Void,
         reconnect: @escaping () -> Void, copy: @escaping () -> Void, dismiss: @escaping () -> Void) {
        self.settings = settings; self.storage = storage; self.quit = quit; self.restart = restart
        self.reconnect = reconnect; self.copy = copy; self.dismiss = dismiss
    }

    func perform(_ action: BrainBarMenuAction) {
        dismiss()
        DispatchQueue.main.async { [self] in
            switch action {
            case .settings: settings()
            case .quit: quit()
            case .restart: restart()
            case .row(.connectDrive), .row(.reconnectDrive): reconnect()
            case .row(.openStorageSettings): storage()
            case .row(.copyReport): copy()
            }
        }
    }
}
