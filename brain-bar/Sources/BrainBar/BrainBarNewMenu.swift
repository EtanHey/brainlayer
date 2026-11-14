import AppKit

/// No registration or cached value: an absent key is false, and each open re-reads it.
enum BrainBarNewMenuPreference {
    static let key = "BrainBarNewMenuEnabled"
    static func isEnabled(in defaults: UserDefaults = .standard) -> Bool { defaults.bool(forKey: key) }
}

/// The report is limited to presentation words and health-check diagnostics, never source contents.
enum BrainBarMenuReport {
    static func text(presentation: BrainBarMenuPresentation, verdict: BrainBarHealthVerdict?,
                     version: String, now: Date) -> String {
        let timestamp = ISO8601DateFormatter().string(from: now)
        let words: [String] = presentation.rows.filter { $0.kind != .separator }.flatMap { row -> [String] in
            row.strings + (row.action.map { [$0.label] } ?? [])
        }
        let checks: [String] = verdict?.checks.map { "\($0.name): \($0.state.rawValue) — \($0.note)" } ?? ["Unavailable"]
        let report = (["BrainBar report", "Version: \(version)", "Time: \(timestamp)", "", "Menu:"]
                + words + ["", "Checks:"] + checks).joined(separator: "\n") + "\n"
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
