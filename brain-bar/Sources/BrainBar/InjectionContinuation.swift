import Foundation

/// Builds the clipboard command for "copy to continue thread" (QA #51).
///
/// Each injection burst maps to a Claude Code session; copying a resume command
/// lets the user pick that exact thread back up, mirroring the repo-golem `-c`
/// continue pattern Etan referenced.
enum InjectionContinuation {
    static func resumeCommand(sessionID: String) -> String {
        let trimmed = sessionID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "claude --continue" }
        return "claude --resume \(trimmed)"
    }
}
