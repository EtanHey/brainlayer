import Foundation

/// Builds the clipboard command for "copy to continue thread" (QA #51).
///
/// Each injection burst maps to a Claude Code conversation; copying a resume
/// command must use Claude Code's resumable JSONL UUID, not BrainLayer's
/// internal session_id.
enum InjectionContinuation {
    static func resumeCommand(conversationID: String, fallbackSessionID: String = "") -> String {
        let resumableID = conversationID.trimmingCharacters(in: .whitespacesAndNewlines)
        if !resumableID.isEmpty {
            return "claude --resume \(resumableID)"
        }
        let fallback = fallbackSessionID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !fallback.isEmpty else { return "claude --continue" }
        return "claude --resume \(fallback)"
    }
}
