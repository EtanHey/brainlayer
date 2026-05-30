import Foundation

/// Builds the clipboard command for "copy to continue thread" (QA #51).
///
/// Each injection burst maps to a Claude Code conversation; copying a resume
/// command must use Claude Code's resumable JSONL UUID, not BrainLayer's
/// internal session_id.
enum InjectionContinuation {
    static func resumeCommand(
        conversationID: String,
        fallbackSessionID: String = "",
        projectPath: String = ""
    ) -> String {
        let resumableID = conversationID.trimmingCharacters(in: .whitespacesAndNewlines)
        if UUID(uuidString: resumableID) != nil {
            return withProjectPath(projectPath, command: "claude --resume \(resumableID)")
        }
        let fallback = fallbackSessionID.trimmingCharacters(in: .whitespacesAndNewlines)
        let safeCharacters = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        guard !fallback.isEmpty, fallback.unicodeScalars.allSatisfy({ safeCharacters.contains($0) }) else {
            return "claude --continue"
        }
        return withProjectPath(projectPath, command: "claude --resume \(fallback)")
    }

    static func projectPath(fromClaudeSourceFile sourceFile: String) -> String {
        let components = sourceFile.split(separator: "/").map(String.init)
        guard let projectsIndex = components.lastIndex(of: "projects"),
              components.indices.contains(projectsIndex + 1) else {
            return ""
        }
        let encoded = components[projectsIndex + 1]
        guard encoded.hasPrefix("-") else { return "" }
        return "/" + encoded.dropFirst().split(separator: "-").joined(separator: "/")
    }

    private static func withProjectPath(_ projectPath: String, command: String) -> String {
        let path = projectPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return command }
        return "cd \(shellEscaped(path)) && \(command)"
    }

    private static func shellEscaped(_ value: String) -> String {
        let safeCharacters = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "/._-"))
        if value.unicodeScalars.allSatisfy({ safeCharacters.contains($0) }) {
            return value
        }
        return "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
