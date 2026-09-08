import Foundation

/// Rebuilds the human-readable conversation from the immutable transcript pointer
/// stored on a chunk. Database context remains an explicit fallback when the source
/// was moved, archived, or written by a provider shape we cannot parse safely.
enum SourceConversationReader {
    private struct ParsedTurn {
        let lineNumber: Int
        let content: String
        let sender: String
        let createdAt: String
        let attributionAgent: String
        let sessionID: String
        let sourceEndOffset: Int64
    }

    static func reconstruct(
        target: BrainDatabase.ConversationChunk,
        fileManager: FileManager = .default
    ) -> BrainDatabase.ExpandedConversation? {
        guard let sourceURL = resolvedSourceURL(
            target.sourceFile,
            fileManager: fileManager
        ),
        let data = try? Data(contentsOf: sourceURL) else {
            return nil
        }

        let turns = parsedTurns(from: data)
        guard !turns.isEmpty else { return nil }

        let targetIndex = bestTargetIndex(for: target, in: turns)
        let entries = turns.enumerated().map { index, turn in
            BrainDatabase.ConversationChunk(
                chunkID: "\(sourceURL.path):\(turn.lineNumber)",
                content: turn.content,
                contentType: turn.sender == "user" ? "user_message" : "assistant_text",
                sender: turn.sender,
                importance: index == targetIndex ? target.importance : 0,
                createdAt: turn.createdAt,
                summary: "",
                isTarget: index == targetIndex,
                sourceFile: sourceURL.path,
                tags: index == targetIndex ? target.tags : []
            )
        }
        let renderedTarget = targetIndex.map { entries[$0] } ?? target
        let storingAgent = storingAgent(
            targetIndex: targetIndex,
            turns: turns,
            entries: entries
        )

        return BrainDatabase.ExpandedConversation(
            target: renderedTarget,
            entries: entries,
            sourceFile: sourceURL.path,
            storingAgent: storingAgent,
            origin: .sourceJSONL
        )
    }

    private static func parsedTurns(from data: Data) -> [ParsedTurn] {
        let lines = data.split(separator: 0x0A, omittingEmptySubsequences: false)
        var sourceEndOffset: Int64 = 0
        return lines.enumerated().compactMap { index, line in
            sourceEndOffset += Int64(line.count)
            if index < lines.count - 1 {
                sourceEndOffset += 1
            }
            return parsedTurn(
                data: Data(line),
                lineNumber: index + 1,
                sourceEndOffset: sourceEndOffset
            )
        }
    }

    private static func resolvedSourceURL(
        _ sourceFile: String,
        fileManager: FileManager
    ) -> URL? {
        let trimmed = sourceFile.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let direct = URL(fileURLWithPath: trimmed).standardizedFileURL
        if fileManager.isReadableFile(atPath: direct.path) {
            return direct
        }

        let basename = direct.lastPathComponent
        guard basename.hasSuffix(".jsonl") else { return nil }
        let home = fileManager.homeDirectoryForCurrentUser
        let roots = [
            home.appendingPathComponent(".claude/projects", isDirectory: true),
            home.appendingPathComponent(".claude-archive", isDirectory: true),
            home.appendingPathComponent(".codex/sessions", isDirectory: true),
        ]
        for root in roots where fileManager.fileExists(atPath: root.path) {
            guard let enumerator = fileManager.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsPackageDescendants, .skipsHiddenFiles]
            ) else { continue }
            for case let candidate as URL in enumerator where candidate.lastPathComponent == basename {
                if fileManager.isReadableFile(atPath: candidate.path) {
                    return candidate
                }
            }
        }
        return nil
    }

    private static func parsedTurn(
        data: Data,
        lineNumber: Int,
        sourceEndOffset: Int64
    ) -> ParsedTurn? {
        guard let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }

        let payload = raw["payload"] as? [String: Any]
        let providerMessage: [String: Any]
        if raw["type"] as? String == "response_item",
           payload?["type"] as? String == "message" {
            providerMessage = payload ?? [:]
        } else {
            providerMessage = raw
        }
        let nestedMessage = providerMessage["message"] as? [String: Any]
        let role = normalizedRole(
            nestedMessage?["role"] ?? providerMessage["role"] ?? providerMessage["type"]
        )
        guard role == "user" || role == "assistant" else { return nil }

        let contentValue = nestedMessage?["content"]
            ?? providerMessage["content"]
            ?? providerMessage["text"]
            ?? providerMessage["message"]
        let content = textContent(from: contentValue)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else { return nil }

        return ParsedTurn(
            lineNumber: lineNumber,
            content: content,
            sender: role,
            createdAt: normalizedString(raw["timestamp"] ?? providerMessage["timestamp"]),
            attributionAgent: normalizedString(
                raw["attributionAgent"]
                    ?? providerMessage["attributionAgent"]
                    ?? payload?["attributionAgent"]
            ),
            sessionID: normalizedString(
                raw["sessionId"]
                    ?? providerMessage["sessionId"]
                    ?? payload?["sessionId"]
            ),
            sourceEndOffset: sourceEndOffset
        )
    }

    private static func normalizedRole(_ value: Any?) -> String {
        let role = normalizedString(value).lowercased()
        switch role {
        case "model", "ai", "bot": return "assistant"
        default: return role
        }
    }

    private static func textContent(from value: Any?) -> String {
        if let text = value as? String {
            return text
        }
        if let values = value as? [Any] {
            return values.compactMap { item -> String? in
                guard let block = item as? [String: Any] else {
                    return item as? String
                }
                let type = normalizedString(block["type"]).lowercased()
                guard type.isEmpty || ["text", "input_text", "output_text"].contains(type) else {
                    return nil
                }
                let text = normalizedString(block["text"] ?? block["content"])
                return text.isEmpty ? nil : text
            }.joined(separator: "\n\n")
        }
        if let object = value as? [String: Any] {
            return textContent(from: object["content"] ?? object["text"])
        }
        return ""
    }

    private static func bestTargetIndex(
        for target: BrainDatabase.ConversationChunk,
        in turns: [ParsedTurn]
    ) -> Int? {
        if let sourceEndOffset = target.sourceEndOffset,
           let exactIndex = turns.firstIndex(where: { $0.sourceEndOffset == sourceEndOffset }) {
            return exactIndex
        }

        let needle = normalizedContent(target.content)
        guard !needle.isEmpty else { return nil }
        var best: (index: Int, score: Int)?
        for (index, turn) in turns.enumerated() {
            let candidate = normalizedContent(turn.content)
            let contentScore: Int
            if candidate == needle {
                contentScore = 4
            } else if candidate.contains(needle) || needle.contains(candidate) {
                contentScore = 2
            } else {
                continue
            }
            let senderScore = turn.sender == target.sender.lowercased() ? 1 : 0
            let score = contentScore + senderScore
            if best == nil || score > best!.score {
                best = (index, score)
            }
        }
        return best?.index
    }

    private static func storingAgent(
        targetIndex: Int?,
        turns: [ParsedTurn],
        entries: [BrainDatabase.ConversationChunk]
    ) -> String {
        if let targetIndex {
            let targetAgent = turns[targetIndex].attributionAgent
            if !targetAgent.isEmpty { return targetAgent }
        }

        let contractPattern = #"cmuxlayer contract for ([A-Za-z0-9._-]+):"#
        if let regex = try? NSRegularExpression(pattern: contractPattern) {
            for entry in entries where entry.sender == "user" {
                let range = NSRange(entry.content.startIndex..., in: entry.content)
                guard let match = regex.firstMatch(in: entry.content, range: range),
                      let capture = Range(match.range(at: 1), in: entry.content) else { continue }
                return String(entry.content[capture])
            }
        }

        if let sessionID = turns.lazy.map(\.sessionID).first(where: { !$0.isEmpty }),
           let identity = InjectionRecipientIdentity.resolve(sessionID: sessionID),
           !identity.agentName.isEmpty {
            return identity.agentName
        }
        return turns.lazy.map(\.attributionAgent).first(where: { !$0.isEmpty }) ?? ""
    }

    private static func normalizedString(_ value: Any?) -> String {
        (value as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    private static func normalizedContent(_ content: String) -> String {
        content
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}
