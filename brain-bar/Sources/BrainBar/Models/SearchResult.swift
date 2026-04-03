import Foundation

struct SearchResult: Equatable, Identifiable {
    let chunkID: String
    let score: Double
    let project: String
    let date: String
    let summary: String
    let snippet: String
    let importance: Int?
    let tags: [String]
    let contentType: String
    let sessionID: String

    var id: String { chunkID }
    var displayText: String {
        let preferred = summary.isEmpty ? snippet : summary
        let cleaned = preferred.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        if !cleaned.isEmpty {
            return cleaned
        }
        return chunkID
    }

    var compactMetadata: String {
        var parts: [String] = []
        if !project.isEmpty {
            parts.append(project)
        }
        let trimmedDate = String(date.prefix(10))
        if !trimmedDate.isEmpty {
            parts.append(trimmedDate)
        }
        if let importance {
            parts.append("imp \(importance)")
        }
        parts.append("score \(String(format: "%.2f", score))")
        return parts.joined(separator: " • ")
    }

    var tagSummary: String? {
        guard !tags.isEmpty else { return nil }
        return tags.joined(separator: ", ")
    }

    /// Relevance tier derived from BM25 score.
    var relevanceTier: String {
        if score >= 15 { return "●●●" }
        if score >= 8 { return "●●○" }
        return "●○○"
    }

    /// Human-readable source label from content_type.
    var sourceLabel: String {
        switch contentType {
        case "user_message": return "conversation"
        case "assistant_text": return "conversation"
        case "ai_code": return "code"
        case "stack_trace": return "error"
        case "decision": return "decision"
        case "idea": return "idea"
        case "journal": return "journal"
        case "bookmark": return "bookmark"
        case "note": return "note"
        case "learning": return "learning"
        case "todo": return "todo"
        default: return contentType.isEmpty ? "memory" : contentType
        }
    }

    init(
        chunkID: String,
        score: Double = 0,
        project: String = "",
        date: String = "",
        summary: String = "",
        snippet: String = "",
        importance: Int? = nil,
        tags: [String] = [],
        contentType: String = "",
        sessionID: String = ""
    ) {
        self.chunkID = chunkID
        self.score = score
        self.project = project
        self.date = date
        self.summary = summary
        self.snippet = snippet
        self.importance = importance
        self.tags = tags
        self.contentType = contentType
        self.sessionID = sessionID
    }

    init(payload: [String: Any]) {
        chunkID = payload["chunk_id"] as? String ?? payload["id"] as? String ?? ""
        if let score = payload["score"] as? Double {
            self.score = score
        } else if let score = payload["score"] as? NSNumber {
            self.score = score.doubleValue
        } else {
            self.score = 0
        }
        project = payload["project"] as? String ?? ""
        date = (payload["date"] as? String) ?? (payload["created_at"] as? String) ?? ""
        summary = payload["summary"] as? String ?? ""
        snippet = (payload["snippet"] as? String) ?? (payload["content"] as? String) ?? ""
        if let importance = payload["importance"] as? Int {
            self.importance = importance
        } else if let importance = payload["importance"] as? Double {
            self.importance = Int(importance)
        } else if let importance = payload["importance"] as? NSNumber {
            self.importance = importance.intValue
        } else {
            self.importance = nil
        }
        tags = SearchResult.decodeTags(payload["tags"])
        contentType = payload["content_type"] as? String ?? ""
        sessionID = payload["session_id"] as? String ?? ""
    }

    init(rowID: String, title: String, metadata: String, tags: [String] = []) {
        chunkID = rowID
        score = SearchResult.extractScore(from: metadata) ?? 0
        project = ""
        date = ""
        summary = title
        snippet = title
        importance = SearchResult.extractImportance(from: metadata)
        self.tags = tags
        contentType = ""
        sessionID = ""
    }

    private static func decodeTags(_ raw: Any?) -> [String] {
        if let tags = raw as? [String] {
            return tags
        }
        guard let text = raw as? String, let data = text.data(using: .utf8) else {
            return []
        }
        return (try? JSONSerialization.jsonObject(with: data) as? [String]) ?? []
    }

    private static func extractImportance(from metadata: String) -> Int? {
        guard let range = metadata.range(of: "imp ") else { return nil }
        let digits = metadata[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }

    private static func extractScore(from metadata: String) -> Double? {
        guard let range = metadata.range(of: "score ") else { return nil }
        let token = metadata[range.upperBound...].split(separator: " ").first ?? ""
        return Double(token)
    }
}
