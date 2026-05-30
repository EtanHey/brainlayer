import Foundation

struct InjectionChunk: Equatable, Sendable, Identifiable {
    let id: String
    let content: String
    let summary: String
    let source: String
    let sourceFile: String
    let tags: [String]
    let contentType: String
    let claudeConversationID: String

    var displayText: String {
        let preferred = summary.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? content
            : summary
        return Self.elide(preferred, limit: 80)
    }

    var kind: InjectionKind {
        InjectionKind.classify(source: source, sourceFile: sourceFile, tags: tags, content: content)
    }

    var claudeProjectPath: String {
        InjectionContinuation.projectPath(fromClaudeSourceFile: sourceFile)
    }

    init(
        id: String,
        content: String,
        summary: String,
        source: String,
        sourceFile: String,
        tags: [String],
        contentType: String,
        claudeConversationID: String = ""
    ) {
        self.id = id
        self.content = content
        self.summary = summary
        self.source = source
        self.sourceFile = sourceFile
        self.tags = tags
        self.contentType = contentType
        self.claudeConversationID = claudeConversationID
    }

    init(row: [String: Any]) {
        id = row["id"] as? String ?? ""
        content = row["content"] as? String ?? ""
        summary = row["summary"] as? String ?? ""
        source = row["source"] as? String ?? ""
        sourceFile = row["source_file"] as? String ?? ""
        contentType = row["content_type"] as? String ?? ""
        claudeConversationID = row["claude_conversation_id"] as? String ?? ""
        tags = InjectionChunk.decodeTags(row["tags"])
    }

    static func elide(_ text: String, limit: Int) -> String {
        let collapsed = text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
        guard collapsed.count > limit else { return collapsed }
        return "\(collapsed.prefix(max(limit - 1, 1)))…"
    }

    private static func decodeTags(_ value: Any?) -> [String] {
        if let tags = value as? [String] {
            return tags
        }
        guard let text = value as? String,
              let data = text.data(using: .utf8),
              let decoded = try? JSONSerialization.jsonObject(with: data) as? [String] else {
            return []
        }
        return decoded
    }
}

enum InjectionKind: String, CaseIterable, Equatable, Sendable {
    case memoryCheckpoint
    case realtimeCapture
    case storedMemory
    case dailyDigest
    case videoKnowledge
    case chat
    case toolSession
    case quickCapture
    case checkpoint
    case other

    var glyph: String {
        switch self {
        case .memoryCheckpoint: return "🧠"
        case .realtimeCapture: return "💬"
        case .storedMemory: return "📝"
        case .dailyDigest: return "🌅"
        case .videoKnowledge: return "🎬"
        case .chat: return "📱"
        case .toolSession: return "🛠"
        case .quickCapture: return "⚡"
        case .checkpoint: return "🏷"
        case .other: return "📄"
        }
    }

    var label: String {
        switch self {
        case .memoryCheckpoint: return "Memory Checkpoint"
        case .realtimeCapture: return "Realtime Capture"
        case .storedMemory: return "Stored Memory"
        case .dailyDigest: return "Daily Digest"
        case .videoKnowledge: return "Video Knowledge"
        case .chat: return "Chat"
        case .toolSession: return "Tool Session"
        case .quickCapture: return "Quick Capture"
        case .checkpoint: return "Checkpoint"
        case .other: return "Other"
        }
    }

    var modalTitle: String {
        switch self {
        case .memoryCheckpoint, .checkpoint:
            return "Memory Checkpoint"
        default:
            return label
        }
    }

    var paletteIndex: Int {
        Self.allCases.firstIndex(of: self) ?? 0
    }

    static func classify(source: String, sourceFile: String, tags: [String], content: String) -> InjectionKind {
        let normalizedSource = source.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let normalizedFile = sourceFile.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let normalizedTags = tags.map { $0.lowercased() }
        let normalizedContent = content.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        if normalizedTags.contains(where: { $0.contains("pr-merge") }) ||
            normalizedContent.hasPrefix("[checkpoint]") {
            return .checkpoint
        }

        if normalizedSource == "precompact-hook" || normalizedFile.hasPrefix("precompact:") {
            return .memoryCheckpoint
        }
        if normalizedSource == "realtime_watcher" || normalizedSource == "claude_code" {
            return .realtimeCapture
        }
        if normalizedSource == "mcp" || normalizedSource == "brain_store" || normalizedSource == "manual" {
            return .storedMemory
        }
        if normalizedSource == "digest" {
            return .dailyDigest
        }
        if normalizedSource == "youtube" {
            return .videoKnowledge
        }
        if normalizedSource == "whatsapp" {
            return .chat
        }
        if ["cursor", "codex_cli", "codex"].contains(normalizedSource) {
            return .toolSession
        }
        if normalizedSource == "quick-capture" {
            return .quickCapture
        }
        return .other
    }
}

enum InjectionTypeFilter: String, CaseIterable, Equatable, Sendable {
    case all
    case memory
    case stored
    case realtime
    case checkpoint
    case video
    case chat
    case tool

    var label: String {
        switch self {
        case .all: return "All"
        case .memory: return "Memory"
        case .stored: return "Stored"
        case .realtime: return "Realtime"
        case .checkpoint: return "Checkpoint"
        case .video: return "Video"
        case .chat: return "Chat"
        case .tool: return "Tool"
        }
    }

    func contains(_ kind: InjectionKind) -> Bool {
        switch self {
        case .all:
            return true
        case .memory:
            return kind == .memoryCheckpoint || kind == .dailyDigest
        case .stored:
            return kind == .storedMemory || kind == .quickCapture
        case .realtime:
            return kind == .realtimeCapture
        case .checkpoint:
            return kind == .checkpoint || kind == .memoryCheckpoint
        case .video:
            return kind == .videoKnowledge
        case .chat:
            return kind == .chat
        case .tool:
            return kind == .toolSession
        }
    }
}

struct InjectionEvent: Equatable, Identifiable, Sendable {
    let id: Int64
    let sessionID: String
    let timestamp: String
    let query: String
    let chunkIDs: [String]
    let tokenCount: Int
    let chunks: [InjectionChunk]
    let claudeConversationID: String

    var chunkCount: Int { chunkIDs.count }

    var uniqueChunkIDs: [String] {
        var seen = Set<String>()
        return chunkIDs.filter { seen.insert($0).inserted }
    }

    var primaryChunk: InjectionChunk? {
        guard let firstChunkID = uniqueChunkIDs.first else { return chunks.first }
        return chunks.first { $0.id == firstChunkID }
    }

    var primaryKind: InjectionKind {
        primaryChunk?.kind ?? .other
    }

    var allKinds: [InjectionKind] {
        let kinds = chunks.map(\.kind)
        return kinds.isEmpty ? [.other] : kinds
    }

    var claudeProjectPath: String {
        chunks.lazy.map(\.claudeProjectPath).first { !$0.isEmpty } ?? ""
    }

    func matches(typeFilter: InjectionTypeFilter) -> Bool {
        if chunks.isEmpty, !chunkIDs.isEmpty, typeFilter != .all {
            return true
        }
        return allKinds.contains { typeFilter.contains($0) }
    }

    var displayTitle: String {
        if let chunk = primaryChunk, !chunk.displayText.isEmpty {
            return chunk.displayText
        }
        return InjectionChunk.elide(query, limit: 80)
    }

    var expandedRowKindLabel: String? {
        let label = primaryKind.label
        return Self.normalizedForDedupe(label) == Self.normalizedForDedupe(displayTitle) ? nil : label
    }

    var triggeredByText: String {
        "Triggered by: \(InjectionChunk.elide(query, limit: 96))"
    }

    var expandedRowTriggeredByText: String? {
        Self.normalizedForDedupe(query) == Self.normalizedForDedupe(displayTitle) ? nil : triggeredByText
    }

    var modalTitle: String {
        primaryKind.modalTitle
    }

    func openingModalTitle(forChunkID chunkID: String) -> String {
        chunks.first { $0.id == chunkID }?.kind.modalTitle ?? "Conversation"
    }

    var summaryLine: String {
        "\(query) • \(chunkCount) chunks • \(tokenCount) tok"
    }

    private static func normalizedForDedupe(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
            .lowercased()
    }

    init(
        id: Int64,
        sessionID: String,
        timestamp: String,
        query: String,
        chunkIDs: [String],
        tokenCount: Int,
        chunks: [InjectionChunk] = [],
        claudeConversationID: String = ""
    ) {
        self.id = id
        self.sessionID = sessionID
        self.timestamp = timestamp
        self.query = query
        self.chunkIDs = chunkIDs
        self.tokenCount = tokenCount
        self.chunks = chunks
        self.claudeConversationID = claudeConversationID
    }

    init(row: [String: Any]) throws {
        if let intID = row["id"] as? Int {
            id = Int64(intID)
        } else if let intID = row["id"] as? Int64 {
            id = intID
        } else {
            id = 0
        }
        sessionID = row["session_id"] as? String ?? ""
        timestamp = row["timestamp"] as? String ?? ""
        query = row["query"] as? String ?? ""
        tokenCount = row["token_count"] as? Int ?? 0
        claudeConversationID = row["claude_conversation_id"] as? String ?? ""

        if let rawChunkIDs = row["chunk_ids"] as? [String] {
            self.chunkIDs = rawChunkIDs
        } else if let text = row["chunk_ids"] as? String,
                  let data = text.data(using: .utf8),
                  let decoded = try? JSONSerialization.jsonObject(with: data) as? [String] {
            self.chunkIDs = decoded
        } else {
            self.chunkIDs = []
        }

        if let rawChunks = row["chunks"] as? [[String: Any]] {
            chunks = rawChunks.map(InjectionChunk.init(row:))
        } else {
            chunks = []
        }
    }
}
