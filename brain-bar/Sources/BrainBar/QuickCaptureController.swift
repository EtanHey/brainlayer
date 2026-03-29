// QuickCaptureController.swift — Business logic for capture + search flows.
//
// All database access goes through the BrainDatabase instance.
// No direct SQLite calls here — this is the coordination layer.

import Foundation

enum QuickCaptureController {

    struct CaptureResult {
        let chunkID: String
        let formatted: String
    }

    struct SearchResult {
        let count: Int
        let formatted: String
        let results: [[String: Any]]
    }

    enum CaptureError: LocalizedError {
        case emptyContent

        var errorDescription: String? {
            switch self {
            case .emptyContent: return "Content cannot be empty"
            }
        }
    }

    /// Store a quick capture note into BrainLayer.
    static func capture(
        db: BrainDatabase,
        content: String,
        tags: [String],
        importance: Int = 5
    ) throws -> CaptureResult {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw CaptureError.emptyContent }

        let stored = try db.store(
            content: trimmed,
            tags: tags,
            importance: importance,
            source: "quick-capture"
        )

        let formatted = Formatters.formatStoreResult(chunkId: stored.chunkID)
        return CaptureResult(chunkID: stored.chunkID, formatted: formatted)
    }

    /// Search BrainLayer and return formatted results.
    static func search(
        db: BrainDatabase,
        query: String,
        limit: Int = 10
    ) throws -> SearchResult {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return SearchResult(count: 0, formatted: "", results: [])
        }

        let results = try db.search(query: trimmed, limit: limit)
        let formatted = Formatters.formatSearchResults(
            query: trimmed,
            results: results,
            total: results.count
        )
        return SearchResult(count: results.count, formatted: formatted, results: results)
    }
}
