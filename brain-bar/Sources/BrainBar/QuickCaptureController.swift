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

        var candidateResultsByID: [String: [String: Any]] = [:]
        var candidateOrder: [String] = []
        let candidates = try db.searchCandidates(query: trimmed, limit: limit)
        for candidate in candidates {
            candidateOrder.append(candidate.id)
            candidateResultsByID[candidate.id] = try searchResult(from: candidate, db: db)
        }

        var seenChunkIDs = Set<String>()
        var results: [[String: Any]] = []
        let fallbackRows = try db.search(query: trimmed, limit: limit)
        for row in fallbackRows {
            guard let chunkID = row["chunk_id"] as? String else { continue }
            guard seenChunkIDs.insert(chunkID).inserted else { continue }
            if let candidateResult = candidateResultsByID[chunkID] {
                results.append(candidateResult)
            } else {
                results.append(searchResult(from: row))
            }
            if results.count >= limit { break }
        }
        for chunkID in candidateOrder {
            guard seenChunkIDs.insert(chunkID).inserted else { continue }
            guard let candidateResult = candidateResultsByID[chunkID] else { continue }
            results.append(candidateResult)
            if results.count >= limit { break }
        }
        let formatted = Formatters.formatSearchResults(
            query: trimmed,
            results: results,
            total: results.count
        )
        return SearchResult(count: results.count, formatted: formatted, results: results)
    }

    private static func searchResult(from candidate: SearchQueryCandidate, db: BrainDatabase) throws -> [String: Any] {
        let chunk = try db.getChunk(id: candidate.id)
        let fullContent = (chunk?["content"] as? String) ?? candidate.previewText
        return [
            "chunk_id": candidate.id,
            "content": candidate.previewText,
            "full_content": fullContent,
            "created_at": candidate.date,
            "source_file": chunk?["source_file"] as Any,
            "project": candidate.project,
            "content_type": chunk?["content_type"] as Any,
            "importance": candidate.importance,
            "summary": chunk?["summary"] as Any,
            "tags": chunk?["tags"] as Any,
            "source": chunk?["source"] as Any,
            "score": candidate.lexicalScore,
        ]
    }

    private static func searchResult(from row: [String: Any]) -> [String: Any] {
        let previewText = (row["preview_text"] as? String) ?? (row["content"] as? String) ?? "Untitled result"
        return [
            "chunk_id": row["chunk_id"] as Any,
            "content": previewText,
            "full_content": (row["content"] as? String) ?? previewText,
            "created_at": row["created_at"] as Any,
            "source_file": row["source_file"] as Any,
            "project": row["project"] as Any,
            "content_type": row["content_type"] as Any,
            "importance": row["importance"] as Any,
            "summary": row["summary"] as Any,
            "tags": row["tags"] as Any,
            "source": row["source"] as Any,
            "score": row["score"] as Any,
        ]
    }
}
