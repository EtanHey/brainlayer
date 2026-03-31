import Foundation

struct InjectionEvent: Equatable, Identifiable, Sendable {
    let id: Int64
    let sessionID: String
    let timestamp: String
    let query: String
    let chunkIDs: [String]
    let tokenCount: Int

    var chunkCount: Int { chunkIDs.count }

    var summaryLine: String {
        "\(query) • \(chunkCount) chunks • \(tokenCount) tok"
    }

    init(
        id: Int64,
        sessionID: String,
        timestamp: String,
        query: String,
        chunkIDs: [String],
        tokenCount: Int
    ) {
        self.id = id
        self.sessionID = sessionID
        self.timestamp = timestamp
        self.query = query
        self.chunkIDs = chunkIDs
        self.tokenCount = tokenCount
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

        if let rawChunkIDs = row["chunk_ids"] as? [String] {
            self.chunkIDs = rawChunkIDs
        } else if let text = row["chunk_ids"] as? String,
                  let data = text.data(using: .utf8),
                  let decoded = try JSONSerialization.jsonObject(with: data) as? [String] {
            self.chunkIDs = decoded
        } else {
            self.chunkIDs = []
        }
    }
}
