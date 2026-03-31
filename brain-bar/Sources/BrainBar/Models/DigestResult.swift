import Foundation

struct DigestResult: Equatable {
    let mode: String
    let attempted: Int?
    let enriched: Int?
    let skipped: Int?
    let failed: Int?
    let chunks: Int
    let entities: Int
    let relations: Int
    let actionItems: [String]

    init(
        mode: String,
        attempted: Int?,
        enriched: Int?,
        skipped: Int?,
        failed: Int?,
        chunks: Int,
        entities: Int,
        relations: Int,
        actionItems: [String]
    ) {
        self.mode = mode
        self.attempted = attempted
        self.enriched = enriched
        self.skipped = skipped
        self.failed = failed
        self.chunks = chunks
        self.entities = entities
        self.relations = relations
        self.actionItems = actionItems
    }

    init(payload: [String: Any]) {
        mode = payload["mode"] as? String ?? "digest"
        attempted = payload["attempted"] as? Int
        enriched = payload["enriched"] as? Int
        skipped = payload["skipped"] as? Int
        failed = payload["failed"] as? Int

        let stats = payload["stats"] as? [String: Any] ?? [:]
        chunks = payload["chunks_created"] as? Int ?? stats["chunks_created"] as? Int ?? payload["chunks"] as? Int ?? 0
        entities = payload["entities_created"] as? Int ?? stats["entities_found"] as? Int ?? payload["entities"] as? Int ?? 0
        relations = payload["relations_created"] as? Int ?? stats["relations_created"] as? Int ?? payload["relations"] as? Int ?? 0

        let extracted = payload["extracted"] as? [String: Any] ?? [:]
        let rawItems = (payload["action_items"] as? [Any]) ?? (extracted["action_items"] as? [Any]) ?? []
        actionItems = rawItems.compactMap {
            if let text = $0 as? String {
                return text
            }
            if let dict = $0 as? [String: Any] {
                return dict["description"] as? String
            }
            return nil
        }
    }
}
