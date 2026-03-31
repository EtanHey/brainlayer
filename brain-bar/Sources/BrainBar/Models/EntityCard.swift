import Foundation

struct EntityCard: Equatable {
    struct Relation: Equatable {
        let relationType: String
        let targetName: String
    }

    struct Memory: Equatable {
        let type: String
        let date: String
        let content: String
    }

    let id: String
    let name: String
    let entityType: String
    let profile: [String: String]
    let hardConstraints: [String: String]
    let preferences: [String: String]
    let contactInfo: [String: String]
    let relations: [Relation]
    let memories: [Memory]
    let memoryCount: Int
    let metadata: [String: String]
    let chunks: [String]

    init(
        id: String,
        name: String,
        entityType: String = "",
        profile: [String: String] = [:],
        hardConstraints: [String: String] = [:],
        preferences: [String: String] = [:],
        contactInfo: [String: String] = [:],
        relations: [Relation] = [],
        memories: [Memory] = [],
        memoryCount: Int? = nil,
        metadata: [String: String] = [:],
        chunks: [String] = []
    ) {
        self.id = id
        self.name = name
        self.entityType = entityType
        self.profile = profile
        self.hardConstraints = hardConstraints
        self.preferences = preferences
        self.contactInfo = contactInfo
        self.relations = relations
        self.memories = memories
        self.memoryCount = memoryCount ?? memories.count
        self.metadata = metadata
        self.chunks = chunks
    }

    init(lookupPayload: [String: Any]) {
        id = (lookupPayload["entity_id"] as? String) ?? (lookupPayload["id"] as? String) ?? ""
        name = lookupPayload["name"] as? String ?? "Unknown"
        entityType = lookupPayload["entity_type"] as? String ?? ""
        profile = [:]
        hardConstraints = [:]
        preferences = [:]
        contactInfo = [:]
        relations = ((lookupPayload["relations"] as? [[String: Any]]) ?? []).map {
            Relation(
                relationType: $0["relation_type"] as? String ?? "related_to",
                targetName: ($0["target_name"] as? String) ?? (($0["name"] as? String) ?? (($0["target"] as? [String: Any])?["name"] as? String ?? ""))
            )
        }
        memories = []
        memoryCount = 0
        metadata = EntityCard.decodeMetadata(lookupPayload["metadata"])
        chunks = ((lookupPayload["chunks"] as? [[String: Any]]) ?? []).compactMap {
            ($0["content"] as? String) ?? ($0["summary"] as? String)
        }
    }

    private static func decodeMetadata(_ raw: Any?) -> [String: String] {
        if let metadata = raw as? [String: String] {
            return metadata
        }
        guard let text = raw as? String, let data = text.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        var result: [String: String] = [:]
        for (key, value) in parsed {
            result[key] = String(describing: value)
        }
        return result
    }
}
