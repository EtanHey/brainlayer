import Foundation

struct EntityCard: Equatable {
    struct Relation: Equatable {
        let relationType: String
        let targetName: String
        let targetEntityId: String?
        let direction: String  // "outgoing" or "incoming"
        let validUntil: Date?
        let expiredAt: Date?

        init(
            relationType: String,
            targetName: String,
            targetEntityId: String? = nil,
            direction: String = "outgoing",
            validUntil: Date? = nil,
            expiredAt: Date? = nil
        ) {
            self.relationType = relationType
            self.targetName = targetName
            self.targetEntityId = targetEntityId
            self.direction = direction
            self.validUntil = validUntil
            self.expiredAt = expiredAt
        }

        var displayText: String {
            if direction == "incoming" {
                return "\(targetName) \(relationType)"
            }
            return "\(relationType) \(targetName)"
        }
    }

    struct Memory: Equatable {
        let type: String
        let date: String
        let content: String
    }

    let id: String
    let name: String
    let entityType: String
    let description: String
    let profile: [String: String]
    let hardConstraints: [String: String]
    let preferences: [String: String]
    let contactInfo: [String: String]
    let relations: [Relation]
    let memories: [Memory]
    let memoryCount: Int
    let metadata: [String: String]
    let chunks: [String]
    let importance: Double?
    let altitudeTierTitle: String?

    init(
        id: String,
        name: String,
        entityType: String = "",
        description: String = "",
        profile: [String: String] = [:],
        hardConstraints: [String: String] = [:],
        preferences: [String: String] = [:],
        contactInfo: [String: String] = [:],
        relations: [Relation] = [],
        memories: [Memory] = [],
        memoryCount: Int? = nil,
        metadata: [String: String] = [:],
        chunks: [String] = [],
        importance: Double? = nil,
        altitudeTierTitle: String? = nil
    ) {
        self.id = id
        self.name = name
        self.entityType = entityType
        self.description = description
        self.profile = profile
        self.hardConstraints = hardConstraints
        self.preferences = preferences
        self.contactInfo = contactInfo
        self.relations = relations
        self.memories = memories
        self.memoryCount = memoryCount ?? memories.count
        self.metadata = metadata
        self.chunks = chunks
        self.importance = importance
        self.altitudeTierTitle = altitudeTierTitle
    }

    init(lookupPayload: [String: Any], importance: Double? = nil, altitudeTierTitle: String? = nil) {
        id = (lookupPayload["entity_id"] as? String) ?? (lookupPayload["id"] as? String) ?? ""
        name = lookupPayload["name"] as? String ?? "Unknown"
        entityType = lookupPayload["entity_type"] as? String ?? ""
        description = lookupPayload["description"] as? String ?? ""
        profile = [:]
        hardConstraints = [:]
        preferences = [:]
        contactInfo = [:]
        relations = ((lookupPayload["relations"] as? [[String: Any]]) ?? [])
            .filter { ($0["relation_type"] as? String) != "co_occurs_with" }
            .map {
                Relation(
                    relationType: $0["relation_type"] as? String ?? "related_to",
                    targetName: ($0["target_name"] as? String) ?? (($0["name"] as? String) ?? (($0["target"] as? [String: Any])?["name"] as? String ?? "")),
                    targetEntityId: ($0["target_entity_id"] as? String) ?? ($0["target_id"] as? String),
                    direction: $0["direction"] as? String ?? "outgoing",
                    validUntil: KGTemporalDate.parse($0["valid_until"] ?? $0["validUntil"]),
                    expiredAt: KGTemporalDate.parse($0["expired_at"] ?? $0["expiredAt"])
                )
            }
        memories = []
        memoryCount = 0
        metadata = EntityCard.decodeMetadata(lookupPayload["metadata"])
        chunks = ((lookupPayload["chunks"] as? [[String: Any]]) ?? []).compactMap {
            ($0["content"] as? String) ?? ($0["summary"] as? String)
        }
        self.importance = importance ?? EntityCard.decodeDouble(lookupPayload["importance"])
        self.altitudeTierTitle = altitudeTierTitle ?? (lookupPayload["altitude_tier"] as? String)
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

    private static func decodeDouble(_ raw: Any?) -> Double? {
        if let value = raw as? Double {
            return value
        }
        if let value = raw as? Int {
            return Double(value)
        }
        if let text = raw as? String {
            return Double(text)
        }
        return nil
    }
}
