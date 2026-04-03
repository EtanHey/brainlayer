import Foundation

enum TextFormatter {
    private enum Alignment {
        case left
        case right
        case center
    }

    static func formatSearchResults(query: String, results: [SearchResult], total: Int) -> String {
        if total == 0 {
            return "┌─ brain_search: \"\(truncate(query, maxLen: 50))\"\n│ No results found.\n└─"
        }

        var lines = ["┌─ brain_search: \"\(truncate(query, maxLen: 50))\" ─ \(total) result\(total == 1 ? "" : "s")", "│"]

        for (index, result) in results.enumerated() {
            let displayText = truncate(result.summary.isEmpty ? result.snippet : result.summary, maxLen: 72)
            let importance = result.importance.map { String(format: "%2d", $0) } ?? " ─"
            lines.append("├─ [\(index + 1)] \(String(result.chunkID.prefix(12)))  score:\(scoreString(result.score))  imp:\(importance)  \(String(result.date.prefix(10)))")
            lines.append("│  \(pad(result.project, width: 16)) │ \(displayText)")
            if !result.tags.isEmpty {
                lines.append("│  tags: \(result.tags.prefix(4).joined(separator: ", "))")
            }
            lines.append("│")
        }

        lines.append("└─")
        return lines.joined(separator: "\n")
    }

    static func formatKGFacts(entity: String, facts: [BrainDatabase.KGFact]) -> String {
        guard !facts.isEmpty else { return "" }
        var lines = [
            "┌─ KG: \(entity)",
            "│"
        ]
        // Group by relation type for scannability
        let grouped = Dictionary(grouping: facts) { $0.relationType }
        for (relType, group) in grouped.sorted(by: { $0.key < $1.key }) {
            let arrow = group[0].direction == "incoming" ? "←" : "→"
            let entities = group.map(\.relatedEntity).joined(separator: ", ")
            lines.append("│  \(arrow) \(relType): \(entities)")
        }
        lines.append("└─")
        return lines.joined(separator: "\n")
    }

    static func formatEntityCard(_ entity: EntityCard) -> String {
        var lines = [
            "┌─ Entity: \(entity.name)",
            "│ id: \(entity.id)  type: \(entity.entityType.isEmpty ? "unknown" : entity.entityType)"
        ]

        if !entity.description.isEmpty {
            lines.append("│ \(truncate(entity.description, maxLen: 100))")
        }

        for key in ["role", "company", "location", "email", "phone"] {
            if let value = entity.profile[key], !value.isEmpty {
                lines.append("│ \(key): \(value)")
            }
        }

        if !entity.hardConstraints.isEmpty {
            lines.append("├─ Constraints")
            for (key, value) in entity.hardConstraints.sorted(by: { $0.key < $1.key }).prefix(5) {
                lines.append("│   \(key): \(value)")
            }
        }

        if !entity.preferences.isEmpty {
            lines.append("├─ Preferences")
            for (key, value) in entity.preferences.sorted(by: { $0.key < $1.key }).prefix(5) {
                lines.append("│   \(key): \(value)")
            }
        }

        if !entity.contactInfo.isEmpty {
            lines.append("├─ Contact")
            for (key, value) in entity.contactInfo.sorted(by: { $0.key < $1.key }).prefix(5) {
                lines.append("│   \(key): \(value)")
            }
        }

        if !entity.relations.isEmpty {
            lines.append("├─ Relations (\(entity.relations.count))")
            for relation in entity.relations.prefix(8) {
                let arrow = relation.direction == "incoming" ? "←" : "→"
                lines.append("│   \(arrow) \(relation.relationType): \(relation.targetName)")
            }
        }

        if !entity.memories.isEmpty {
            lines.append("├─ Memories (\(entity.memoryCount))")
            for memory in entity.memories.prefix(5) {
                lines.append("│   [\(pad(memory.type, width: 8))] \(String(memory.date.prefix(10))) \(truncate(memory.content, maxLen: 60))")
            }
        }

        lines.append("└─")
        return lines.joined(separator: "\n")
    }

    static func formatEntitySimple(_ entity: EntityCard) -> String {
        var lines = [
            "┌─ Entity: \(entity.name)",
            "│ id: \(entity.id)  type: \(entity.entityType.isEmpty ? "unknown" : entity.entityType)"
        ]

        if !entity.description.isEmpty {
            lines.append("│ \(truncate(entity.description, maxLen: 100))")
        }

        if !entity.relations.isEmpty {
            lines.append("├─ Relations (\(entity.relations.count))")
            for relation in entity.relations.prefix(8) {
                let arrow = relation.direction == "incoming" ? "←" : "→"
                lines.append("│   \(arrow) \(relation.relationType): \(relation.targetName)")
            }
        }

        if !entity.chunks.isEmpty {
            lines.append("├─ Associated memories (\(entity.chunks.count))")
            for chunk in entity.chunks.prefix(5) {
                lines.append("│   \(truncate(chunk, maxLen: 60))")
            }
        }

        if !entity.metadata.isEmpty {
            lines.append("├─ Metadata")
            for (key, value) in entity.metadata.sorted(by: { $0.key < $1.key }).prefix(5) {
                lines.append("│   \(key): \(truncate(value, maxLen: 50))")
            }
        }

        lines.append("└─")
        return lines.joined(separator: "\n")
    }

    static func formatDigestResult(_ result: DigestResult) -> String {
        if let attempted = result.attempted,
           let enriched = result.enriched,
           let skipped = result.skipped,
           let failed = result.failed {
            return [
                "┌─ brain_digest (enrich)",
                "│ Attempted: \(attempted)  Enriched: \(enriched)  Skipped: \(skipped)  Failed: \(failed)",
                "└─"
            ].joined(separator: "\n")
        }

        var lines = [
            "┌─ brain_digest (\(result.mode))",
            "│ Chunks: \(result.chunks)  Entities: \(result.entities)  Relations: \(result.relations)"
        ]

        if !result.actionItems.isEmpty {
            lines.append("├─ Action items (\(result.actionItems.count))")
            for item in result.actionItems.prefix(5) {
                lines.append("│   • \(truncate(item, maxLen: 60))")
            }
        }

        lines.append("└─")
        return lines.joined(separator: "\n")
    }

    static func formatStats(_ stats: StatsResult) -> String {
        [
            "┌─ BrainLayer Stats",
            "│ Chunks: \(formatNumber(stats.totalChunks))",
            "│ Projects: \(stats.projects.prefix(12).joined(separator: ", "))\(stats.projects.count > 12 ? "..." : "")",
            "│ Types: \(stats.contentTypes.joined(separator: ", "))",
            "└─"
        ].joined(separator: "\n")
    }

    static func formatKGSearch(_ result: KGSearchResult) -> String {
        var lines = [
            "┌─ Entity search: \"\(result.entityName)\" (query: \"\(truncate(result.query, maxLen: 40))\") ─ \(result.results.count) result\(result.results.count == 1 ? "" : "s")"
        ]

        if !result.facts.isEmpty {
            lines.append("├─ Knowledge Graph (\(result.facts.count) fact\(result.facts.count == 1 ? "" : "s"))")
            for fact in result.facts.prefix(5) {
                lines.append("│   \(fact.source) ─[\(fact.relation)]→ \(fact.target)")
            }
            lines.append("│")
        }

        if !result.results.isEmpty {
            lines.append("├─ Memories (\(result.results.count))")
            for (index, memory) in result.results.enumerated() {
                lines.append("│ [\(index + 1)] \(String(memory.chunkID.prefix(12)))  score:\(scoreString(memory.score))")
                lines.append("│     \(truncate(memory.snippet, maxLen: 60))")
            }
        }

        lines.append("└─")
        return lines.joined(separator: "\n")
    }

    private static func truncate(_ text: String, maxLen: Int = 80) -> String {
        let clean = text.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        guard clean.count > maxLen else { return clean }
        return String(clean.prefix(maxLen - 1)) + "…"
    }

    private static func pad(_ text: String, width: Int, align: Alignment = .left) -> String {
        let truncated = text.count > width ? String(text.prefix(width - 1)) + "…" : text
        switch align {
        case .left:
            return truncated.padding(toLength: width, withPad: " ", startingAt: 0)
        case .right:
            return String(repeating: " ", count: max(0, width - truncated.count)) + truncated
        case .center:
            let totalPadding = max(0, width - truncated.count)
            let leftPadding = totalPadding / 2
            let rightPadding = totalPadding - leftPadding
            return String(repeating: " ", count: leftPadding) + truncated + String(repeating: " ", count: rightPadding)
        }
    }

    private static func scoreString(_ score: Double) -> String {
        score == 0 ? "0.00" : String(format: "%.2f", score)
    }

    private static func formatNumber(_ number: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.groupingSeparator = ","
        return formatter.string(from: NSNumber(value: number)) ?? "\(number)"
    }
}
