import Foundation

enum TextFormatter {
    private enum Alignment {
        case left
        case right
        case center
    }

    static func formatSearchResults(query: String, results: [SearchResult], total: Int) -> String {
        if total == 0 {
            return "<brain_search query=\"\(escapeXML(query))\" returned=\"0\" tool=\"brain_search\">\n\nNo results found.\n\n</brain_search>"
        }

        var lines = ["<brain_search query=\"\(escapeXML(query))\" returned=\"\(total)\" tool=\"brain_search\">"]
        lines.append("")
        lines.append("## Search: \"\(truncate(query, maxLen: 60))\"")
        lines.append("\(total) result\(total == 1 ? "" : "s") │ sorted by relevance")
        lines.append("")

        for (index, result) in results.enumerated() {
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("")

            let displayText = smartTruncate(result.summary.isEmpty ? result.snippet : result.summary, maxLen: 120)
            let datePart = String(result.date.prefix(10))
            let projectPart = result.project.isEmpty ? "" : result.project

            // Metadata line: relevance │ source │ date │ project
            var meta: [String] = [result.relevanceTier]
            if !result.sourceLabel.isEmpty { meta.append("from: \(result.sourceLabel)") }
            if !datePart.isEmpty { meta.append(datePart) }
            if !projectPart.isEmpty { meta.append("project: \(projectPart)") }

            lines.append("### ◇ [\(index + 1)] \(displayText)")
            lines.append(meta.joined(separator: " │ "))

            if !result.tags.isEmpty {
                lines.append("tags: \(result.tags.prefix(4).joined(separator: ", "))")
            }
            lines.append("")
        }

        lines.append("</brain_search>")
        return lines.joined(separator: "\n")
    }

    static func formatKGFacts(entity: String, facts: [BrainDatabase.KGFact]) -> String {
        guard !facts.isEmpty else { return "" }
        var lines: [String] = []
        lines.append("### ◆ \(entity)")
        lines.append("connections: \(facts.count)")
        lines.append("")

        // Split into outgoing and incoming, group by relation type
        let outgoing = facts.filter { $0.direction == "outgoing" }
        let incoming = facts.filter { $0.direction == "incoming" }

        if !outgoing.isEmpty {
            let grouped = Dictionary(grouping: outgoing) { $0.relationType }
            for (relType, group) in grouped.sorted(by: { $0.key < $1.key }) {
                let entities = group.map(\.relatedEntity).joined(separator: ", ")
                lines.append("→ \(relType.uppercased()): \(entities)")
            }
        }

        if !incoming.isEmpty {
            let grouped = Dictionary(grouping: incoming) { $0.relationType }
            for (relType, group) in grouped.sorted(by: { $0.key < $1.key }) {
                let entities = group.map(\.relatedEntity).joined(separator: ", ")
                lines.append("← \(relType.uppercased()): \(entities)")
            }
        }

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
        let typePart = entity.entityType.isEmpty ? "unknown" : entity.entityType
        var lines = ["<brain_entity name=\"\(escapeXML(entity.name))\" type=\"\(typePart)\" tool=\"brain_entity\">"]
        lines.append("")
        lines.append("## ◆ \(entity.name) [\(typePart.capitalized)]")

        if !entity.description.isEmpty {
            lines.append(truncate(entity.description, maxLen: 120))
        }

        lines.append("")

        // Grouped relations: outgoing then incoming, grouped by type, uppercase labels
        let outgoing = entity.relations.filter { $0.direction != "incoming" }
        let incoming = entity.relations.filter { $0.direction == "incoming" }

        if !outgoing.isEmpty {
            lines.append("### Outgoing relationships (\(outgoing.count))")
            let grouped = Dictionary(grouping: outgoing) { $0.relationType }
            for (relType, group) in grouped.sorted(by: { $0.key < $1.key }) {
                let targets = group.map(\.targetName).joined(separator: ", ")
                lines.append("→ \(relType.uppercased()): \(targets)")
            }
            lines.append("")
        }

        if !incoming.isEmpty {
            lines.append("### Incoming relationships (\(incoming.count))")
            let grouped = Dictionary(grouping: incoming) { $0.relationType }
            for (relType, group) in grouped.sorted(by: { $0.key < $1.key }) {
                let targets = group.map(\.targetName).joined(separator: ", ")
                lines.append("← \(relType.uppercased()): \(targets)")
            }
            lines.append("")
        }

        if entity.relations.isEmpty {
            lines.append("No relationships found.")
            lines.append("")
        }

        if !entity.metadata.isEmpty {
            lines.append("### Metadata")
            for (key, value) in entity.metadata.sorted(by: { $0.key < $1.key }).prefix(6) {
                lines.append("- \(key): \(truncate(value, maxLen: 60))")
            }
            lines.append("")
        }

        lines.append("</brain_entity>")
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

    private static func escapeXML(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }

    /// Truncate at sentence boundary when possible. Falls back to word boundary.
    private static func smartTruncate(_ text: String, maxLen: Int) -> String {
        let clean = text.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        guard clean.count > maxLen else { return clean }

        let cutoff = String(clean.prefix(maxLen))
        // Try sentence boundary (. ! ?)
        if let lastSentence = cutoff.range(of: "[.!?]\\s", options: .regularExpression, range: cutoff.startIndex..<cutoff.endIndex, locale: nil)?.upperBound {
            let trimmed = String(cutoff[cutoff.startIndex..<lastSentence]).trimmingCharacters(in: .whitespaces)
            if trimmed.count > maxLen / 3 {
                return trimmed
            }
        }
        // Fall back to word boundary
        if let lastSpace = cutoff.lastIndex(of: " ") {
            return String(cutoff[cutoff.startIndex..<lastSpace]) + "…"
        }
        return cutoff + "…"
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
