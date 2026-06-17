import Foundation

enum TextFormatter {
    private enum Alignment {
        case left
        case right
        case center
    }

    static func formatSearchResults(query: String, results: [SearchResult], total: Int, detail: String = "compact") -> String {
        let truncatedQuery = truncate(query, maxLen: 50)

        if total == 0 {
            return [
                "## Search results for \"\(truncatedQuery)\" - 0 of 0 shown",
                "",
                "No results found.",
            ].joined(separator: "\n")
        }

        // Only the explicit "full" detail level exposes chunk IDs, so they can be
        // chained into brain_update/brain_expand/brain_supersede/brain_archive.
        // Compact (the default) intentionally hides them.
        let includeChunkID = detail == "full"

        var lines = ["## Search results for \"\(truncatedQuery)\" - \(results.count) of \(total) shown"]

        for (index, result) in results.enumerated() {
            let title = titleLine(for: result)
            let preview = truncate(result.displayText, maxLen: 200)
            let source = sourceBasename(result.sourceFile.isEmpty ? result.project : result.sourceFile)
            let datePart = formattedDate(result.date)
            lines.append("")
            lines.append("### \(index + 1). \(title)")
            if includeChunkID && !result.chunkID.isEmpty {
                lines.append("- ID: \(result.chunkID)")
            }
            lines.append("- Source: \(source.isEmpty ? "unknown" : source)")
            lines.append("- Date: \(datePart.isEmpty ? "unknown" : datePart)")
            lines.append("- Preview: \(preview)")
        }
        return lines.joined(separator: "\n")
    }

    static func formatKGFacts(entity: String, facts: [BrainDatabase.KGFact]) -> String {
        guard !facts.isEmpty else { return "" }
        var lines: [String] = []
        lines.append("## Entity: \(entity)")
        lines.append("")
        lines.append("### KG Facts")
        for fact in facts.prefix(20) {
            lines.append("- \(fact.relationType): \(fact.relatedEntity)")
        }

        return lines.joined(separator: "\n")
    }

    static func formatEntityCard(_ entity: EntityCard) -> String {
        var lines = ["## Entity: \(entity.name)"]

        if !entity.description.isEmpty {
            lines.append("")
            lines.append(truncate(entity.description, maxLen: 200))
        }

        appendEntitySections(entity, to: &lines)
        return lines.joined(separator: "\n")
    }

    static func formatEntitySimple(_ entity: EntityCard) -> String {
        var lines = ["## Entity: \(entity.name)"]

        if !entity.description.isEmpty {
            lines.append("")
            lines.append(truncate(entity.description, maxLen: 200))
        }

        appendEntitySections(entity, to: &lines)
        return lines.joined(separator: "\n")
    }

    static func formatRecalledContext(query: String, results: [SearchResult]) -> String {
        let truncatedQuery = truncate(query, maxLen: 80)
        if results.isEmpty {
            return "## Recalled context for \"\(truncatedQuery)\"\n\nNo context available."
        }

        var lines = ["## Recalled context for \"\(truncatedQuery)\""]
        for (index, result) in results.enumerated() {
            let source = sourceBasename(result.sourceFile.isEmpty ? result.project : result.sourceFile)
            lines.append("")
            lines.append("### Chunk \(index + 1) - \(source.isEmpty ? "unknown" : source)")
            let content = result.snippet.isEmpty ? result.displayText : result.snippet
            let fullText = content.trimmingCharacters(in: .whitespacesAndNewlines)
            if fullText.count <= 1500 {
                lines.append(fullText)
            } else {
                lines.append(String(fullText.prefix(1500)) + "...")
                lines.append("")
                lines.append("Reference: \(result.chunkID)")
            }
        }
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
        var lines = ["## Search results for \"\(truncate(result.query, maxLen: 50))\" - \(result.results.count) of \(result.results.count) shown"]

        if !result.facts.isEmpty {
            lines.append("")
            lines.append("### KG Facts for \(result.entityName)")
            for fact in result.facts.prefix(5) {
                lines.append("- \(fact.source) \(fact.relation) \(fact.target)")
            }
        }

        for (index, memory) in result.results.enumerated() {
            let title = truncate(memory.snippet.isEmpty ? "Untitled result" : memory.snippet, maxLen: 100)
            let preview = truncate(memory.snippet, maxLen: 200)
            lines.append("")
            lines.append("### \(index + 1). \(title)")
            lines.append("- Source: unknown")
            lines.append("- Date: unknown")
            lines.append("- Preview: \(preview)")
        }

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

    private static func titleLine(for result: SearchResult) -> String {
        let preferred = result.summary.isEmpty ? result.displayText : result.summary
        let title = preferred
            .split(separator: "\n", maxSplits: 1)
            .first
            .map(String.init) ?? preferred
        return truncate(title.isEmpty ? "Untitled result" : title, maxLen: 100)
    }

    private static func sourceBasename(_ source: String) -> String {
        let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: "\\", with: "/")
        guard !trimmed.isEmpty else { return "" }
        return URL(fileURLWithPath: trimmed).lastPathComponent
    }

    private static func formattedDate(_ raw: String) -> String {
        String(raw.prefix(10))
    }

    private static func formattedDate(_ date: Date?) -> String? {
        guard let date else { return nil }
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }

    private static func relationLine(_ relation: EntityCard.Relation) -> String {
        var line = "- \(relation.relationType): \(relation.targetName)"
        if let expired = formattedDate(relation.expiredAt) {
            line += " (expired \(expired))"
        }
        return line
    }

    private static func appendEntitySections(_ entity: EntityCard, to lines: inout [String]) {
        appendKeyValueSection("Profile", values: entity.profile, to: &lines)
        appendKeyValueSection("Constraints", values: entity.hardConstraints, to: &lines)
        appendKeyValueSection("Preferences", values: entity.preferences, to: &lines)
        appendKeyValueSection("Contact", values: entity.contactInfo, to: &lines)

        lines.append("")
        lines.append("### KG Facts")
        if entity.relations.isEmpty {
            lines.append("- None")
        } else {
            for relation in entity.relations.prefix(20) {
                lines.append(relationLine(relation))
            }
        }

        lines.append("")
        lines.append("### Recent context")
        let memoryLines = entity.memories.map(\.content) + entity.chunks
        if memoryLines.isEmpty {
            lines.append("- None")
        } else {
            for memory in memoryLines.prefix(5) {
                lines.append("- \(truncate(memory, maxLen: 150))")
            }
        }

        lines.append("")
        lines.append("### Likely follow-ups")
        let followUps = entity.relations.map(\.targetName).filter { !$0.isEmpty }
        if followUps.isEmpty {
            lines.append("- None")
        } else {
            for target in followUps.prefix(5) {
                lines.append("- \(target)")
            }
        }
    }

    private static func appendKeyValueSection(_ title: String, values: [String: String], to lines: inout [String]) {
        let items = values
            .filter { !$0.value.isEmpty }
            .sorted { $0.key < $1.key }
        guard !items.isEmpty else { return }

        lines.append("")
        lines.append("### \(title)")
        for (key, value) in items.prefix(8) {
            lines.append("- \(key): \(value)")
        }
    }

    private static func formatNumber(_ number: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.groupingSeparator = ","
        return formatter.string(from: NSNumber(value: number)) ?? "\(number)"
    }
}
