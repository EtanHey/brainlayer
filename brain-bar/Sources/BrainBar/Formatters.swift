// Formatters.swift — Beautiful terminal output formatting for MCP tool responses.
//
// Port of Python _format.py. Uses Unicode box-drawing characters.
// ANSI color codes optional via useColor parameter.

import Foundation

enum Formatters {

    // MARK: - ANSI Color Codes

    private static let orange = "\u{1b}[38;2;232;121;36m"
    private static let blue   = "\u{1b}[38;2;88;166;255m"
    private static let green  = "\u{1b}[38;2;63;185;80m"
    private static let reset  = "\u{1b}[0m"

    private static func val(_ text: String, _ useColor: Bool) -> String {
        useColor ? "\(orange)\(text)\(reset)" : text
    }

    private static func key(_ text: String, _ useColor: Bool) -> String {
        useColor ? "\(blue)\(text)\(reset)" : text
    }

    private static func num(_ value: Any, _ useColor: Bool) -> String {
        let s: String
        if let i = value as? Int {
            s = formatNumber(i)
        } else if let d = value as? Double {
            s = String(format: "%.2f", d)
        } else {
            s = "\(value)"
        }
        return useColor ? "\(green)\(s)\(reset)" : s
    }

    // MARK: - Helpers

    private static func truncate(_ text: String?, maxLen: Int = 80) -> String {
        guard let text, !text.isEmpty else { return "" }
        let clean = text.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
        if clean.count <= maxLen { return clean }
        return String(clean.prefix(maxLen - 1)) + "\u{2026}"
    }

    private static func pad(_ text: String?, width: Int, align: Alignment = .left) -> String {
        let t = text ?? ""
        let s = t.count > width ? String(t.prefix(width - 1)) + "\u{2026}" : t
        switch align {
        case .left:   return s.padding(toLength: width, withPad: " ", startingAt: 0)
        case .right:  return String(repeating: " ", count: max(0, width - s.count)) + s
        case .center:
            let pad = max(0, width - s.count)
            let left = pad / 2
            return String(repeating: " ", count: left) + s + String(repeating: " ", count: pad - left)
        }
    }

    private enum Alignment { case left, right, center }

    private static let decimalFormatter: NumberFormatter = {
        let fmt = NumberFormatter()
        fmt.numberStyle = .decimal
        fmt.groupingSeparator = ","
        return fmt
    }()

    private static func formatNumber(_ n: Int) -> String {
        return decimalFormatter.string(from: NSNumber(value: n)) ?? "\(n)"
    }

    private static func parseTags(_ raw: Any?) -> [String] {
        if let arr = raw as? [String] { return arr }
        guard let str = raw as? String, !str.isEmpty else { return [] }
        // Parse JSON array string like '["a", "b"]'
        if let data = str.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data) as? [String] {
            return parsed
        }
        return []
    }

    private static func basename(_ raw: Any?) -> String {
        guard let text = raw as? String else { return "unknown" }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: "\\", with: "/")
        guard !trimmed.isEmpty else { return "unknown" }
        return URL(fileURLWithPath: trimmed).lastPathComponent
    }

    private static func dateOnly(_ raw: Any?) -> String {
        guard let text = raw as? String, !text.isEmpty else { return "unknown" }
        return String(text.prefix(10))
    }

    private static func relationTarget(_ rel: [String: Any]) -> String {
        if let target = rel["target"] as? [String: Any] {
            return target["name"] as? String ?? ""
        }
        return (rel["target_name"] as? String)
            ?? (rel["name"] as? String)
            ?? (rel["target"] as? String)
            ?? ""
    }

    private static func expiredDate(_ rel: [String: Any]) -> String? {
        let raw = (rel["expired_at"] as? String) ?? (rel["expiredAt"] as? String)
        guard let raw, !raw.isEmpty else { return nil }
        return String(raw.prefix(10))
    }

    private static func appendKeyValueSection(
        _ title: String,
        values: [String: Any]?,
        to lines: inout [String],
        skip: Set<String> = []
    ) {
        guard let values else { return }
        let items = values
            .filter { key, value in
                !skip.contains(key) && !(value is NSNull) && !String(describing: value).isEmpty
            }
            .sorted { $0.key < $1.key }
        guard !items.isEmpty else { return }

        lines.append("")
        lines.append("### \(title)")
        for (key, value) in items.prefix(8) {
            lines.append("- \(key): \(value)")
        }
    }

    // MARK: - Search Results

    static func formatSearchResults(
        query: String,
        results: [[String: Any]],
        total: Int,
        useColor: Bool = true
    ) -> String {
        let q = truncate(query, maxLen: 50)

        if total == 0 {
            return "## Search results for \"\(q)\" - 0 of 0 shown\n\nNo results found."
        }

        var lines: [String] = []
        lines.append("## Search results for \"\(q)\" - \(results.count) of \(total) shown")

        for (i, r) in results.enumerated() {
            let summary = r["summary"] as? String ?? ""
            let snippet = (r["snippet"] as? String) ?? (r["content"] as? String) ?? ""
            let title = truncate(summary.isEmpty ? snippet : summary, maxLen: 100)
            let source = basename(r["source_file"] ?? r["project"])
            let date = dateOnly(r["date"] ?? r["created_at"])
            let preview = truncate(snippet.isEmpty ? summary : snippet, maxLen: 200)
            lines.append("")
            lines.append("### \(i + 1). \(title.isEmpty ? "Untitled result" : title)")
            lines.append("- Source: \(source)")
            lines.append("- Date: \(date)")
            lines.append("- Preview: \(preview)")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Store Result

    static func formatStoreResult(
        chunkId: String,
        superseded: String? = nil,
        queued: Bool = false,
        useColor: Bool = true
    ) -> String {
        if queued {
            return "\u{2502} \u{23f3} Memory queued (DB busy) \u{2500} will flush on next successful store."
        }
        var parts = ["\u{2714} Stored \u{2192} \(val(chunkId, useColor))"]
        if let superseded {
            parts.append(" (superseded \(val(superseded, useColor)))")
        }
        return parts.joined()
    }

    // MARK: - Entity Card

    static func formatEntityCard(entity: [String: Any], useColor: Bool = true) -> String {
        let name = entity["name"] as? String ?? "Unknown"

        var lines: [String] = ["## Entity: \(name)"]
        if let description = entity["description"] as? String, !description.isEmpty {
            lines.append("")
            lines.append(truncate(description, maxLen: 200))
        }

        appendKeyValueSection(
            "Profile",
            values: entity["profile"] as? [String: Any],
            to: &lines,
            skip: ["hard_constraints", "preferences", "contact_info", "description"]
        )
        appendKeyValueSection("Constraints", values: entity["hard_constraints"] as? [String: Any], to: &lines)
        appendKeyValueSection("Preferences", values: entity["preferences"] as? [String: Any], to: &lines)
        appendKeyValueSection("Contact", values: entity["contact_info"] as? [String: Any], to: &lines)

        lines.append("")
        lines.append("### KG Facts")
        if let relations = entity["relations"] as? [[String: Any]], !relations.isEmpty {
            for rel in relations.prefix(8) {
                let rtype = rel["relation_type"] as? String ?? ""
                var line = "- \(rtype): \(relationTarget(rel))"
                if let expired = expiredDate(rel) {
                    line += " (expired \(expired))"
                }
                lines.append(line)
            }
        } else {
            lines.append("- None")
        }

        lines.append("")
        lines.append("### Recent context")
        if let memories = entity["memories"] as? [[String: Any]], !memories.isEmpty {
            for mem in memories.prefix(5) {
                let mcontent = truncate(
                    (mem["content"] as? String) ?? (mem["summary"] as? String),
                    maxLen: 150
                )
                lines.append("- \(mcontent)")
            }
        } else {
            lines.append("- None")
        }

        lines.append("")
        lines.append("### Likely follow-ups")
        let relations = entity["relations"] as? [[String: Any]] ?? []
        let followUps = relations.map { relationTarget($0) }.filter { !$0.isEmpty }
        if followUps.isEmpty {
            lines.append("- None")
        } else {
            for target in followUps.prefix(5) {
                lines.append("- \(target)")
            }
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Entity Simple

    static func formatEntitySimple(entity: [String: Any], useColor: Bool = true) -> String {
        if entity.isEmpty { return "" }

        let name = entity["name"] as? String ?? "Unknown"

        var lines: [String] = ["## Entity: \(name)"]

        // Relations
        lines.append("")
        lines.append("### KG Facts")
        if let relations = entity["relations"] as? [[String: Any]], !relations.isEmpty {
            for rel in relations.prefix(8) {
                let rtype = rel["relation_type"] as? String ?? "related_to"
                var line = "- \(rtype): \(relationTarget(rel))"
                if let expired = expiredDate(rel) {
                    line += " (expired \(expired))"
                }
                lines.append(line)
            }
        } else {
            lines.append("- None")
        }

        // Chunks
        lines.append("")
        lines.append("### Recent context")
        if let chunks = entity["chunks"] as? [[String: Any]], !chunks.isEmpty {
            for c in chunks.prefix(5) {
                let snippet = truncate(c["content"] as? String, maxLen: 150)
                lines.append("- \(snippet)")
            }
        } else {
            lines.append("- None")
        }

        lines.append("")
        lines.append("### Likely follow-ups")
        let relations = entity["relations"] as? [[String: Any]] ?? []
        let followUps = relations.map { relationTarget($0) }.filter { !$0.isEmpty }
        if followUps.isEmpty {
            lines.append("- None")
        } else {
            for target in followUps.prefix(5) {
                lines.append("- \(target)")
            }
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Stats

    static func formatStats(stats: [String: Any], useColor: Bool = true) -> String {
        let total = stats["total_chunks"] as? Int ?? 0
        let projects = stats["projects"] as? [String] ?? []
        let types = stats["content_types"] as? [String] ?? []

        var lines: [String] = []
        lines.append("\u{250c}\u{2500} \(key("BrainLayer Stats", useColor))")
        lines.append("\u{2502} \(key("Chunks:", useColor)) \(num(total, useColor))")
        let projStr = projects.prefix(12).joined(separator: ", ") + (projects.count > 12 ? "..." : "")
        lines.append("\u{2502} \(key("Projects:", useColor)) \(projStr)")
        lines.append("\u{2502} \(key("Types:", useColor)) \(types.joined(separator: ", "))")
        lines.append("\u{2514}\u{2500}")
        return lines.joined(separator: "\n")
    }

    // MARK: - Digest Result

    static func formatDigestResult(result: [String: Any], useColor: Bool = true) -> String {
        let mode = result["mode"] as? String ?? "digest"

        // Enrich mode
        if result["attempted"] != nil {
            let attempted = result["attempted"] as? Int ?? 0
            let enriched = result["enriched"] as? Int ?? 0
            let skipped = result["skipped"] as? Int ?? 0
            let failed = result["failed"] as? Int ?? 0
            return [
                "\u{250c}\u{2500} \(key("brain_digest", useColor)) (enrich)",
                "\u{2502} \(key("Attempted:", useColor)) \(num(attempted, useColor))  \(key("Enriched:", useColor)) \(num(enriched, useColor))  \(key("Skipped:", useColor)) \(num(skipped, useColor))  \(key("Failed:", useColor)) \(num(failed, useColor))",
                "\u{2514}\u{2500}",
            ].joined(separator: "\n")
        }

        // Digest / connect mode
        let stats = result["stats"] as? [String: Any] ?? [:]
        let chunks = (result["chunks_created"] as? Int)
            ?? (stats["chunks_created"] as? Int)
            ?? (result["chunks"] as? Int) ?? 0
        let entities = (result["entities_created"] as? Int)
            ?? (stats["entities_found"] as? Int)
            ?? (result["entities"] as? Int) ?? 0
        let relations = (result["relations_created"] as? Int)
            ?? (stats["relations_created"] as? Int)
            ?? (result["relations"] as? Int) ?? 0

        var lines: [String] = []
        lines.append("\u{250c}\u{2500} \(key("brain_digest", useColor)) (\(mode))")
        lines.append("\u{2502} \(key("Chunks:", useColor)) \(num(chunks, useColor))  \(key("Entities:", useColor)) \(num(entities, useColor))  \(key("Relations:", useColor)) \(num(relations, useColor))")

        // Action items
        let extracted = result["extracted"] as? [String: Any] ?? [:]
        let actions = (result["action_items"] as? [[String: Any]])
            ?? (extracted["action_items"] as? [[String: Any]])
            ?? []
        if !actions.isEmpty {
            lines.append("\u{251c}\u{2500} Action items (\(num(actions.count, useColor)))")
            for a in actions.prefix(5) {
                let desc = truncate(a["description"] as? String ?? "\(a)", maxLen: 60)
                lines.append("\u{2502}   \u{2022} \(desc)")
            }
        }

        lines.append("\u{2514}\u{2500}")
        return lines.joined(separator: "\n")
    }

    // MARK: - KG Search

    static func formatKGSearch(
        entityName: String,
        results: [[String: Any]],
        facts: [[String: Any]],
        query: String,
        useColor: Bool = true
    ) -> String {
        let total = results.count
        let q = truncate(query, maxLen: 50)
        var lines = ["## Search results for \"\(q)\" - \(total) of \(total) shown"]

        if !facts.isEmpty {
            lines.append("")
            lines.append("### KG Facts for \(entityName)")
            for f in facts.prefix(5) {
                let src = f["source"] as? String ?? ""
                let rel = f["relation"] as? String ?? ""
                let tgt = f["target"] as? String ?? ""
                lines.append("- \(src) \(rel) \(tgt)".trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }

        for (i, r) in results.enumerated() {
            let snippet = (r["snippet"] as? String) ?? (r["content"] as? String) ?? ""
            let summary = (r["summary"] as? String) ?? snippet
            let titleSource = summary.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? "Untitled result"
            let source = basename(r["source_file"] ?? r["project"])
            let date = dateOnly(r["date"] ?? r["created_at"])
            lines.append("")
            lines.append("### \(i + 1). \(truncate(titleSource.isEmpty ? "Untitled result" : titleSource, maxLen: 100))")
            lines.append("- Source: \(source)")
            lines.append("- Date: \(date)")
            lines.append("- Preview: \(truncate(snippet.isEmpty ? summary : snippet, maxLen: 200))")
        }

        return lines.joined(separator: "\n")
    }
}
