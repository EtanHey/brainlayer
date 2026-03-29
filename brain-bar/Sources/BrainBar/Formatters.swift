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

    // MARK: - Search Results

    static func formatSearchResults(
        query: String,
        results: [[String: Any]],
        total: Int,
        useColor: Bool = true
    ) -> String {
        let q = truncate(query, maxLen: 50)

        if total == 0 {
            return "\u{250c}\u{2500} \(key("brain_search:", useColor)) \"\(val(q, useColor))\"\n\u{2502} No results found.\n\u{2514}\u{2500}"
        }

        var lines: [String] = []
        let countStr = num(total, useColor)
        let suffix = total != 1 ? "s" : ""
        lines.append("\u{250c}\u{2500} \(key("brain_search:", useColor)) \"\(val(q, useColor))\" \u{2500} \(countStr) result\(suffix)")
        lines.append("\u{2502}")

        for (i, r) in results.enumerated() {
            let score = r["score"] as? Double ?? 0
            let chunkId = String((r["chunk_id"] as? String ?? "").prefix(12))
            let project = truncate(r["project"] as? String, maxLen: 16)
            let date = String((r["created_at"] as? String ?? "").prefix(10))
            let importance = r["importance"]
            let summary = r["summary"] as? String ?? ""
            let content = r["content"] as? String ?? ""
            let displayText = truncate(summary.isEmpty ? content : summary, maxLen: 150)

            let impStr: String
            if let imp = importance as? Double {
                impStr = String(format: "%2d", Int(imp))
            } else if let imp = importance as? Int {
                impStr = String(format: "%2d", imp)
            } else {
                impStr = " \u{2500}"
            }

            let scoreStr = score > 0 ? String(format: "%.2f", score) : "0.00"

            lines.append("\u{251c}\u{2500} [\(num(i + 1, useColor))] \(chunkId)  \(key("score:", useColor))\(val(scoreStr, useColor))  \(key("imp:", useColor))\(val(impStr, useColor))  \(date)")
            lines.append("\u{2502}  \(pad(project, width: 16)) \u{2502} \(displayText)")

            let tags = parseTags(r["tags"])
            if !tags.isEmpty {
                let tagStr = tags.prefix(4).joined(separator: ", ")
                lines.append("\u{2502}  \(key("tags:", useColor)) \(tagStr)")
            }
            // Separator between results, but not after the last one
            if i < results.count - 1 {
                lines.append("\u{2502}")
            }
        }

        lines.append("\u{2514}\u{2500}")
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
        let eid = (entity["entity_id"] as? String) ?? (entity["id"] as? String) ?? ""
        let entityType = (entity["entity_type"] as? String)
            ?? ((entity["profile"] as? [String: Any])?["entity_type"] as? String)
            ?? ""

        var lines: [String] = []
        lines.append("\u{250c}\u{2500} \(key("Entity:", useColor)) \(val(name, useColor))")
        lines.append("\u{2502} \(key("id:", useColor)) \(eid)  \(key("type:", useColor)) \(entityType.isEmpty ? "unknown" : entityType)")

        // Profile
        if let profile = entity["profile"] as? [String: Any] {
            for k in ["role", "company", "location", "email", "phone"] {
                if let v = profile[k] as? String {
                    lines.append("\u{2502} \(key("\(k):", useColor)) \(v)")
                }
            }
        }

        // Constraints
        if let constraints = entity["hard_constraints"] as? [String: Any], !constraints.isEmpty {
            lines.append("\u{251c}\u{2500} Constraints")
            for (k, v) in Array(constraints.prefix(5)) {
                lines.append("\u{2502}   \(k): \(v)")
            }
        }

        // Preferences
        if let prefs = entity["preferences"] as? [String: Any], !prefs.isEmpty {
            lines.append("\u{251c}\u{2500} Preferences")
            for (k, v) in Array(prefs.prefix(5)) {
                lines.append("\u{2502}   \(k): \(v)")
            }
        }

        // Contact
        if let contact = entity["contact_info"] as? [String: Any], !contact.isEmpty {
            lines.append("\u{251c}\u{2500} Contact")
            for (k, v) in Array(contact.prefix(5)) {
                lines.append("\u{2502}   \(k): \(v)")
            }
        }

        // Relations
        if let relations = entity["relations"] as? [[String: Any]], !relations.isEmpty {
            lines.append("\u{251c}\u{2500} Relations (\(num(relations.count, useColor)))")
            for rel in relations.prefix(8) {
                let rtype = rel["relation_type"] as? String ?? ""
                let target: String
                if let t = rel["target"] as? [String: Any] {
                    target = t["name"] as? String ?? ""
                } else {
                    target = rel["target"] as? String ?? ""
                }
                lines.append("\u{2502}   \u{2192} \(rtype): \(target)")
            }
        }

        // Memories
        if let memories = entity["memories"] as? [[String: Any]], !memories.isEmpty {
            let memCount = entity["memory_count"] as? Int ?? memories.count
            lines.append("\u{251c}\u{2500} Memories (\(num(memCount, useColor)))")
            for mem in memories.prefix(5) {
                let mtype = pad(mem["type"] as? String, width: 8)
                let mdate = String((mem["date"] as? String ?? "").prefix(10))
                let mcontent = truncate(
                    (mem["content"] as? String) ?? (mem["summary"] as? String),
                    maxLen: 60
                )
                lines.append("\u{2502}   [\(mtype)] \(mdate) \(mcontent)")
            }
        }

        lines.append("\u{2514}\u{2500}")
        return lines.joined(separator: "\n")
    }

    // MARK: - Entity Simple

    static func formatEntitySimple(entity: [String: Any], useColor: Bool = true) -> String {
        if entity.isEmpty { return "" }

        let name = entity["name"] as? String ?? "Unknown"
        let eid = entity["id"] as? String ?? ""
        let etype = entity["entity_type"] as? String ?? ""

        var lines: [String] = []
        lines.append("\u{250c}\u{2500} \(key("Entity:", useColor)) \(val(name, useColor))")
        lines.append("\u{2502} \(key("id:", useColor)) \(eid)  \(key("type:", useColor)) \(etype.isEmpty ? "unknown" : etype)")

        // Relations
        if let relations = entity["relations"] as? [[String: Any]], !relations.isEmpty {
            lines.append("\u{251c}\u{2500} Relations (\(num(relations.count, useColor)))")
            for rel in relations.prefix(8) {
                let rtype = rel["relation_type"] as? String ?? "related_to"
                let target = (rel["target_name"] as? String) ?? (rel["name"] as? String) ?? ""
                lines.append("\u{2502}   \u{2192} \(rtype): \(target)")
            }
        }

        // Chunks
        if let chunks = entity["chunks"] as? [[String: Any]], !chunks.isEmpty {
            lines.append("\u{251c}\u{2500} Associated memories (\(num(chunks.count, useColor)))")
            for c in chunks.prefix(5) {
                let snippet = truncate(c["content"] as? String, maxLen: 60)
                lines.append("\u{2502}   \(snippet)")
            }
        }

        // Metadata
        if let metadata = entity["metadata"] as? [String: Any] {
            let interesting = metadata.filter { k, v in
                !["id", "name", "entity_type"].contains(k) && !(v is NSNull)
            }
            if !interesting.isEmpty {
                lines.append("\u{251c}\u{2500} Metadata")
                for (k, v) in Array(interesting.prefix(5)) {
                    let valStr = v is String ? truncate(v as? String, maxLen: 50) : "\(v)"
                    lines.append("\u{2502}   \(k): \(valStr)")
                }
            }
        }

        lines.append("\u{2514}\u{2500}")
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
        let q = truncate(query, maxLen: 40)

        var lines: [String] = []
        let suffix = total != 1 ? "s" : ""
        lines.append("\u{250c}\u{2500} \(key("Entity search:", useColor)) \"\(val(entityName, useColor))\" (\(key("query:", useColor)) \"\(q)\") \u{2500} \(num(total, useColor)) result\(suffix)")

        if !facts.isEmpty {
            lines.append("\u{251c}\u{2500} Knowledge Graph (\(num(facts.count, useColor)) fact\(facts.count != 1 ? "s" : ""))")
            for f in facts.prefix(5) {
                let src = f["source"] as? String ?? ""
                let rel = f["relation"] as? String ?? ""
                let tgt = f["target"] as? String ?? ""
                lines.append("\u{2502}   \(src) \u{2500}[\(rel)]\u{2192} \(tgt)")
            }
            lines.append("\u{2502}")
        }

        if !results.isEmpty {
            lines.append("\u{251c}\u{2500} Memories (\(num(total, useColor)))")
            for (i, r) in results.enumerated() {
                let score = r["score"] as? Double ?? 0
                let chunkId = String((r["chunk_id"] as? String ?? "").prefix(12))
                let snippet = truncate(
                    (r["snippet"] as? String) ?? (r["content"] as? String),
                    maxLen: 60
                )
                let scoreStr = score > 0 ? String(format: "%.2f", score) : "0.00"
                lines.append("\u{2502} [\(num(i + 1, useColor))] \(chunkId)  \(key("score:", useColor))\(val(scoreStr, useColor))")
                lines.append("\u{2502}     \(snippet)")
            }
        }

        lines.append("\u{2514}\u{2500}")
        return lines.joined(separator: "\n")
    }
}
