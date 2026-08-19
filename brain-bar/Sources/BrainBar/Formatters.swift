// Formatters.swift — Beautiful terminal output formatting for MCP tool responses.
//
// Port of Python _format.py. Uses Unicode box-drawing characters.
// ANSI color codes optional via useColor parameter.

import Foundation

/// The canonical brain_store outcome vocabulary, shared with the Python MCP path
/// (`src/brainlayer/mcp/_format.py: STORE_OUTCOMES`) so an agent can branch on the
/// outcome without knowing which server answered.
///
/// `merged` is Python-only in practice: BrainBar deliberately writes NULL simhash
/// columns (see the comment in `BrainDatabase.store`), so it has no near-duplicate
/// merge to report and its dedupe surfaces as `duplicate`. It is still rendered
/// here so both paths speak one vocabulary.
enum StoreOutcome: String, Sendable, Equatable {
    case stored
    case duplicate
    case merged
    case deferred
    case rejected
    case error

    /// The value published in the MCP response's structured `status` field.
    var status: String { rawValue.uppercased() }

    /// Whether a new chunk row was written. `false` means the content is already
    /// in BrainLayer under the returned chunk_id and re-storing it is wrong.
    var storedNew: Bool { self == .stored }
}

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

    /// Render one brain_store outcome that resolved to a chunk, so an agent cannot
    /// misread it. Every rendering states three things, because agents that could
    /// not tell a fresh write from a suppressed duplicate kept re-storing:
    ///   (a) the OUTCOME WORD — branchable without parsing prose;
    ///   (b) the CANONICAL chunk_id the write resolved to;
    ///   (c) the INSTRUCTION — above all, whether re-storing is right or wrong.
    ///
    /// For the outcomes that store nothing (`rejected`, `error`) use
    /// `formatStoreFailure`, which has no chunk_id to report.
    static func formatStoreResult(
        chunkId: String,
        superseded: String? = nil,
        tags: [String] = [],
        queued: Bool = false,
        queuedReason: String = "DB_BUSY",
        outcome: StoreOutcome? = nil,
        useColor: Bool = true
    ) -> String {
        let resolved = outcome ?? (queued ? .deferred : .stored)
        let id = val(chunkId, useColor)

        switch resolved {
        case .deferred:
            // The prefix stays "STORED (deferred)": Etan retired a bare "DEFERRED:"
            // on 2026-08-09 because it read as failure and agents re-stored on it.
            // The 2026-08-19 ask is met by the explicit WILL-be-stored promise, not
            // by renaming the prefix back. Machines branch on the structured status.
            let idSuffix = chunkId.isEmpty ? "" : " \u{2192} \(id)"
            let reasonLabel = queuedReason == "DB_BUSY" ? "DB busy" : queuedReason
            return "\u{2502} \u{2714} STORED (deferred): \(reasonLabel)\(idSuffix) \u{2500} durably queued; "
                + "the drain persists it automatically \u{2500} it WILL be stored under exactly this "
                + "chunk_id. Do NOT re-store, do NOT retry, and do NOT save a fallback copy."

        case .duplicate:
            return "\u{2714} DUPLICATE \u{2192} \(id) \u{2500} an identical memory was already stored; "
                + "the existing chunk was refreshed, not re-inserted. Do NOT re-store \u{2500} reference \(id)."

        case .merged:
            return "\u{2714} MERGED \u{2192} \(id) \u{2500} a near-identical memory already existed and "
                + "your content was merged into it. Do NOT re-store \u{2500} reference \(id)."

        case .rejected, .error:
            // No chunk exists for these; callers should use formatStoreFailure.
            return formatStoreFailure(outcome: resolved, reason: nil, useColor: useColor)

        case .stored:
            var parts = ["\u{2714} STORED \u{2192} \(id) \u{2500} new memory written"]
            if !tags.isEmpty {
                parts.append(" [tags: \(tags.joined(separator: ", "))]")
            }
            if let superseded {
                parts.append(", superseding \(val(superseded, useColor))")
            }
            parts.append(". Nothing further to do.")
            return parts.joined()
        }
    }

    /// Render a brain_store outcome that stored nothing. These are the only two
    /// outcomes with no chunk_id, and saying so explicitly is what stops an agent
    /// from treating a failure as a deferred write it must not retry.
    static func formatStoreFailure(
        outcome: StoreOutcome,
        reason: String?,
        useColor: Bool = true
    ) -> String {
        switch outcome {
        case .rejected:
            return "\u{2716} REJECTED: \(reason ?? "content is not eligible for storage") \u{2500} "
                + "nothing was stored and there is no chunk_id. Do NOT retry the same content; "
                + "rewrite it or drop it."
        default:
            return "\u{2716} ERROR: \(reason ?? "store failed") \u{2500} the memory was NOT stored and "
                + "there is no chunk_id. Retry once; if it repeats, report it \u{2500} do NOT save a "
                + "fallback copy."
        }
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
            let title = truncate(titleSource, maxLen: 100)
            let source = basename(r["source_file"] ?? r["project"])
            let date = dateOnly(r["date"] ?? r["created_at"])
            lines.append("")
            lines.append("### \(i + 1). \(title.isEmpty ? "Untitled result" : title)")
            lines.append("- Source: \(source)")
            lines.append("- Date: \(date)")
            lines.append("- Preview: \(truncate(snippet.isEmpty ? summary : snippet, maxLen: 200))")
        }

        return lines.joined(separator: "\n")
    }
}
