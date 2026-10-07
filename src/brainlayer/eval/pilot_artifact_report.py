"""Offline analysis and reporting for historical faceted-tag pilot artifacts."""

import json


def analyze_results(results, chunks_meta):
    """Analyze pilot results and return stats."""
    valid = [r for r in results if r["parsed"] is not None and isinstance(r["parsed"], dict)]
    invalid = [r for r in results if r["parsed"] is None or not isinstance(r["parsed"], dict)]

    # Collect all tags
    all_topics = []
    all_activities = []
    all_domains = []
    confidences = []
    noise_count = 0

    for r in valid:
        p = r["parsed"]
        topics = p.get("b_topics", [])
        if not isinstance(topics, list):
            topics = []
        all_topics.extend(topics)
        if "_noise" in topics:
            noise_count += 1
        act = p.get("c_activity", "")
        if act:
            all_activities.append(act)
        domains = p.get("d_domain", [])
        if not isinstance(domains, list):
            domains = []
        all_domains.extend(domains)
        conf = p.get("e_confidence")
        if conf is not None:
            confidences.append(conf)

    # Topic frequency
    topic_freq = {}
    for t in all_topics:
        topic_freq[t] = topic_freq.get(t, 0) + 1

    # Activity frequency
    act_freq = {}
    for a in all_activities:
        act_freq[a] = act_freq.get(a, 0) + 1

    # Domain frequency
    dom_freq = {}
    for d in all_domains:
        dom_freq[d] = dom_freq.get(d, 0) + 1

    # Confidence distribution
    conf_buckets = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
    for c in confidences:
        if c < 0.3:
            conf_buckets["0.0-0.3"] += 1
        elif c < 0.5:
            conf_buckets["0.3-0.5"] += 1
        elif c < 0.7:
            conf_buckets["0.5-0.7"] += 1
        elif c < 0.9:
            conf_buckets["0.7-0.9"] += 1
        else:
            conf_buckets["0.9-1.0"] += 1

    # Compare with existing tags
    tag_comparison = []
    for r in valid:
        old_tags = r.get("old_tags", "")
        if old_tags:
            try:
                old_list = json.loads(old_tags) if isinstance(old_tags, str) else old_tags
            except (json.JSONDecodeError, TypeError):
                old_list = []
            new_topics = r["parsed"].get("b_topics", [])
            if old_list or new_topics:
                tag_comparison.append(
                    {
                        "chunk_id": r["chunk_id"][:40],
                        "source": r["source"],
                        "old": old_list[:5] if old_list else [],
                        "new": new_topics,
                        "confidence": r["parsed"].get("e_confidence", 0),
                    }
                )

    return {
        "total": len(results),
        "valid_json": len(valid),
        "invalid_json": len(invalid),
        "errors": [r["error"] for r in invalid],
        "unique_topics": len(set(all_topics)),
        "topic_freq": dict(sorted(topic_freq.items(), key=lambda x: -x[1])[:30]),
        "activity_freq": dict(sorted(act_freq.items(), key=lambda x: -x[1])),
        "domain_freq": dict(sorted(dom_freq.items(), key=lambda x: -x[1])),
        "confidence_dist": conf_buckets,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "noise_count": noise_count,
        "tag_comparisons": tag_comparison[:20],
    }


def write_report(stats, results_path):
    """Write results to markdown."""
    lines = [
        "# Enrichment Pilot Results — Faceted Tag Prompt v2",
        "",
        "> **Date:** 2026-03-19",
        "> **Model:** Gemini 2.5 Flash",
        f"> **Chunks tested:** {stats['total']}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Valid JSON responses:** {stats['valid_json']}/{stats['total']} ({stats['valid_json'] / stats['total'] * 100:.1f}%)",
        f"- **Parse failures:** {stats['invalid_json']}",
        f"- **Unique topic tags generated:** {stats['unique_topics']}",
        f"- **Average confidence:** {stats['avg_confidence']:.3f}",
        f"- **Noise-flagged chunks:** {stats['noise_count']}",
        "",
        "---",
        "",
        "## Confidence Distribution",
        "",
        "| Range | Count |",
        "|-------|-------|",
    ]
    for bucket, count in stats["confidence_dist"].items():
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Topic Tags (top 30)",
            "",
            "| Tag | Count |",
            "|-----|-------|",
        ]
    )
    for tag, count in stats["topic_freq"].items():
        lines.append(f"| `{tag}` | {count} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Activity Distribution",
            "",
            "| Activity | Count |",
            "|----------|-------|",
        ]
    )
    for act, count in stats["activity_freq"].items():
        lines.append(f"| `{act}` | {count} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Domain Distribution",
            "",
            "| Domain | Count |",
            "|--------|-------|",
        ]
    )
    for dom, count in stats["domain_freq"].items():
        lines.append(f"| `{dom}` | {count} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Tag Comparison: Old vs New (sample)",
            "",
            "| Source | Old Tags | New Topics | Confidence |",
            "|--------|----------|------------|------------|",
        ]
    )
    for comp in stats["tag_comparisons"]:
        old_str = ", ".join(comp["old"][:3]) if comp["old"] else "(none)"
        new_str = ", ".join(comp["new"][:3]) if comp["new"] else "(none)"
        lines.append(f"| {comp['source']} | {old_str} | {new_str} | {comp['confidence']:.2f} |")

    if stats["errors"]:
        lines.extend(
            [
                "",
                "---",
                "",
                "## Errors",
                "",
            ]
        )
        for err in stats["errors"][:10]:
            lines.append(f"- {err}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Verdict",
            "",
            "**TODO:** Fill in after reviewing results.",
            "",
        ]
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines))
    print(f"\nReport written to {results_path}")
