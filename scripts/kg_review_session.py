#!/usr/bin/env python3
"""CLI for the KG flag-batch review session driver (voice + visual surfaces).

Usage:
    python3 scripts/kg_review_session.py next --batch B.json --decisions D.json [--category C]
    python3 scripts/kg_review_session.py record --batch B.json --decisions D.json --cluster-id ID --decision-json '{...}'
    python3 scripts/kg_review_session.py rule --batch B.json --decisions D.json --rule-json '{...}'
    python3 scripts/kg_review_session.py stats --batch B.json --decisions D.json

`next` prints {"cluster": {...}, "speak": "..."} or {"cluster": null} when done.
See brainlayer/kg_review_session.py for the decisions-file contract.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brainlayer.kg_review_session import (  # noqa: E402
    apply_rule,
    next_undecided,
    record_decision,
    speak_text,
    stats,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_next = sub.add_parser("next")
    p_next.add_argument("--batch", required=True)
    p_next.add_argument("--decisions", required=True)
    p_next.add_argument("--category")

    p_rec = sub.add_parser("record")
    p_rec.add_argument("--batch", required=True)
    p_rec.add_argument("--decisions", required=True)
    p_rec.add_argument("--cluster-id", required=True)
    p_rec.add_argument("--decision-json", required=True)

    p_rule = sub.add_parser("rule")
    p_rule.add_argument("--batch", required=True)
    p_rule.add_argument("--decisions", required=True)
    p_rule.add_argument("--rule-json", required=True)

    p_stats = sub.add_parser("stats")
    p_stats.add_argument("--batch", required=True)
    p_stats.add_argument("--decisions", required=True)

    args = ap.parse_args()

    if args.cmd == "next":
        cluster = next_undecided(args.batch, args.decisions, category=args.category)
        out = {"cluster": cluster}
        if cluster:
            out["speak"] = speak_text(cluster)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.cmd == "record":
        stamped = record_decision(args.batch, args.decisions, args.cluster_id, json.loads(args.decision_json))
        print(json.dumps({"recorded": args.cluster_id, "decision": stamped}, ensure_ascii=False))
    elif args.cmd == "rule":
        n = apply_rule(args.batch, args.decisions, json.loads(args.rule_json))
        print(json.dumps({"applied": n}))
    elif args.cmd == "stats":
        print(json.dumps(stats(args.batch, args.decisions), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
