#!/usr/bin/env python3
"""Compatibility error for the retired cloud batch monitor."""

import sys


def main() -> int:
    print("ERROR: batch enrichment monitoring is retired; saved checkpoints and results are preserved", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
