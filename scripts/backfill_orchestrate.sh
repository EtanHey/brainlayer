#!/usr/bin/env bash
# Stale operator commands must fail without resetting checkpoints or submitting jobs.
printf '%s\n' 'ERROR: batch enrichment orchestration is retired; saved checkpoints and results are preserved' >&2
exit 1
