# Cloud model retirement ratchet

The required PR ratchet row measures **no cloud model call reachable anywhere**.
It combines source inspection, two fresh installed Python wheels, actual native
BrainBar router tests, and a strict exact-head report validator. It does not claim
deployment, a live embedding model, or production database/socket coverage.

Run `scripts/retirement_run.py` with the repository, its full commit SHA, a fresh
work directory outside that repository, an output JSON path, and
`--dependency-ref origin/main` (or a dependency-baseline tag). The resolved
merge-base must be an ancestor, and both installed profiles must prove the R10b
SDK-removal state. CI uses the PR base reference rather than a permanent commit
pin. Missing dependency evidence is PENDING_R10B,
never GREEN. macOS, a working Swift/XCTest toolchain, and Python 3.12 are required.
Build/install failures and unavailable capabilities fail the gate.

The candidate is built from a verified Git archive, with a declared validation
build stamp. Default and dev installs have private HOME, cache, configuration,
database and temporary paths, synthetic credentials, offline model loading,
isolated Python, an early attempt recorder, and an OS network denial boundary.
All wheel module origins and installed package bytes must match. Both profiles
require cloud SDKs to be absent from dependency declarations and import discovery;
Drive/OAuth imports and local bge routing remain required. Synthetic vectors prove
local persistence without claiming a real embedding model run.

The static AST scan resolves absolute, relative, from-list, aliased and lazy
imports, including Opus L2 replay mutations. Scripts, hooks, Swift, shell and
dashboard sources also belong to the hashed corpus. A closed policy binds only
transport-capable imports and HTTP, subprocess and dynamic-code call ASTs to
their path and owner, without enclosing-body hashes. Non-Python files are bound
to exact bytes only when they contain URLSession, NWConnection, CFStream, curl,
fetch or socket transport tokens. An unknown or changed binding fails. Policy changes require
an explicit non-cloud purpose and pair review; the runner never updates policy.
The generated policy is evidence of classifications, not a cloud exception list.
Run `python scripts/retirement_policy_generate.py --check` to verify it, or
`--write` to carry existing purposes forward by site key. New sites require an
explicit reviewed key-to-purpose JSON supplied with `--purposes`; without it,
generation fails without writing. Harmless Python helper/docstring edits do not
change call bindings. The immutable runner also checks generator freshness.

Each runtime profile must pass all twelve transport controls, then record zero
forbidden candidate attempts. The only admitted child is the exact local
`/bin/ps -o lstart= -p <self>` provenance read under the OS boundary. Taxonomy
provenance is bound to the measured SHA. CLI/MCP retirement shims, hotlane vectors,
historical replay, saved-result drain, watcher flush and Drive/OAuth imports all
execute from the installed package. Retained Ollama rejects remote, credentialed
and ambiguous endpoints before sending; its loopback/proxy/redirect policy is
also bound by the static inventory.

Native proof compiles the actual XCTest bundle from the same archive, loads it
through the actual XCTest host, and runs under OS network denial plus private
connect/sendto interposition. All three router, persistence and boundary tests
must execute with zero skips, exactly one blocked network control and no candidate
attempt. Plain `swift test` skips the two harness-bound tests when its boundary is
unavailable; those skips can never satisfy the retirement ratchet.

CI always uploads the report and raw logs. A failed/missing job, stale head,
incomplete inventory, changed hash, missing control or skipped mandatory leg is
RED. Validation uses explicit checks and remains armed under `python -O`.
The ratchet table cannot reuse a successful report from a failed job. Release,
installed launchd rerendering and key removal remain separate release gates.
