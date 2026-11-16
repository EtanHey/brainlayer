# Golems approval helper signing

`sign-golems-approve.yml` is manual dispatch only, from brainlayer `main`.
The lead supplies a reviewed full EtanHey/golems commit SHA. Before building, the
Actions checks out only fixed `master` with full history. The runner then freshly
fetches `origin/master` and requires the supplied SHA
to be an ancestor of (or equal to) master before locally checking out that exact
commit. No Actions checkout consumes the caller SHA. Fetch failure refuses; a stale tracking
ref is never accepted as fallback. It must contain
`native/golems-approve/GolemsApprove.entitlements` and `Info.plist`; missing files
refuse by name. No entitlements or bundle metadata are synthesized. The old
`30f54d0717a93f3f5dbd87b3a89661ee17e8775d` deliberately fails this admission gate.

The committed Developer ID provisioning profile is public signing metadata,
not a private key or credential. Its pinned SHA-256 is
`7287dac8a8a9b3308da5b96428471db1afb100ee6bf26250f2d5ff79082eef4f`.
It authorizes `PPN23G925Y.com.golems.approve`; the workflow accepts only that exact
application/team/access-group entitlement dictionary, with no runtime relaxations.
A different profile requires a reviewed pin update. No GitHub secret is created
or changed by this workflow.

## Owner prerequisite: protected developer-id environment (HOLD)

The owner must create `developer-id` in EtanHey/brainlayer, set EtanHey as the sole
required reviewer, prevent self-review, and allow only deployment **branch** `main`
(no tags or wildcard branches). A run initiated as EtanHey cannot be self-approved;
the owner/lead must account for the initiating identity when arranging the later run.
Move all six BRAINBAR Developer ID/notary secrets listed in
`scripts/golems_approve_release.py` into that environment, then remove their
repository-scoped copies so another agent-written workflow cannot obtain them.
This lane creates no environment, changes no settings, and reads/writes no secrets.
`brainbar-release.yml` is unchanged; the lead owns its separate migration PR after
the owner step. Coordinate that migration before the next BrainBar release.

**Until the owner removes repository-scoped signing authority, the enclave gives
no protection against an agent with brainlayer write access.** Declaring an
environment cannot retract authority exposed to other repository workflows.
GitHub also falls back to repository secrets with the same names: a nonempty value
is not evidence of its scope. The metadata gate verifies protections, not secret
placement/removal; the owner/lead must confirm that custody change separately.

Before the signing job is scheduled (avoiding implicit creation of an unprotected
environment), the build job uses read-only GitHub API metadata to require the named
environment, sole owner review, self-review prevention and exactly the main branch
policy. Missing/unknown protection, API failure or denied metadata access refuses
by the `developer-id` name. The protected sign job declares `environment:
developer-id`, rechecks metadata after its approval wait, and refuses missing
secrets by name before import or signing, without printing values.
[GitHub's environment API](https://docs.github.com/en/rest/deployments/environments)
and [branch-policy API](https://docs.github.com/en/rest/deployments/branch-policies)
are read with the workflow's `actions: read` token; no administration permission
or new credential is requested. Actual owner protections/approval remain a hosted
qualification gate, not something local tests prove.

Build runs without signing secrets on a separate macOS runner. Signing uses the
six BRAINBAR Developer ID/notary secrets resolved under `developer-id` and the existing BrainBar
release pattern: decode the PKCS#12 only into RUNNER_TEMP, import into a temporary
keychain, sign with hardened runtime/timestamp, notarize, and staple. Passwords,
identity and notary credentials never go into GITHUB_ENV, outputs, repository,
logs or uploaded artifacts. Credential-bearing tool output is suppressed. The
PKCS#12 and temporary keychain necessarily exist only on the ephemeral signing
runner; EXIT cleanup and a separate always() step remove both. Cleanup failure
prevents verification/publication. Nothing is imported into a developer Mac.

Verification happens after cleanup, without secret environment variables. It
checks strict/deep signature validity, pinned Team/identifier requirement,
exact entitlements, stapled ticket, Gatekeeper assessment, embedded profile hash,
and unchanged committed Info.plist. An unknown argument with empty stdin must
launch and return the helper's normal refusal (exit 2, empty stdout, bounded
15-second timeout). At the inspected 30f54d07 source this reaches neither keychain
access nor LAContext/Touch ID. The lead must preserve that property when choosing
a newer reviewed source SHA: this probe is not enrolment, approval, or proof of
Secure Enclave/keychain access. The source SHA is a deliberate trust input, not
an automatic approval of arbitrary golems code.

The final zip is extracted and verified again before uploading exactly three
files: GolemsApprove.zip, SHA256SUMS, requirement.txt. A separate Linux job alone
has contents:write and creates prerelease `golems-approve-<first 12 SHA chars>`.
The workflow refuses an existing release and never overwrites its assets, but the
release is **mutable**: another contents:write holder can replace assets or recreate
it. SHA256SUMS lives beside those assets and is a download convenience, not an
independent trust anchor. Enrolment's trust check is its committed `codesign -R`
requirement plus the pinned embedded provisioning-profile hash. The source's full
SHA is in the release notes.

BrainBar currently selects `latest-stable` Xcode rather than a concrete version;
this workflow keeps that selection and records `xcodebuild -version` and
`swift --version` from the build runner in requirement.txt. That is toolchain
provenance, not a reproducible-build claim.

Local validation: `actionlint .github/workflows/sign-golems-approve.yml`,
`bash -n scripts/sign-golems-approve.sh`, and
`pytest tests/test_golems_approve_release.py -q`.
These are lint/refusal evidence only. Signed build, Apple notarization, launch,
release publication, and installed-helper qualification require a later lead run.
