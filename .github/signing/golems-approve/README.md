# Golems approval helper signing

`sign-golems-approve.yml` is manual dispatch only, from brainlayer `main`.
The lead supplies a reviewed full EtanHey/golems commit SHA. Actions checks out only fixed `master` with full history. The runner then freshly
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

## Optional signing environment

The signing-secrets decision is still open. The repository variable
`GOLEMS_APPROVE_SIGNING_ENV` is the only switch:

- Unset or empty: the signing job has no environment and uses the existing
  repository-level BRAINBAR secrets, as today. No environment is required or
  created, and no environment metadata API is called.
- Set to an existing environment name: the signing job runs in that environment.
  Admission requires exactly deployment branch `main` and tag pattern `v*`, with
  the correct branch/tag types and no extra patterns. Missing environment,
  unreadable policy or a different policy refuses before scheduling signing.
  The selected policy is checked again before reading signing credentials.

Names use letters, digits, dots, dashes or underscores, start with a letter or
digit, and are at most 255 characters. The switch is resolved once on the build
runner; the chosen signing job and subsequent check use that captured value.
The two signing jobs are mutually exclusive and share the same signing steps.
Publishing requires the chosen job to succeed and the other job to be skipped.
Neither this workflow nor its admission check adds a release approval click.
The workflow remains manual dispatch from brainlayer `main`; allowing `v*` in
an environment's policy does not add a tag trigger to this workflow.

In both modes, the workflow still requires a golems commit already in master,
the pinned profile, exact committed entitlements and Info.plist, signing,
notarization, cleanup, and final verification. The named mode adds deployment-ref
restrictions for this job. It does not prove secret placement or remove signing
credentials available at repository scope: GitHub can fall back to repository
secrets with the same names. Repository-level signing credentials remain usable
by other workflows with access, so this switch alone does not provide signing-key
custody against a repository writer. Enrolment's signature/profile checks are
separate from that still-open custody decision.

No environment, repository variable, secret or setting is created or changed by
this workflow or this PR. There is no instruction to move secrets now.
`brainbar-release.yml` remains unchanged. Named-mode metadata uses GET-only
[environment](https://docs.github.com/en/rest/deployments/environments) and
[deployment-policy](https://docs.github.com/en/rest/deployments/branch-policies)
API calls with `actions: read`; no administration permission or new credential
is needed. Actual hosted policy enforcement and signing remain qualification
gates, not something local tests prove.

Build runs without signing secrets on a separate macOS runner. Signing uses the
six existing BRAINBAR Developer ID/notary secrets in the selected mode and the BrainBar
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
