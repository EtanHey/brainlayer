# BrainBar healthy-note renders

Status: offscreen render evidence, not installed-app verification.

The before and after PNGs render the production Jobs and Backups views with synthetic supervisor
and calendar evidence: all steps succeed, and a weekly run replaces the nightly. The fixture
models the MBP's reported calendar shape; its dates, PIDs and backup values are synthetic.
Historical backup dates in this fixture are not current operational evidence.

- Before: healthy owner notes have amber icons; Maintenance says Awaiting next run; nightly has no run.
- After: owner notes are neutral; Maintenance says Healthy; nightly says Covered by weekly with a date.

The before renders were supplied by the first implementer. Regenerate the after renders from the
repository root without starting BrainBar or connecting to its production socket:

```sh
cd brain-bar
BRAINBAR_SOCKET_PATH=/nonexistent BRAINLAYER_FORBID_BRAINBAR_SOCKET=1 \
  BRAINBAR_ZERO_WARNINGS_RENDER_DIR="$PWD/../docs/test-artifacts/brainbar-zero-warnings/after" \
  swift test --filter BrainBarTests.BrainBarZeroWarningsRenderTests
```

The test also emits OCR text for assertions. Only the PNGs are committed here.
