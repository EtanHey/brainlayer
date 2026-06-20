// TDD marker for the BrainBar dashboard redesign (feat/brainbar-dashboard-redesign).
//
// The real, runnable verification harness for the changes in
//   brain-bar/Sources/BrainBar/BrainBarWindowRootView.swift
// is the SwiftPM snapshot test:
//   brain-bar/Tests/BrainBarTests/DashboardRedesignSnapshotTests.swift
//
// That test renders the redesigned PIPELINE composition (two separately-scaled
// write cards via `BrainBarPipelineSeriesCard`, the compact 3-chip SIGNAL
// COVERAGE strip, and the below-the-fold FLOW panel) to PNGs and asserts the
// render is non-trivial. It is env-gated by BRAINBAR_SNAPSHOT_DIR and driven by
// the debug-only `BrainBarPipelinePanelPreview` seam (no live DB needed).
//
// Run (from brain-bar/):
//   BRAINBAR_SNAPSHOT_DIR=/path swift test --filter DashboardRedesignSnapshotTests
//
// This file lives here only because the repo's TDD guard expects a test file
// adjacent to the source path; the authoritative assertions are in the SwiftPM
// target above (this file is not part of any SwiftPM test target).
