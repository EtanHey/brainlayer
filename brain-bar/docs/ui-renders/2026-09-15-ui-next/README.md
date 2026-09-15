# BrainBar UI next render comparison

These PNGs are deterministic `BrainBarDashboardSnapshotTests` fixture renders. They do not show a
launched app or prove pointer-event delivery.

- `before-default.png` and `before-details.png`: `main` at `d207572f` (BrainBar 1.5.32).
- `after-default.png` and `after-details.png`: `wt/brainbar-ui-next` at `8c28f5ff`.

The images were reduced to 1,200 pixels wide for review; the snapshot harness retained the full-size
outputs locally.
