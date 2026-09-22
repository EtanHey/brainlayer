# BrainBar chart ceiling visual evidence

Deterministic, headless fixture renders for baseline `d3eca8cefbf3c41220677a0148e82408c151e7fc` and production candidate `85698d83ea171c72d2938d3bdcc4a2c78e752a10`.

| View | Before SHA-256 | After SHA-256 | Before size | After size |
|---|---|---|---:|---:|
| compact | `cf167fba94b4c1fe66f224a5305196361c7dd1bf789197fe96c217045e31438e` | `f85757da775d9480512463180483ae1e3ec03009ac76d21a9b8ddc68f91254b2` | 1520×1280 | 1520×1280 |
| default | `d014bbf05ad29b3d2fbbeb1593263966c2ac8973661af284508bef8ddc191201` | `46e8037f310e28d6c5e8ee7540db1d125ccd21518d8fbaf39c4a848963b712f4` | 1920×1030 | 1920×1078 |
| wide | `b8f59e50af892c7d81b72d4d06d22e1413db0a59a5b0177e8420035fc9c916ea` | `83313f102889179ee26a6ce94e481d3070918897428233191588704f9963fcbb` | 2560×1280 | 2560×1280 |
| stale | `273045f57ef7603e36cfb08baea5a859299c9e86c0ce1db974d751483ce91046` | `53a8ac0b6ba66b89514c905c888af0d3677e1606bad2a5fb4464e5a1c705c5e6` | 1920×1280 | 1920×1280 |
| details expanded | `cdae36c3ca12caa2b2a7db8454ef49dc20fc9c744fd8cf023adffe0614fdfa92` | `7395ee1a0536ce49e28b63c4258489668ff8acc0bfb67035f0b29ee24229bb09` | 1920×2400 | 1920×2400 |

Verification on the production tree: 145 focused dashboard tests passed and `swift build --target BrainBar` passed. Independent Codex code review and the lead-routed Claude UX pass both passed.

These are source-built fixture pixels only. Visual verification was not performed on the installed app, and these artifacts do not establish installed, deployed, or live behavior.
