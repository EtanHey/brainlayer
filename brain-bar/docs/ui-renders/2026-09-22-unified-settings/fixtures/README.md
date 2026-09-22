# Unified BrainBar Settings visual evidence

Source-built, headless fixture renders. The before image comes from baseline `d3eca8cefbf3c41220677a0148e82408c151e7fc` and shows the standalone Settings content. The after images come from reviewed production commit `f668323ce74f7778a198be070fbdf21d411e2496` and show Settings inside the BrainBar window at three widths. The fixtures differ, so these images demonstrate the window/navigation change, not a pixel-for-pixel content comparison.

| Image | Pixels | SHA-256 |
|---|---:|---|
| [Before: standalone Settings](before/standalone-settings.png) | 1400×2160 | `c5dd8f069f19088376f5166f236c7682d0e11adc87dbd93752f7e17ecc79fbb2` |
| [After: compact](after/unified-settings-compact.png) | 1520×1440 | `b25570011e94847bb8d428ad068c547742ce5000b06e31ee1454d2ba9509a9ed` |
| [After: default](after/unified-settings-default.png) | 1920×1440 | `9c6efb3bf2cf43e81f5808b55f7b6bda8f9f03493a9f9335a18c1062707b3900` |
| [After: wide](after/unified-settings-wide.png) | 2560×1440 | `bed390e9a1462d38802fd2706abedc164d8c80007f5fdeb8197e84a780d37d79` |

The before snapshot test passed (1/1). The after renderer passed its directional-state and chart-marker probes; all four PNGs were inspected at original resolution. Both render paths use isolated fixtures and do not open the canonical database or installed app. Installed visual QA not performed.
