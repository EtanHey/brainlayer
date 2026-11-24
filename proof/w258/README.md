# Activity: ingestion and memory reads

Invented data only. Offscreen NDRenderHarness renders of the actual Activity card, 900 pt window width (852 pt page, 820 pt chart/header).
Round 1 uses the same independent read fixture before and after. Before is the reviewed c0b69eee drawing, with a behavior-preserving geometry extraction for fail-first tests. Data includes zero stretches, bursts, reads above saves, and reads below saves.
The after card anchors zero to the visible axis and gives read calls a teal data colour. All captions use the selected chart bucket span with slash units (up to 4,000/24 min); a zero peak has no caption. The approved first shortening step fits every range, retaining exact totals, 13/11 pt type, all three labels, inline scales and the 22 pt header.
The hover readout shows counts for its selected bucket, including the third series.
These images do not establish installed-app or live MCP acceptance; those belong to the release batch.

| State | Before | After |
|---|---|---|
| burst, day, dark, peak, hover | ![before](before/burst-day-dark-peak-hover.png) | ![after](after/burst-day-dark-peak-hover.png) |
| burst, day, dark, rest | ![before](before/burst-day-dark-rest.png) | ![after](after/burst-day-dark-rest.png) |
| burst, day, dark, zero, hover | ![before](before/burst-day-dark-zero-hover.png) | ![after](after/burst-day-dark-zero-hover.png) |
| burst, day, light, peak, hover | ![before](before/burst-day-light-peak-hover.png) | ![after](after/burst-day-light-peak-hover.png) |
| burst, day, light, rest | ![before](before/burst-day-light-rest.png) | ![after](after/burst-day-light-rest.png) |
| burst, day, light, zero, hover | ![before](before/burst-day-light-zero-hover.png) | ![after](after/burst-day-light-zero-hover.png) |
| burst, hour, dark, peak, hover | ![before](before/burst-hour-dark-peak-hover.png) | ![after](after/burst-hour-dark-peak-hover.png) |
| burst, hour, dark, rest | ![before](before/burst-hour-dark-rest.png) | ![after](after/burst-hour-dark-rest.png) |
| burst, hour, dark, zero, hover | ![before](before/burst-hour-dark-zero-hover.png) | ![after](after/burst-hour-dark-zero-hover.png) |
| burst, hour, light, peak, hover | ![before](before/burst-hour-light-peak-hover.png) | ![after](after/burst-hour-light-peak-hover.png) |
| burst, hour, light, rest | ![before](before/burst-hour-light-rest.png) | ![after](after/burst-hour-light-rest.png) |
| burst, hour, light, zero, hover | ![before](before/burst-hour-light-zero-hover.png) | ![after](after/burst-hour-light-zero-hover.png) |
| burst, threeHours, dark, peak, hover | ![before](before/burst-threeHours-dark-peak-hover.png) | ![after](after/burst-threeHours-dark-peak-hover.png) |
| burst, threeHours, dark, rest | ![before](before/burst-threeHours-dark-rest.png) | ![after](after/burst-threeHours-dark-rest.png) |
| burst, threeHours, dark, zero, hover | ![before](before/burst-threeHours-dark-zero-hover.png) | ![after](after/burst-threeHours-dark-zero-hover.png) |
| burst, threeHours, light, peak, hover | ![before](before/burst-threeHours-light-peak-hover.png) | ![after](after/burst-threeHours-light-peak-hover.png) |
| burst, threeHours, light, rest | ![before](before/burst-threeHours-light-rest.png) | ![after](after/burst-threeHours-light-rest.png) |
| burst, threeHours, light, zero, hover | ![before](before/burst-threeHours-light-zero-hover.png) | ![after](after/burst-threeHours-light-zero-hover.png) |
| quiet, day, dark, rest | ![before](before/quiet-day-dark-rest.png) | ![after](after/quiet-day-dark-rest.png) |
| quiet, day, light, rest | ![before](before/quiet-day-light-rest.png) | ![after](after/quiet-day-light-rest.png) |
| quiet, hour, dark, rest | ![before](before/quiet-hour-dark-rest.png) | ![after](after/quiet-hour-dark-rest.png) |
| quiet, hour, light, rest | ![before](before/quiet-hour-light-rest.png) | ![after](after/quiet-hour-light-rest.png) |
| quiet, threeHours, dark, rest | ![before](before/quiet-threeHours-dark-rest.png) | ![after](after/quiet-threeHours-dark-rest.png) |
| quiet, threeHours, light, rest | ![before](before/quiet-threeHours-light-rest.png) | ![after](after/quiet-threeHours-light-rest.png) |
| readsLess, day, dark, rest | ![before](before/readsLess-day-dark-rest.png) | ![after](after/readsLess-day-dark-rest.png) |
| readsLess, day, light, rest | ![before](before/readsLess-day-light-rest.png) | ![after](after/readsLess-day-light-rest.png) |
| readsLess, hour, dark, rest | ![before](before/readsLess-hour-dark-rest.png) | ![after](after/readsLess-hour-dark-rest.png) |
| readsLess, hour, light, rest | ![before](before/readsLess-hour-light-rest.png) | ![after](after/readsLess-hour-light-rest.png) |
| readsLess, threeHours, dark, rest | ![before](before/readsLess-threeHours-dark-rest.png) | ![after](after/readsLess-threeHours-dark-rest.png) |
| readsLess, threeHours, light, rest | ![before](before/readsLess-threeHours-light-rest.png) | ![after](after/readsLess-threeHours-light-rest.png) |
| readsMore, day, dark, rest | ![before](before/readsMore-day-dark-rest.png) | ![after](after/readsMore-day-dark-rest.png) |
| readsMore, day, light, rest | ![before](before/readsMore-day-light-rest.png) | ![after](after/readsMore-day-light-rest.png) |
| readsMore, hour, dark, rest | ![before](before/readsMore-hour-dark-rest.png) | ![after](after/readsMore-hour-dark-rest.png) |
| readsMore, hour, light, rest | ![before](before/readsMore-hour-light-rest.png) | ![after](after/readsMore-hour-light-rest.png) |
| readsMore, threeHours, dark, rest | ![before](before/readsMore-threeHours-dark-rest.png) | ![after](after/readsMore-threeHours-dark-rest.png) |
| readsMore, threeHours, light, rest | ![before](before/readsMore-threeHours-light-rest.png) | ![after](after/readsMore-threeHours-light-rest.png) |
| readsOnly, day, dark, rest | ![before](before/readsOnly-day-dark-rest.png) | ![after](after/readsOnly-day-dark-rest.png) |
| readsOnly, day, light, rest | ![before](before/readsOnly-day-light-rest.png) | ![after](after/readsOnly-day-light-rest.png) |
| readsOnly, hour, dark, rest | ![before](before/readsOnly-hour-dark-rest.png) | ![after](after/readsOnly-hour-dark-rest.png) |
| readsOnly, hour, light, rest | ![before](before/readsOnly-hour-light-rest.png) | ![after](after/readsOnly-hour-light-rest.png) |
| readsOnly, threeHours, dark, rest | ![before](before/readsOnly-threeHours-dark-rest.png) | ![after](after/readsOnly-threeHours-dark-rest.png) |
| readsOnly, threeHours, light, rest | ![before](before/readsOnly-threeHours-light-rest.png) | ![after](after/readsOnly-threeHours-light-rest.png) |
| savesOnly, day, dark, rest | ![before](before/savesOnly-day-dark-rest.png) | ![after](after/savesOnly-day-dark-rest.png) |
| savesOnly, day, light, rest | ![before](before/savesOnly-day-light-rest.png) | ![after](after/savesOnly-day-light-rest.png) |
| savesOnly, hour, dark, rest | ![before](before/savesOnly-hour-dark-rest.png) | ![after](after/savesOnly-hour-dark-rest.png) |
| savesOnly, hour, light, rest | ![before](before/savesOnly-hour-light-rest.png) | ![after](after/savesOnly-hour-light-rest.png) |
| savesOnly, threeHours, dark, rest | ![before](before/savesOnly-threeHours-dark-rest.png) | ![after](after/savesOnly-threeHours-dark-rest.png) |
| savesOnly, threeHours, light, rest | ![before](before/savesOnly-threeHours-light-rest.png) | ![after](after/savesOnly-threeHours-light-rest.png) |
| sixDigit, day, dark, peak, hover | ![before](before/sixDigit-day-dark-peak-hover.png) | ![after](after/sixDigit-day-dark-peak-hover.png) |
| sixDigit, day, dark, rest | ![before](before/sixDigit-day-dark-rest.png) | ![after](after/sixDigit-day-dark-rest.png) |
| sixDigit, day, dark, zero, hover | ![before](before/sixDigit-day-dark-zero-hover.png) | ![after](after/sixDigit-day-dark-zero-hover.png) |
| sixDigit, day, light, peak, hover | ![before](before/sixDigit-day-light-peak-hover.png) | ![after](after/sixDigit-day-light-peak-hover.png) |
| sixDigit, day, light, rest | ![before](before/sixDigit-day-light-rest.png) | ![after](after/sixDigit-day-light-rest.png) |
| sixDigit, day, light, zero, hover | ![before](before/sixDigit-day-light-zero-hover.png) | ![after](after/sixDigit-day-light-zero-hover.png) |
| sixDigit, hour, dark, peak, hover | ![before](before/sixDigit-hour-dark-peak-hover.png) | ![after](after/sixDigit-hour-dark-peak-hover.png) |
| sixDigit, hour, dark, rest | ![before](before/sixDigit-hour-dark-rest.png) | ![after](after/sixDigit-hour-dark-rest.png) |
| sixDigit, hour, dark, zero, hover | ![before](before/sixDigit-hour-dark-zero-hover.png) | ![after](after/sixDigit-hour-dark-zero-hover.png) |
| sixDigit, hour, light, peak, hover | ![before](before/sixDigit-hour-light-peak-hover.png) | ![after](after/sixDigit-hour-light-peak-hover.png) |
| sixDigit, hour, light, rest | ![before](before/sixDigit-hour-light-rest.png) | ![after](after/sixDigit-hour-light-rest.png) |
| sixDigit, hour, light, zero, hover | ![before](before/sixDigit-hour-light-zero-hover.png) | ![after](after/sixDigit-hour-light-zero-hover.png) |
| sixDigit, threeHours, dark, peak, hover | ![before](before/sixDigit-threeHours-dark-peak-hover.png) | ![after](after/sixDigit-threeHours-dark-peak-hover.png) |
| sixDigit, threeHours, dark, rest | ![before](before/sixDigit-threeHours-dark-rest.png) | ![after](after/sixDigit-threeHours-dark-rest.png) |
| sixDigit, threeHours, dark, zero, hover | ![before](before/sixDigit-threeHours-dark-zero-hover.png) | ![after](after/sixDigit-threeHours-dark-zero-hover.png) |
| sixDigit, threeHours, light, peak, hover | ![before](before/sixDigit-threeHours-light-peak-hover.png) | ![after](after/sixDigit-threeHours-light-peak-hover.png) |
| sixDigit, threeHours, light, rest | ![before](before/sixDigit-threeHours-light-rest.png) | ![after](after/sixDigit-threeHours-light-rest.png) |
| sixDigit, threeHours, light, zero, hover | ![before](before/sixDigit-threeHours-light-zero-hover.png) | ![after](after/sixDigit-threeHours-light-zero-hover.png) |
| unavailable, day, dark, rest | ![before](before/unavailable-day-dark-rest.png) | ![after](after/unavailable-day-dark-rest.png) |
| unavailable, day, light, rest | ![before](before/unavailable-day-light-rest.png) | ![after](after/unavailable-day-light-rest.png) |
| unavailable, hour, dark, rest | ![before](before/unavailable-hour-dark-rest.png) | ![after](after/unavailable-hour-dark-rest.png) |
| unavailable, hour, light, rest | ![before](before/unavailable-hour-light-rest.png) | ![after](after/unavailable-hour-light-rest.png) |
| unavailable, threeHours, dark, rest | ![before](before/unavailable-threeHours-dark-rest.png) | ![after](after/unavailable-threeHours-dark-rest.png) |
| unavailable, threeHours, light, rest | ![before](before/unavailable-threeHours-light-rest.png) | ![after](after/unavailable-threeHours-light-rest.png) |
| unread, day, dark, rest | ![before](before/unread-day-dark-rest.png) | ![after](after/unread-day-dark-rest.png) |
| unread, day, light, rest | ![before](before/unread-day-light-rest.png) | ![after](after/unread-day-light-rest.png) |
| unread, hour, dark, rest | ![before](before/unread-hour-dark-rest.png) | ![after](after/unread-hour-dark-rest.png) |
| unread, hour, light, rest | ![before](before/unread-hour-light-rest.png) | ![after](after/unread-hour-light-rest.png) |
| unread, threeHours, dark, rest | ![before](before/unread-threeHours-dark-rest.png) | ![after](after/unread-threeHours-dark-rest.png) |
| unread, threeHours, light, rest | ![before](before/unread-threeHours-light-rest.png) | ![after](after/unread-threeHours-light-rest.png) |
