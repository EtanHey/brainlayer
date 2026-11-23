# New-design wake render proof

Before source: 4349370f0b8250ef45e3598d223cb0924a6a3145
After source: b12151b4928c18ee1ebe864708355c1a232c5f53

16 pictures: New Overview and actual NDMenuBuilder rows, after wake and at the wake timeout, dark and light.
All inputs are invented. The same BrainBarWakeRefreshTests and NDRenderHarness rendered both arms offscreen.
The before arm retained the new tests and substituted origin/main production Sources.
Each picture was opened and inspected by the implementer.

Before: both states read Not confirmed; timeout has no wake reason or amber mark.
After: pending wake reads Checking after sleep; timeout gives the missing post-wake reading reason with amber.
The tests also cover fresh post-wake confirmation and a retained red fault.
This is source/render proof, not installed-app or live acceptance.
