// w308: post N single scroll events (point delta 9, phase began then ended, as the flick's first event) at the probe,
// 150 ms apart, logging each post time on CACurrentMediaTime.   pdrv <x> <y> <n>
import Cocoa
import QuartzCore
let a = CommandLine.arguments
let p = CGPoint(x: Double(a[1])!, y: Double(a[2])!); let n = Int(a[3])!
let src = CGEventSource(stateID: .hidSystemState)
CGWarpMouseCursorPosition(p)
CGEvent(mouseEventSource: src, mouseType: .mouseMoved, mouseCursorPosition: p, mouseButton: .left)?.post(tap: .cghidEventTap)
usleep(300000)
func scroll(_ dx: Double, _ phase: Int64) {
    let e = CGEvent(scrollWheelEvent2Source: src, units: .pixel, wheelCount: 2, wheel1: 0, wheel2: Int32(dx), wheel3: 0)!
    e.setIntegerValueField(.scrollWheelEventIsContinuous, value: 1)
    e.setIntegerValueField(.scrollWheelEventScrollPhase, value: phase)
    e.setDoubleValueField(.scrollWheelEventPointDeltaAxis2, value: dx)
    e.post(tap: .cghidEventTap)
}
var t = CACurrentMediaTime() + 0.1
for i in 0..<n {
    while CACurrentMediaTime() < t {}
    print(String(format: "%.6f post %d", CACurrentMediaTime(), i)); fflush(stdout)
    scroll(9, 1)
    usleep(20000); scroll(0, 4)
    t += 0.15 + Double(i % 7) / 120 * 0.37   // spread the phase against vsync
}
