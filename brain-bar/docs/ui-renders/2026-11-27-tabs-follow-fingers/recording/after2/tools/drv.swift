// w300 gesture driver: synthesized trackpad scroll phases, clicks and a pill drag through the real NSEvent path
// (CGEvent posted at the HID tap), with every input logged on the CACurrentMediaTime clock (mach host time, the same
// clock AVFoundation stamps screen frames with), plus a 2 ms poller of the capture window's bounds.
//
//   drv place <x> <y>          move the BrainBarCapture window (AX) so its top-left is at (x, y) in points
//   drv bounds                 print the window bounds
//   drv run <log> [steps]      run the gesture script; steps = comma list of 1..9 (default all; 9 added by w308)
import Cocoa
import ApplicationServices
import QuartzCore

let owner = "BrainBarCapture"

func windowInfo() -> (pid: pid_t, rect: CGRect)? {
    for w in CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] ?? [] {
        guard (w[kCGWindowOwnerName as String] as? String) == owner, (w[kCGWindowLayer as String] as? Int) == 0,
              let b = w[kCGWindowBounds as String] as? [String: Any],
              let r = CGRect(dictionaryRepresentation: b as CFDictionary) else { continue }
        return ((w[kCGWindowOwnerPID as String] as? Int).map { pid_t($0) } ?? 0, r)
    }
    return nil
}

func place(_ x: Double, _ y: Double) {
    guard let (pid, _) = windowInfo() else { print("no window"); exit(1) }
    let app = AXUIElementCreateApplication(pid)
    var value: CFTypeRef?
    AXUIElementCopyAttributeValue(app, kAXWindowsAttribute as CFString, &value)
    guard let windows = value as? [AXUIElement], let w = windows.first else { print("no AX window"); exit(1) }
    var p = CGPoint(x: x, y: y)
    let v = AXValueCreate(.cgPoint, &p)!
    let err = AXUIElementSetAttributeValue(w, kAXPositionAttribute as CFString, v)
    print("place err=\(err.rawValue)")
}

var logLines: [String] = []
let logLock = NSLock()
func log(_ s: String) {
    let line = String(format: "%.6f ", CACurrentMediaTime()) + s
    logLock.lock(); logLines.append(line); logLock.unlock()
}

/// Sleep until an absolute CACurrentMediaTime deadline (coarse sleep, then a short spin).
func waitUntil(_ t: Double) {
    while true {
        let now = CACurrentMediaTime()
        if now >= t { return }
        if t - now > 0.002 { usleep(useconds_t((t - now - 0.0015) * 1e6)) }
    }
}

let src = CGEventSource(stateID: .hidSystemState)

func scroll(dx: Double, phase: Int64, momentum: Int64 = 0) {
    let e = CGEvent(scrollWheelEvent2Source: src, units: .pixel, wheelCount: 2, wheel1: 0, wheel2: Int32(dx.rounded()), wheel3: 0)!
    e.setIntegerValueField(.scrollWheelEventIsContinuous, value: 1)
    e.setIntegerValueField(.scrollWheelEventScrollPhase, value: phase)
    e.setIntegerValueField(.scrollWheelEventMomentumPhase, value: momentum)
    // Only the point delta (what NSEvent.scrollingDeltaX reads for a continuous event). Setting the fixed-point line
    // delta makes the system recompute the point delta from it (x8), measured with evmon.swift.
    e.setDoubleValueField(.scrollWheelEventPointDeltaAxis2, value: dx)
    e.post(tap: .cghidEventTap)
}

var origin = CGPoint.zero  // window top-left, points

func pt(_ wx: Double, _ wy: Double) -> CGPoint { CGPoint(x: origin.x + wx, y: origin.y + wy) }

func moveMouse(_ p: CGPoint) {
    log("cursor \(p.x - origin.x) \(p.y - origin.y)")
    CGWarpMouseCursorPosition(p)
    CGEvent(mouseEventSource: src, mouseType: .mouseMoved, mouseCursorPosition: p, mouseButton: .left)?.post(tap: .cghidEventTap)
}

func mouse(_ type: CGEventType, _ p: CGPoint) {
    if type == .leftMouseDragged { CGWarpMouseCursorPosition(p); log("cursor \(p.x - origin.x) \(p.y - origin.y)") }
    let e = CGEvent(mouseEventSource: src, mouseType: type, mouseCursorPosition: p, mouseButton: .left)!
    e.setIntegerValueField(.mouseEventClickState, value: 1)
    e.post(tap: .cghidEventTap)
}

let frame = 1.0 / 120.0
// Page header: right of the headline, inside the pages, clear of the tab bar.
let header = (x: 760.0, y: 130.0)

/// A swipe: `steps` deltas, one per 120 Hz frame, starting at t0. Returns the time after the last event.
@discardableResult
func swipe(_ name: String, deltas: [Double], at t0: Double, holdBeforeEnd: Double = 0, momentum: [Double] = []) -> Double {
    moveMouse(pt(header.x, header.y))
    var t = t0
    waitUntil(t)
    log("input-start \(name)")
    // NSEvent truncates a synthesized point delta to whole points: send whole points, carrying the remainder, so the
    // total travel is exact.
    var acc = 0.0, sent = 0.0
    func whole(_ d: Double) -> Double { acc += d; let k = acc.rounded() - sent; sent += k; return k }
    for (i, d) in deltas.enumerated() {
        waitUntil(t)
        let k = whole(d)
        if k != 0 && sent == k { log("first-move \(name)") }
        scroll(dx: k, phase: i == 0 ? 1 : 2)
        t += frame
    }
    log("travel \(name) \(sent)")
    if holdBeforeEnd > 0 { log("hold \(name)"); t += holdBeforeEnd }
    waitUntil(t)
    scroll(dx: 0, phase: 4)
    log("released \(name)")
    t += frame
    for (i, d) in momentum.enumerated() {
        waitUntil(t)
        scroll(dx: d.rounded(), phase: 0, momentum: i == 0 ? 1 : (i == momentum.count - 1 ? 3 : 2))
        t += frame
    }
    if !momentum.isEmpty { log("momentum-end \(name)") }
    return t
}

/// Ease-in-out finger travel of `total` pt over `seconds` (velocity 0 at both ends: no flick on release).
func smooth(total: Double, seconds: Double) -> [Double] {
    let n = Int(seconds * 120)
    var out: [Double] = []
    var prev = 0.0
    for i in 1...n {
        let u = Double(i) / Double(n)
        let s = u * u * (3 - 2 * u)
        out.append(total * (s - prev)); prev = s
    }
    return out
}

func flick(_ sign: Double) -> (deltas: [Double], momentum: [Double]) {
    // 10 steps over 83 ms ramping to 30 pt/frame (≈3600 pt/s at release), then decaying momentum.
    let d = (1...10).map { sign * Double(min(30, 6 + $0 * 3)) }
    var m: [Double] = []
    var v = 28.0
    while v > 0.5 { m.append(sign * v); v *= 0.88 }
    return (d, m)
}

func click(_ name: String, wx: Double, wy: Double, at t: Double) {
    waitUntil(t)
    let p = pt(wx, wy)
    moveMouse(p)
    waitUntil(t + 0.03)
    log("input-start \(name)")
    mouse(.leftMouseDown, p)
    waitUntil(t + 0.03 + 0.06)
    mouse(.leftMouseUp, p)
    log("released \(name)")
}

func runScript(_ path: String, steps: Set<Int>, tabs: [String: Double], tabY: Double) {
    // window poller
    var polling = true
    let poller = Thread {
        var last = CGRect.null
        while polling {
            if let (_, r) = windowInfo(), r != last { last = r; log("window \(r.origin.x) \(r.origin.y) \(r.width) \(r.height)") }
            usleep(2000)
        }
    }
    poller.start()
    if let app = NSRunningApplication.runningApplications(withBundleIdentifier: "com.brainlayer.brainbar.capture").first {
        app.activate()
    }
    var t = CACurrentMediaTime() + 0.3
    // make the window key the way a user would: one click on the empty page header (activate() alone does not take
    // focus from another app on macOS 14+), so every gesture runs against the active window's appearance
    click("activate-window", wx: header.x, wy: header.y, at: t)
    t += 1.0
    // sync marker: a cursor jump the recording can see
    waitUntil(t); moveMouse(pt(header.x, header.y)); waitUntil(t + 0.3)
    log("sync-cursor-jump"); moveMouse(pt(header.x + 60, header.y)); t += 0.8
    moveMouse(pt(header.x, header.y))
    t += 0.5

    func setup(_ tab: String) { log("setup \(tab)"); click("setup-\(tab)", wx: tabs[tab]!, wy: tabY, at: t); t += 1.2; moveMouse(pt(header.x, header.y)) }

    if steps.contains(1) {
        t = swipe("g1a-slow-overview-backups", deltas: smooth(total: -600, seconds: 1.5), at: t) + 1.2
        t = swipe("g1b-slow-backups-jobs", deltas: smooth(total: -600, seconds: 1.5), at: t) + 1.2
    }
    if steps.contains(2) {
        let f = flick(+1)
        t = swipe("g2-flick-jobs-backups", deltas: f.deltas, at: t, momentum: f.momentum) + 1.2
    }
    if steps.contains(3) {
        setup("overview")
        click("g3-click-far-overview-jobs", wx: tabs["jobs"]!, wy: tabY, at: t); t += 1.0
        moveMouse(pt(header.x, header.y)); t += 0.2
    }
    if steps.contains(4) {
        setup("overview")
        click("g4a-click-backups", wx: tabs["backups"]!, wy: tabY, at: t)
        click("g4b-click-jobs-120ms", wx: tabs["jobs"]!, wy: tabY, at: t + 0.12)  // its down 120 ms after the first down
        t += 1.2
        moveMouse(pt(header.x, header.y)); t += 0.2
    }
    if steps.contains(5) {
        setup("overview")
        let a = pt(tabs["overview"]!, tabY), b = pt(tabs["backups"]!, tabY)
        waitUntil(t); moveMouse(a); waitUntil(t + 0.05)
        log("input-start g5-pill-drag-overview-backups")
        mouse(.leftMouseDown, a)
        var tt = t + 0.05 + 0.15
        let n = 72  // 0.6 s at 120 Hz
        for i in 1...n {
            waitUntil(tt)
            if i == 1 { log("first-move g5-pill-drag-overview-backups") }
            let u = Double(i) / Double(n), s = u * u * (3 - 2 * u)
            mouse(.leftMouseDragged, CGPoint(x: a.x + (b.x - a.x) * s, y: a.y))
            tt += frame
        }
        tt += 0.1; waitUntil(tt)
        mouse(.leftMouseUp, b)
        log("released g5-pill-drag-overview-backups")
        t = tt + 1.2
        moveMouse(pt(header.x, header.y)); t += 0.2
    }
    if steps.contains(6) {
        setup("backups")
        // 0.45 of the 900 pt window width over 0.6 s, held 1 s, released with no velocity
        t = swipe("g6-held-half-backups-jobs", deltas: smooth(total: -405, seconds: 0.6), at: t, holdBeforeEnd: 1.0) + 1.2
    }
    if steps.contains(7) {
        setup("jobs")
        let f = flick(-1)
        t = swipe("g7-flick-past-last-jobs", deltas: f.deltas, at: t, momentum: f.momentum) + 1.2
    }
    if steps.contains(8) {
        setup("overview")
        let f = flick(-1)
        swipe("g8a-double-flick-1", deltas: f.deltas, at: t, momentum: Array(f.momentum.prefix(30)))
        t = swipe("g8b-double-flick-2", deltas: f.deltas, at: t + 0.5, momentum: f.momentum) + 1.5
    }
    if steps.contains(9) {
        // w308: a far click Overview→Jobs (as g3), then a fresh flick Jobs→Backups (as g2) whose first event goes out
        // G9_DELAY s after the click's mouse-up = the far click's measured time to rest + 100 ms. The pointer leaves the
        // Jobs tab for the page header 50 ms before the flick (swipes start on the header, as every other swipe).
        setup("overview")
        click("g9a-click-far-overview-jobs", wx: tabs["jobs"]!, wy: tabY, at: t)
        let up = t + 0.03 + 0.06
        let delay = Double(ProcessInfo.processInfo.environment["G9_DELAY"] ?? "0.58")!
        log("g9-delay \(delay)")
        waitUntil(up + delay - 0.05)
        let f = flick(+1)
        t = swipe("g9b-flick-after-rest", deltas: f.deltas, at: up + delay, momentum: f.momentum) + 1.2
    }
    waitUntil(t)
    log("script-end")
    polling = false
    usleep(10000)
    try! (logLines.joined(separator: "\n") + "\n").write(toFile: path, atomically: true, encoding: .utf8)
}

let a = CommandLine.arguments
switch a.count > 1 ? a[1] : "" {
case "place": place(Double(a[2])!, Double(a[3])!)
case "probe":
    // probe <dx per event> <events>: began + changed, hold (screenshot), then ended
    guard let (_, r) = windowInfo() else { exit(1) }
    origin = r.origin
    moveMouse(pt(header.x, header.y)); usleep(300000)
    let dx = Double(a[2])!, n = Int(a[3])!
    var t = CACurrentMediaTime() + 0.05
    for i in 0..<n { waitUntil(t); scroll(dx: dx, phase: i == 0 ? 1 : 2); t += frame }
    usleep(500000)
    let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
    p.arguments = ["-x", "-R", "892,33,620,120", a[4]]; try! p.run(); p.waitUntilExit()
    scroll(dx: 0, phase: 4); usleep(800000)
case "bounds": print(windowInfo().map { "\($0.pid) \($0.rect)" } ?? "none")
case "run":
    guard let (_, r) = windowInfo() else { print("no window"); exit(1) }
    origin = r.origin
    // tab centres (window points), passed as env: TABS="overview:370,backups:464,jobs:543" TABY=26
    let env = ProcessInfo.processInfo.environment
    var tabs: [String: Double] = [:]
    for kv in (env["TABS"] ?? "").split(separator: ",") {
        let p = kv.split(separator: ":"); tabs[String(p[0])] = Double(p[1])!
    }
    let steps = Set((a.count > 3 ? a[3] : "1,2,3,4,5,6,7,8,9").split(separator: ",").map { Int($0)! })
    runScript(a[2], steps: steps, tabs: tabs, tabY: Double(env["TABY"] ?? "26")!)
default: print("usage: drv place x y | bounds | run log [steps]")
}
