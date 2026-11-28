import Cocoa
let app = NSApplication.shared
app.setActivationPolicy(.regular)
let w = NSWindow(contentRect: NSRect(x: 100, y: 300, width: 400, height: 300), styleMask: [.titled], backing: .buffered, defer: false)
w.title = "evmon"; w.makeKeyAndOrderFront(nil)
NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { e in
    print(String(format: "dx=%.3f dy=%.3f precise=%d phase=%lu mom=%lu deltaX=%.3f", e.scrollingDeltaX, e.scrollingDeltaY, e.hasPreciseScrollingDeltas ? 1 : 0, e.phase.rawValue, e.momentumPhase.rawValue, e.deltaX)); fflush(stdout)
    return e
}
DispatchQueue.main.asyncAfter(deadline: .now() + 6) { exit(0) }
app.activate(ignoringOtherApps: true)
app.run()
