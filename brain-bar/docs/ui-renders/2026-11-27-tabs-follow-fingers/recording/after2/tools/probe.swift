// w308 latency floor: a borderless window whose layer flips black/white on every scroll event with a nonzero point delta,
// with implicit animations off (one CA commit per event, no SwiftUI). Recorded with the same SCK recorder, it measures
// the system's input-to-recorded-frame latency for the same synthesized scroll events, with almost no app work.
//   probe <x> <y_top> <size> <seconds>
import Cocoa
import QuartzCore
let a = CommandLine.arguments
let x = Double(a[1])!, yTop = Double(a[2])!, size = Double(a[3])!, seconds = Double(a[4])!
final class V: NSView {
    var on = false
    override var acceptsFirstResponder: Bool { true }
    override func scrollWheel(with e: NSEvent) {
        guard e.scrollingDeltaX != 0 else { return }
        on.toggle()
        CATransaction.begin(); CATransaction.setDisableActions(true)
        layer?.backgroundColor = (on ? NSColor.white : NSColor.black).cgColor
        CATransaction.commit()
        print(String(format: "%.6f flip %d", CACurrentMediaTime(), on ? 1 : 0)); fflush(stdout)
    }
}
let app = NSApplication.shared
app.setActivationPolicy(.regular)
let screenH = NSScreen.screens[0].frame.height
let w = NSWindow(contentRect: NSRect(x: x, y: screenH - yTop - size, width: size, height: size), styleMask: [.borderless], backing: .buffered, defer: false)
w.level = .floating
let v = V(frame: NSRect(x: 0, y: 0, width: size, height: size))
v.wantsLayer = true; v.layer?.backgroundColor = NSColor.black.cgColor
w.contentView = v
w.makeKeyAndOrderFront(nil); w.makeFirstResponder(v)
app.activate(ignoringOtherApps: true)
DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { exit(0) }
app.run()
