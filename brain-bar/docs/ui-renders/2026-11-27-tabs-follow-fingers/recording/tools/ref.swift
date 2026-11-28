// w300 reference strip: a small borderless window (ignores the mouse) whose 4 pt bar moves 4 pt per display frame,
// as a linear Core Animation animation the render server runs (no per-frame commits from this process). Placed inside the recorded region, below BrainBar's window, it changes
// every frame, so the recorder must deliver every compositor frame: its step between two recorded frames (1 = no frame
// missing) tells recorder/compositor drops apart from frames the app did not draw.
//   ref <x> <y_top> <width> <seconds>        (points, global top-left origin)
import Cocoa
import QuartzCore

let a = CommandLine.arguments
let x = Double(a[1])!, yTop = Double(a[2])!, width = Double(a[3])!, seconds = Double(a[4])!
let height = 14.0
let slots = Int(width / 4)

final class V: NSView {
    let bar = CALayer()
    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.backgroundColor = NSColor.black.cgColor
        bar.backgroundColor = NSColor.white.cgColor
        bar.anchorPoint = .zero
        bar.frame = CGRect(x: 0, y: 0, width: 4, height: height)
        layer?.addSublayer(bar)
    }
    required init?(coder: NSCoder) { fatalError() }
    /// A render-server animation: the bar steps 4 pt per 120 Hz frame with no per-frame work in this process.
    func start() {
        let anim = CABasicAnimation(keyPath: "position.x")
        anim.fromValue = 0
        anim.toValue = Double(slots) * 4
        anim.duration = Double(slots) / 120
        anim.repeatCount = .infinity
        anim.timingFunction = CAMediaTimingFunction(name: .linear)
        anim.preferredFrameRateRange = CAFrameRateRange(minimum: 120, maximum: 120, preferred: 120)
        bar.add(anim, forKey: "run")
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let screenH = NSScreen.screens[0].frame.height
let w = NSWindow(contentRect: NSRect(x: x, y: screenH - yTop - height, width: width, height: height),
                 styleMask: [.borderless], backing: .buffered, defer: false)
w.ignoresMouseEvents = true
w.level = .floating
let v = V(frame: NSRect(x: 0, y: 0, width: width, height: height))
w.contentView = v
w.orderFrontRegardless()
v.start()
DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { exit(0) }
app.run()
