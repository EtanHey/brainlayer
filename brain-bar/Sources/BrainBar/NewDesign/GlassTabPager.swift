import AppKit
import QuartzCore
import SwiftUI

// A tab bar whose glass selection and pages follow the fingers (tab-swipe spec, agreed by BrainBar and VoiceBar,
// 27 Nov 2026). Self-contained on purpose: no app types, so this one file can be handed to VoiceBar as it is.
//
// 1. Live tracking: the pill and the content follow the fingers 1:1; the pill's frame (x, width) interpolates between
//    neighbouring slots by `progress`.
// 2. Release: one spring settles both to the tab picked by position + projected velocity; the pill ends at the
//    destination slot's resting size.
// 3. Click: the same spring, from the presented position; long press and drag take the same path.
// 4. No layout shift: fixed slot widths measured at the selected weight; the pill is an overlay; a selection never
//    re-lays out a label.
// 5. Interruptible: a new gesture or click retargets from the presented position, carrying velocity.
// 6. Reduce Motion: the content cross-fades; the pill moves without overshoot.

/// The one spring every settle uses: critically damped (damping ratio 1.0), response 0.30 s, velocity carried in.
/// Solved exactly per step, so the result does not depend on the frame rate.
struct GlassTabSpring: Equatable {
    static let standard = GlassTabSpring(response: 0.30)

    /// Seconds for one undamped period (SwiftUI's `response`).
    let response: Double
    var omega: Double { 2 * .pi / response }

    /// The offset from the target and the velocity after `dt` seconds (critically damped: no oscillation, and no
    /// overshoot at all when the incoming velocity is zero).
    func advance(offset x0: Double, velocity v0: Double, by dt: Double) -> (offset: Double, velocity: Double) {
        let w = omega
        let c2 = v0 + w * x0
        let decay = exp(-w * dt)
        return ((x0 + c2 * dt) * decay, (v0 - w * c2 * dt) * decay)
    }
}

/// The pager: one continuous `progress` (0 ... count-1) that the tab bar's pill and the pages both draw. It is moved by
/// tracking (fingers, a dragged pill) and by the spring (release, click, keyboard, an outside change). Every input
/// carries its own timestamp (seconds on the `CACurrentMediaTime` clock, as `NSEvent.timestamp` and display links are),
/// so tests drive it deterministically.
@MainActor final class GlassTabPager: ObservableObject {
    enum Phase: Equatable { case resting, tracking, settling }

    /// UIScrollView's normal deceleration rate (0.998 per ms) projects a release velocity this many seconds ahead.
    static let projection = 0.998 / (1 - 0.998) / 1000
    /// Velocity is measured over the last samples within this window; a pause longer than it means no flick.
    static let velocityWindow = 0.1
    /// A settle never starts faster than this (tabs per second): one frame at 120 Hz moves under 1/3 of a tab. (A click
    /// on a tab two away peaks near 15 on its own, so a retarget mid-settle keeps its real velocity.)
    static let maximumSpeed = 20.0
    /// Rest: closer than this (tabs) and slower than `restSpeed` snaps exactly to the target.
    static let restDistance = 1e-3
    static let restSpeed = 0.02
    /// Rubber band past the first and last tab (Apple's constant; one tab is the dimension).
    static let rubberBand = 0.55

    let count: Int
    let spring: GlassTabSpring
    /// The presented position: what the pill and the pages draw now.
    @Published private(set) var progress: Double
    /// The selected tab: the settle's target, and what VoiceOver and the app read.
    @Published private(set) var selection: Int
    private(set) var phase: Phase = .resting
    /// Tabs per second.
    private(set) var velocity: Double = 0

    /// Reduce Motion: no velocity is carried into a settle (so the pill never overshoots), and no rubber band.
    var reduceMotion = false
    /// Whether frames are being drawn now (a visible window). When not, a change of selection lands at once.
    var isLive: () -> Bool = { false }
    /// A selection the user made here (click, release, keyboard, dragged pill), delivered when the pill and pages come
    /// to REST on it, never mid-motion: whatever the app does on a selection (publish, rebuild, refit) can never cost
    /// a frame of the motion. Not called for `follow` (the app already has that selection).
    var onSelect: ((Int) -> Void)?
    /// The selection the app last has (reported through `onSelect`, or given by `follow`).
    private(set) var committed: Int
    /// The clock: true while the spring needs frames.
    var onNeedsFrames: ((Bool) -> Void)?
    /// The pill and the pages came to rest on `selection` (after a settle, a jump or a release that needed none). The
    /// app moves anything that must not move mid-swipe here (BrainBar: the window's height).
    var onRest: (() -> Void)?
    /// The pill and the pages leave rest (a swipe or a pressed pill takes hold, or a settle starts). The app finishes
    /// anything still moving from the last rest here (BrainBar: a running window resize).
    var onMove: (() -> Void)?
    /// The tab the current motion started from (the selection when the pager last left rest).
    private(set) var motionOrigin: Int

    private var raw: Double = 0
    private var origin = 0
    private var samples: [(time: Double, raw: Double)] = []
    private var lastTick: Double = 0

    init(count: Int, selection: Int = 0, spring: GlassTabSpring = .standard) {
        precondition(count > 0)
        self.count = count
        self.spring = spring
        let index = min(max(selection, 0), count - 1)
        self.selection = index
        committed = index
        motionOrigin = index
        progress = Double(index)
    }

    var lastIndex: Int { count - 1 }
    var needsFrames: Bool { phase == .settling }

    // MARK: Tracking (two fingers, or a dragged pill)

    /// A gesture takes hold: from the presented position, interrupting any settle.
    func grab(at time: Double) {
        if phase == .resting { motionOrigin = selection }
        raw = unrubber(progress)
        origin = selection
        samples = [(time, raw)]
        velocity = 0
        setPhase(.tracking)
    }

    /// The fingers moved by `delta` tabs.
    func drag(by delta: Double, at time: Double) {
        if phase != .tracking { grab(at: time) }
        raw += delta
        record(time)
    }

    /// The fingers are at `position` (in tabs): a dragged pill.
    func drag(to position: Double, at time: Double) {
        if phase != .tracking { grab(at: time) }
        raw = position
        record(time)
    }

    /// The fingers lifted: settle on the tab at the presented position plus the projected velocity. `oneStep` limits a
    /// trackpad flick to the neighbouring tab; a dragged pill may land on any tab.
    func release(at time: Double, oneStep: Bool = true) {
        guard phase == .tracking else { return }
        let v = measuredVelocity(at: time)
        var target = Int((progress + v * Self.projection).rounded())
        if oneStep { target = min(max(target, origin - 1), origin + 1) }
        target = min(max(target, 0), lastIndex)
        settle(to: target, velocity: v, at: time)
    }

    // MARK: Selection (click, keyboard, outside)

    /// The user picked tab `index`: the spring runs from the presented position, carrying the present velocity.
    func select(_ index: Int, at time: Double) {
        let index = min(max(index, 0), lastIndex)
        let v = phase == .tracking ? measuredVelocity(at: time) : velocity
        settle(to: index, velocity: v, at: time)
    }

    /// Something else changed the selection (a menu item, the app's state). Animated only while frames are drawn.
    func follow(_ index: Int, at time: Double = CACurrentMediaTime()) {
        let index = min(max(index, 0), lastIndex)
        guard index != selection else { return }
        committed = index
        if isLive() {
            settle(to: index, velocity: velocity, at: time)
        } else {
            jump(to: index)
        }
    }

    /// Land on `index` at once.
    func jump(to index: Int) {
        let index = min(max(index, 0), lastIndex)
        if selection != index { selection = index }
        progress = Double(index)
        velocity = 0
        samples = []
        setPhase(.resting)
    }

    /// Renders only: draw the pager at `position` without changing the selection.
    func present(_ position: Double) {
        progress = position
        velocity = 0
        setPhase(.resting)
    }

    // MARK: The clock

    /// One frame: advance the spring to `time`.
    func tick(at time: Double) {
        guard phase == .settling else { return }
        // A clock that is behind the input (or a frame after a long stall) never runs the spring backwards or leaps.
        let dt = min(max(time - lastTick, 0), 0.1)
        lastTick = max(lastTick, time)
        let target = Double(selection)
        let next = spring.advance(offset: progress - target, velocity: velocity, by: dt)
        if abs(next.offset) < Self.restDistance, abs(next.velocity) < Self.restSpeed {
            progress = target
            velocity = 0
            setPhase(.resting)
        } else {
            progress = target + next.offset
            velocity = next.velocity
        }
    }

    // MARK: Private

    private func settle(to index: Int, velocity v: Double, at time: Double) {
        if phase == .resting { motionOrigin = selection }
        samples = []
        velocity = reduceMotion ? 0 : min(max(v, -Self.maximumSpeed), Self.maximumSpeed)
        // At the first and last tab there is nothing past the target to show: carried-in velocity toward it is capped
        // at ω·|offset|, the most a critically damped spring can take without crossing the target.
        if index == 0 || index == lastIndex {
            let offset = progress - Double(index)
            let limit = spring.omega * abs(offset)
            if offset != 0, velocity * offset < 0, abs(velocity) > limit { velocity = -limit * offset.sign.unit }
        }
        lastTick = time
        if selection != index { selection = index }
        if progress == Double(index), velocity == 0 {
            setPhase(.resting)
        } else if isLive() {
            setPhase(.settling)
        } else {
            jump(to: index)
        }
    }

    private func record(_ time: Double) {
        let shown = rubber(raw)
        if shown != progress { progress = shown }
        // Two inputs at one instant (a take-hold and its first step) are one sample: the later position.
        if let last = samples.last, last.time == time { samples.removeLast() }
        samples.append((time, raw))
        samples.removeAll { time - $0.time > Self.velocityWindow }
    }

    /// The least-squares slope of the recent samples (tabs per second); none when the fingers paused before lifting.
    private func measuredVelocity(at time: Double) -> Double {
        let recent = samples.filter { time - $0.time <= Self.velocityWindow }
        guard recent.count >= 2, let first = recent.first, let last = recent.last, last.time - first.time > 0.004 else { return 0 }
        let n = Double(recent.count)
        let meanT = recent.reduce(0) { $0 + $1.time } / n
        let meanX = recent.reduce(0) { $0 + $1.raw } / n
        var covariance = 0.0, variance = 0.0
        for sample in recent {
            covariance += (sample.time - meanT) * (sample.raw - meanX)
            variance += (sample.time - meanT) * (sample.time - meanT)
        }
        return variance > 0 ? covariance / variance : 0
    }

    private func setPhase(_ next: Phase) {
        let before = needsFrames
        let rests = next == .resting && phase != .resting
        let moves = next != .resting && phase == .resting
        phase = next
        if needsFrames != before { onNeedsFrames?(needsFrames) }
        if moves { onMove?() }
        if next == .resting, selection != committed {
            committed = selection
            onSelect?(selection)
        }
        if rests { onRest?() }
    }

    /// Past an end the presented position moves less the further the fingers go.
    private func rubber(_ x: Double) -> Double {
        let top = Double(lastIndex)
        if reduceMotion { return min(max(x, 0), top) }
        func band(_ over: Double) -> Double { 1 - 1 / (over * Self.rubberBand + 1) }
        if x < 0 { return -band(-x) }
        if x > top { return top + band(x - top) }
        return x
    }

    private func unrubber(_ shown: Double) -> Double {
        let top = Double(lastIndex)
        func inverse(_ f: Double) -> Double { (1 / (1 - min(f, 0.999)) - 1) / Self.rubberBand }
        if shown < 0 { return -inverse(-shown) }
        if shown > top { return top + inverse(shown - top) }
        return shown
    }
}

private extension FloatingPointSign {
    var unit: Double { self == .minus ? -1 : 1 }
}

// MARK: - The tab bar

/// One tab: its title, an optional count drawn in a small pill after it, and what VoiceOver says.
struct GlassTab: Equatable {
    var title: String
    var count = 0
    var spokenLabel: String
}

/// The bar's measurements. Every slot's width is its title (one weight for all labels) plus its count, so a selection
/// never changes a width (spec 4).
struct GlassTabMetrics: Equatable {
    var fontSize: CGFloat = 13
    var countFontSize: CGFloat = 11
    var padding: CGFloat = 14
    var height: CGFloat = 28
    /// The pill's inset from the track.
    var inset: CGFloat = 2
    var countGap: CGFloat = 6

    /// One weight for every label, selected or not: the pill splits labels on colour only, never mid-word on weight.
    var titleFont: NSFont { .systemFont(ofSize: fontSize, weight: .medium) }
    var countFont: NSFont { .monospacedDigitSystemFont(ofSize: countFontSize, weight: .semibold) }

    func countWidth(_ count: Int) -> CGFloat {
        guard count > 0 else { return 0 }
        let numeral = NSAttributedString(string: String(count), attributes: [.font: countFont]).size().width
        return max(16, ceil(numeral) + 8)
    }

    /// The label's width, count included.
    func labelWidth(_ tab: GlassTab) -> CGFloat {
        let title = NSAttributedString(string: tab.title, attributes: [.font: titleFont]).size().width
        return ceil(title + (tab.count > 0 ? countGap + countWidth(tab.count) : 0))
    }

    func slotWidths(_ tabs: [GlassTab]) -> [CGFloat] { tabs.map { labelWidth($0) + 2 * padding } }

    /// Each slot's frame along the bar (x from the bar's leading edge).
    func slotFrames(_ tabs: [GlassTab]) -> [CGRect] {
        var x: CGFloat = 0
        return slotWidths(tabs).map { width in
            defer { x += width }
            return CGRect(x: x, y: 0, width: width, height: height)
        }
    }

    /// The pill at `progress`: between neighbouring slots, x and width interpolate; past an end it stays on that end.
    func pillFrame(_ tabs: [GlassTab], progress: Double) -> CGRect {
        let frames = slotFrames(tabs)
        guard !frames.isEmpty else { return .zero }
        let p = min(max(progress, 0), Double(frames.count - 1))
        let lower = Int(p.rounded(.down))
        let upper = min(lower + 1, frames.count - 1)
        let t = CGFloat(p - Double(lower))
        let a = frames[lower], b = frames[upper]
        let x = a.minX + (b.minX - a.minX) * t
        let width = a.width + (b.width - a.width) * t
        return CGRect(x: x, y: 0, width: width, height: height).insetBy(dx: inset, dy: inset)
    }

    /// The position (in tabs) whose pill centre is at `x`, for a dragged pill.
    func progress(forPillCentre x: CGFloat, _ tabs: [GlassTab]) -> Double {
        let centres = slotFrames(tabs).map(\.midX)
        guard let first = centres.first, let last = centres.last else { return 0 }
        if x <= first { return Double((x - first) / max(slotWidths(tabs).first ?? 1, 1)) }
        if x >= last { return Double(centres.count - 1) + Double((x - last) / max(slotWidths(tabs).last ?? 1, 1)) }
        for i in 0..<(centres.count - 1) where x <= centres[i + 1] {
            return Double(i) + Double((x - centres[i]) / (centres[i + 1] - centres[i]))
        }
        return Double(centres.count - 1)
    }

    /// The slot under `x`, if any.
    func slot(at x: CGFloat, _ tabs: [GlassTab]) -> Int? {
        slotFrames(tabs).firstIndex { x >= $0.minX && x < $0.maxX }
    }
}

/// Colours the bar draws with; the host passes its own.
struct GlassTabColors {
    var text: Color = .secondary
    var selectedText: Color = .primary
    var track: Color = Color.primary.opacity(0.06)
    /// The pill's fill: the solid pill before macOS 26, and the tint inside the glass on 26 (so the selection reads
    /// clearly in dark and light, not only as refraction).
    var pill: Color = Color(nsColor: .controlBackgroundColor)
    /// A hairline around the pill, so its edge holds against the track in both appearances.
    var pillRim: Color = Color.primary.opacity(0.12)
    var countFill: Color = Color.primary.opacity(0.12)
    var countText: Color = .primary
}

/// The bar: fixed slots, a glass pill overlay at the pager's progress, and the selected colour drawn only inside the
/// pill (two label layers at one weight, split by the pill's shape), so a moving selection never changes a label's
/// weight or place. Click, press-and-drag of the pill, keyboard arrows and VoiceOver all go through the pager.
/// Press feedback: the pressed pill swells to 1.03 and settles back on the same spring; a pressed tab's label dims.
struct GlassTabBar: View {
    @ObservedObject var pager: GlassTabPager
    let tabs: [GlassTab]
    var accessibilityLabel: String
    var metrics = GlassTabMetrics()
    var colors = GlassTabColors()
    /// Time source for clicks and drags (tests replace it).
    var now: () -> Double = CACurrentMediaTime

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.glassTabSolidPill) private var solidPill
    @State private var livePress: GlassTabPress?
    @Environment(\.glassTabPreviewPress) private var previewPress
    /// The press drawn: the pointer's, or a render's stand-in.
    private var press: GlassTabPress? { livePress ?? previewPress }

    var body: some View {
        let frames = metrics.slotFrames(tabs)
        let width = frames.last?.maxX ?? 0
        let pill = metrics.pillFrame(tabs, progress: pager.progress)
        ZStack(alignment: .topLeading) {
            Capsule().fill(colors.track)
            glassPill(pill)
            labels(selected: false, frames: frames)
                .compositingGroup()
                .mask { PillCutout(pill: pill, inverted: true) }
            labels(selected: true, frames: frames)
                .compositingGroup()
                .mask { PillCutout(pill: pill, inverted: false) }
        }
        .frame(width: width, height: metrics.height)
        .contentShape(Capsule())
        .gesture(pressGesture(frames: frames))
        .focusable(interactions: .activate)  // keyboard focus only with keyboard navigation on, like the system control
        .onMoveCommand { direction in
            switch direction {
            case .left: pager.select(pager.selection - 1, at: now())
            case .right: pager.select(pager.selection + 1, at: now())
            default: break
            }
        }
        .onAppear { pager.reduceMotion = reduceMotion }
        .onChange(of: reduceMotion) { _, value in pager.reduceMotion = value }
        .accessibilityRepresentation {
            Picker(accessibilityLabel, selection: Binding(get: { pager.selection }, set: { pager.select($0, at: now()) })) {
                ForEach(tabs.indices, id: \.self) { i in Text(tabs[i].spokenLabel).tag(i) }
            }
            .pickerStyle(.segmented)
        }
    }

    @ViewBuilder private func glassPill(_ pill: CGRect) -> some View {
        let shape = Capsule()
        Group {
            if #available(macOS 26.0, *), !solidPill {
                // Not `.interactive()`: the pill never takes a hit (the bar's gesture does), so the press response is
                // ours, below.
                // The rim is a sibling below nothing interactive (the whole pill takes no hits), not an overlay.
                ZStack {
                    GlassEffectContainer {
                        shape.fill(.clear)
                            .frame(width: pill.width, height: pill.height)
                            .glassEffect(.regular.tint(colors.pill), in: .capsule)
                    }
                    shape.strokeBorder(colors.pillRim, lineWidth: 0.5).frame(width: pill.width, height: pill.height)
                }
            } else {
                ZStack {
                    shape.fill(colors.pill)
                        .shadow(color: .black.opacity(0.12), radius: 1, y: 0.5)
                    shape.strokeBorder(colors.pillRim, lineWidth: 0.5)
                }
                .frame(width: pill.width, height: pill.height)
            }
        }
        .scaleEffect(press?.onPill == true && !reduceMotion ? GlassTabPress.pillScale : 1)
        // Only the press animates here (the offset below follows the pager frame by frame, unanimated).
        .animation(GlassTabPress.pressSpring, value: press?.onPill == true)
        .offset(x: pill.minX, y: pill.minY)
        .allowsHitTesting(false)
    }

    private func labels(selected: Bool, frames: [CGRect]) -> some View {
        ZStack(alignment: .topLeading) {
            ForEach(tabs.indices, id: \.self) { i in
                label(tabs[i], selected: selected)
                    .frame(width: frames[i].width, height: metrics.height)
                    .opacity(press?.slot == i ? GlassTabPress.pressedLabelOpacity : 1)
                    .offset(x: frames[i].minX)
            }
        }
        .frame(width: frames.last?.maxX ?? 0, height: metrics.height, alignment: .topLeading)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private func label(_ tab: GlassTab, selected: Bool) -> some View {
        HStack(spacing: metrics.countGap) {
            Text(tab.title)
                .font(Font(metrics.titleFont))
                .foregroundStyle(selected ? colors.selectedText : colors.text)
                .lineLimit(1)
                .fixedSize()
            if tab.count > 0 {
                Text(String(tab.count))
                    .font(Font(metrics.countFont))
                    .foregroundStyle(colors.countText)
                    .frame(width: metrics.countWidth(tab.count), height: 16)
                    .background(Capsule().fill(colors.countFill))
            }
        }
    }

    /// Click: the slot under the pointer on release (`GlassTabPress`). Press on the pill and move: the pill follows the
    /// pointer, and the release settles it like a trackpad release (any tab may be the target).
    private func pressGesture(frames: [CGRect]) -> some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .local)
            .onChanged { value in
                if livePress == nil {
                    livePress = GlassTabPress.begin(at: value.startLocation, pager: pager, tabs: tabs, metrics: metrics, time: now())
                }
                livePress?.move(translation: value.translation.width, pager: pager, tabs: tabs, metrics: metrics, time: now())
            }
            .onEnded { value in
                livePress?.end(at: value.location, pager: pager, tabs: tabs, metrics: metrics, time: now())
                livePress = nil
            }
    }
}

/// What a pointer press on the bar does to the pager (drawing aside), so tests drive it without a window.
struct GlassTabPress: Equatable {
    static let dragThreshold: CGFloat = 3
    static let pillScale: CGFloat = 1.03
    static let pressedLabelOpacity: Double = 0.6
    /// The pager's spring, for the press swell and its release.
    static let pressSpring = Animation.spring(response: GlassTabSpring.standard.response, dampingFraction: 1.0)

    /// Pressed on the pill: the pill is held where it is drawn (a settle under it stops there) and moving drags it.
    var onPill: Bool
    /// The tab under a press that is not on the pill: its label dims until release; the click commits on release.
    var slot: Int?
    /// The held pill's centre, the origin of a drag.
    var pillCentre: CGFloat
    var dragging = false

    @MainActor static func begin(at location: CGPoint, pager: GlassTabPager, tabs: [GlassTab], metrics: GlassTabMetrics,
                                 time: Double) -> GlassTabPress {
        let pill = metrics.pillFrame(tabs, progress: pager.progress)
        let onPill = pill.insetBy(dx: -metrics.inset, dy: -metrics.inset).contains(location)
        // Hold the pill at its presented position now, so a drag that starts after a pause moves from where it is drawn.
        if onPill { pager.grab(at: time) }
        return GlassTabPress(onPill: onPill, slot: onPill ? nil : metrics.slot(at: location.x, tabs), pillCentre: pill.midX)
    }

    @MainActor mutating func move(translation: CGFloat, pager: GlassTabPager, tabs: [GlassTab], metrics: GlassTabMetrics,
                                  time: Double) {
        guard onPill, dragging || abs(translation) >= Self.dragThreshold else { return }
        dragging = true
        pager.drag(to: metrics.progress(forPillCentre: pillCentre + translation, tabs), at: time)
    }

    @MainActor func end(at location: CGPoint, pager: GlassTabPager, tabs: [GlassTab], metrics: GlassTabMetrics, time: Double) {
        if dragging { return pager.release(at: time, oneStep: false) }
        if let index = metrics.slot(at: location.x, tabs), location.y >= -8, location.y <= metrics.height + 8 {
            pager.select(index, at: time)
        } else if onPill {
            // A held pill let go away from the bar settles on the nearest tab.
            pager.release(at: time, oneStep: false)
        }
    }
}

extension EnvironmentValues {
    /// Draw the pill as a solid capsule instead of Liquid Glass. Offscreen renders only: `cacheDisplay` cannot capture
    /// glass, which the window server composites on screen.
    @Entry var glassTabSolidPill = false
    /// Renders only: draw the bar as if this press were held.
    @Entry var glassTabPreviewPress: GlassTabPress? = nil
}

/// The pill's shape as a mask: inside only, or everything but the inside.
private struct PillCutout: View {
    let pill: CGRect
    let inverted: Bool
    var body: some View {
        Canvas { context, size in
            var path = Path(roundedRect: pill, cornerRadius: pill.height / 2, style: .continuous)
            if inverted {
                var all = Path(CGRect(origin: .zero, size: size))
                all.addPath(path)
                path = all
                context.fill(path, with: .color(.black), style: FillStyle(eoFill: true))
            } else {
                context.fill(path, with: .color(.black))
            }
        }
    }
}

// MARK: - The clock

/// Drives the pager's spring from the display's own frames while a settle runs, and tells the pager whether frames are
/// being drawn (a visible window). Put it once in the window, behind the content.
struct GlassTabPagerClock: NSViewRepresentable {
    let pager: GlassTabPager

    func makeNSView(context: Context) -> ClockView {
        let view = ClockView()
        view.attach(pager)
        return view
    }

    func updateNSView(_ view: ClockView, context: Context) { view.attach(pager) }

    static func dismantleNSView(_ view: ClockView, coordinator: ()) { view.stop() }

    final class ClockView: NSView {
        private weak var pager: GlassTabPager?
        private var link: CADisplayLink?

        override func hitTest(_ point: NSPoint) -> NSView? { nil }

        func attach(_ pager: GlassTabPager) {
            guard self.pager !== pager else { return }
            self.pager = pager
            pager.isLive = { [weak self] in
                guard let window = self?.window else { return false }
                return window.isVisible && window.occlusionState.contains(.visible)
            }
            pager.onNeedsFrames = { [weak self] needs in self?.run(needs) }
            run(pager.needsFrames)
        }

        private var closeObserver: NSObjectProtocol?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            link?.invalidate()
            link = nil
            if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
            closeObserver = nil
            if let window {
                // A closed window draws no frames: a settle still running lands at once, so it never waits mid-way
                // (and nothing waiting for its rest, like a window height, waits with it).
                closeObserver = NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: window,
                                                                       queue: nil) { [weak self] _ in
                    MainActor.assumeIsolated {
                        guard let pager = self?.pager, pager.phase != .resting else { return }
                        pager.jump(to: pager.selection)
                    }
                }
            }
            run(pager?.needsFrames ?? false)
        }

        func stop() {
            link?.invalidate()
            link = nil
            if let closeObserver { NotificationCenter.default.removeObserver(closeObserver) }
            closeObserver = nil
        }

        private func run(_ needs: Bool) {
            if needs, link == nil, window != nil {
                let link = displayLink(target: self, selector: #selector(frame(_:)))
                link.add(to: .main, forMode: .common)
                self.link = link
            }
            link?.isPaused = !needs
        }

        @objc private func frame(_ link: CADisplayLink) {
            pager?.tick(at: link.targetTimestamp)
        }
    }
}
