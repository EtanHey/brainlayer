import AppKit
import SwiftUI

/// Tab swipe: a two-finger sideways swipe on the trackpad moves the pages AND the tab bar's glass pill with the
/// fingers, 1:1, and the release settles on a tab (the shared tab-swipe spec; `GlassTabPager` does the moving). This
/// file decides which scroll events are that swipe. The axis rules are VoiceBar's `NotchHistoryCategoryPager`:
///
/// - A gesture picks its axis once, from its first 6 pt: sideways only when it is 1.5 times more horizontal than
///   vertical. A vertical gesture stays the page's own scroll for its whole length and never moves a page.
/// - Once it is sideways, the whole travel so far is handed over at once (`grab`), then every step as it comes (`move`),
///   so the pages never lag the fingers by the slop.
/// - Its end (or cancel) is the release; the pager picks the tab from position plus projected velocity, one tab at most.
struct NDPagePager: Equatable {
    enum Axis: Equatable { case undecided, horizontal, vertical }

    static let slop: CGFloat = 6
    static let horizontalBias: CGFloat = 1.5

    private(set) var axis: Axis = .undecided
    private var travel: CGSize = .zero

    var isSwiping: Bool { axis == .horizontal }

    mutating func begin() {
        axis = .undecided
        travel = .zero
    }

    /// One more step of the gesture: the sideways distance to move the pages by now, or nil while it is not a page
    /// swipe. On the step that decides the axis, that is the whole sideways travel so far.
    mutating func move(dx: CGFloat, dy: CGFloat) -> CGFloat? {
        switch axis {
        case .vertical: return nil
        case .horizontal: return dx
        case .undecided:
            travel.width += dx
            travel.height += dy
            guard max(abs(travel.width), abs(travel.height)) >= Self.slop else { return nil }
            axis = abs(travel.width) > abs(travel.height) * Self.horizontalBias ? .horizontal : .vertical
            return axis == .horizontal ? travel.width : nil
        }
    }

    /// The gesture ended: whether a page swipe was in progress (and so must be released).
    mutating func end() -> Bool {
        defer { begin() }
        return axis == .horizontal
    }
}

/// Which scroll events belong to a page swipe, from their phases (VoiceBar's `NotchHistorySwipeMonitor.route`): until
/// the axis is known, and for every vertical gesture, the event goes on to the page untouched; once it is a sideways
/// swipe the page never sees it, nor the momentum after it. A plain mouse wheel has no phases and is left alone.
///
/// Precedence: a gesture that begins over something that scrolls sideways itself (an `NSScrollView` wider than its
/// view) belongs to that view for its whole length. No page has one today; the rule keeps a future one working.
struct NDPageSwipe: Equatable {
    struct Event {
        var phase: NSEvent.Phase = []
        var momentumPhase: NSEvent.Phase = []
        var dx: CGFloat = 0
        var dy: CGFloat = 0
        /// The gesture began over the window's pages.
        var inside = true
        /// The gesture began over a view that scrolls sideways itself.
        var overSidewaysScroller = false
    }

    /// What the pages do with an event: take hold and move by the travel so far, move by a step, or let go.
    enum Drive: Equatable {
        case grab(CGFloat)
        case move(CGFloat)
        case release
    }

    struct Decision: Equatable {
        /// Keep the event from the views under it.
        var swallow = false
        var drive: Drive?
    }

    private(set) var pager = NDPagePager()
    private var tracking = false
    private var sideways = false
    private var swallowsMomentum = false

    mutating func route(_ event: Event) -> Decision {
        if !event.momentumPhase.isEmpty {
            let swallow = swallowsMomentum
            if event.momentumPhase.contains(.ended) || event.momentumPhase.contains(.cancelled) { swallowsMomentum = false }
            return Decision(swallow: swallow)
        }
        let phase = event.phase
        if phase.contains(.began) {
            swallowsMomentum = false
            sideways = false
            tracking = event.inside && !event.overSidewaysScroller
            pager.begin()
        }
        guard tracking else { return Decision() }
        if phase.contains(.ended) || phase.contains(.cancelled) {
            tracking = false
            swallowsMomentum = sideways
            // VoiceBar settles a cancelled gesture as an ended one. The page's scroll view sees its gesture end.
            return Decision(swallow: false, drive: pager.end() ? .release : nil)
        }
        guard phase.contains(.began) || phase.contains(.changed) else { return Decision() }
        let wasSideways = sideways
        guard let dx = pager.move(dx: event.dx, dy: event.dy) else { return Decision() }
        sideways = true
        return Decision(swallow: true, drive: wasSideways ? .move(dx) : .grab(dx))
    }

    /// Whether a page swipe is held now (a window closing mid-swipe lets it go).
    var isHolding: Bool { tracking && sideways }

    /// Whether `view` sits inside a scroll view that scrolls sideways (its document is wider than its view, or it shows
    /// a horizontal scroller).
    @MainActor static func scrollsSideways(_ view: NSView?) -> Bool {
        var current = view
        while let v = current {
            if let scroll = v as? NSScrollView {
                if scroll.hasHorizontalScroller { return true }
                if let document = scroll.documentView, document.frame.width > scroll.contentView.bounds.width + 0.5 { return true }
            }
            current = v.superview
        }
        return false
    }
}

extension NDStore {
    /// A page swipe's step, in points of finger travel across `width` (the window): fingers moving left (negative dx)
    /// bring the next page in, exactly as far as they moved.
    func drivePages(_ drive: NDPageSwipe.Drive, at time: Double, width: CGFloat) {
        let width = Double(max(width, 1))
        switch drive {
        case .grab(let dx):
            tabs.grab(at: time)
            tabs.drag(by: -Double(dx) / width, at: time)
        case .move(let dx):
            tabs.drag(by: -Double(dx) / width, at: time)
        case .release:
            tabs.release(at: time)
        }
    }
}


/// Where the swipe monitor registers. The app's own local event monitors in the window; tests count registrations.
@MainActor protocol NDScrollEventMonitoring: AnyObject {
    /// `swallows` answers whether the event is kept from the views under it.
    func add(_ swallows: @escaping @MainActor (NSEvent) -> Bool) -> Any?
    func remove(_ token: Any)
}

@MainActor final class NDLocalScrollMonitors: NDScrollEventMonitoring {
    static let shared = NDLocalScrollMonitors()
    func add(_ swallows: @escaping @MainActor (NSEvent) -> Bool) -> Any? {
        NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            MainActor.assumeIsolated { swallows(event) } ? nil : event
        }
    }
    func remove(_ token: Any) { NSEvent.removeMonitor(token) }
}

/// Feeds the window's two-finger scrolls to `NDPageSwipe`. The page's vertical `ScrollView` would take these events
/// whole, so a local monitor looks at each one first.
///
/// Lifetime (review r1 of #1870): the dashboard window is retained and reused (`isReleasedWhenClosed = false`), so its
/// content never leaves it on close. The monitor therefore follows the window itself, not the view's attachment: it
/// registers while the window is visible (on attach, `didBecomeKey`, `didBecomeMain`, `didChangeOcclusionState`), and
/// `willClose` removes it and lets go of a swipe in progress (the pages settle on a tab). Registration is idempotent; detaching or dismantling the
/// view removes it too. Offscreen renders and tests that host the root without a window register nothing.
///
/// Deltas: the gesture phases decide, not `hasPreciseScrollingDeltas`. A trackpad (and a Magic Mouse) reports phases
/// with precise point deltas; a notched mouse wheel reports neither, and the router ignores phase-less events, so a
/// line-based delta never reaches the pager.
struct NDPageSwipeMonitor: NSViewRepresentable {
    /// A swipe's step, its event time (seconds, the `CACurrentMediaTime` clock) and the width the travel is a share of.
    let onDrive: @MainActor (NDPageSwipe.Drive, Double, CGFloat) -> Void

    func makeNSView(context: Context) -> Probe { Probe() }
    func updateNSView(_ view: Probe, context: Context) { view.onDrive = onDrive }
    static func dismantleNSView(_ view: Probe, coordinator: ()) { view.detach() }

    final class Probe: NSView {
        var onDrive: (@MainActor (NDPageSwipe.Drive, Double, CGFloat) -> Void)?
        /// Tests inject a counting registry.
        var monitors: any NDScrollEventMonitoring = NDLocalScrollMonitors.shared
        /// Tests use a private center: AppKit's own observers of these window notifications expect a real window.
        var notifications = NotificationCenter.default
        private var token: Any?
        private(set) var swipe = NDPageSwipe()
        private weak var observed: AnyObject?
        private var isWindowVisible: () -> Bool = { false }
        private var hitView: (NSPoint) -> NSView? = { _ in nil }
        private var observers: [NSObjectProtocol] = []

        /// Whether the local monitor is registered now.
        var isListening: Bool { token != nil }

        override func hitTest(_ point: NSPoint) -> NSView? { nil }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard let window else { return detach() }
            attach(to: window, isVisible: { [weak window] in window?.isVisible ?? false }, hitView: { [weak window] location in
                // `hitTest` takes a point in the content view's superview's coordinates.
                window?.contentView.flatMap { $0.hitTest($0.superview?.convert(location, from: nil) ?? location) }
            })
        }

        /// Follow `target`'s (the window's) open and close. Tests pass a stand-in object and post its notifications.
        func attach(to target: AnyObject, isVisible: @escaping () -> Bool, hitView: @escaping (NSPoint) -> NSView? = { _ in nil }) {
            if observed === target { return listenIfVisible() }
            detach()
            observed = target
            isWindowVisible = isVisible
            self.hitView = hitView
            let center = notifications
            observers.append(center.addObserver(forName: NSWindow.willCloseNotification, object: target, queue: nil) { [weak self] _ in
                MainActor.assumeIsolated { self?.stopListening() }
            })
            for name in [NSWindow.didBecomeKeyNotification, NSWindow.didBecomeMainNotification,
                         NSWindow.didChangeOcclusionStateNotification] {
                observers.append(center.addObserver(forName: name, object: target, queue: nil) { [weak self] _ in
                    MainActor.assumeIsolated { self?.listenIfVisible() }
                })
            }
            listenIfVisible()
        }

        func detach() {
            for observer in observers { notifications.removeObserver(observer) }
            observers = []
            stopListening()
            observed = nil
            isWindowVisible = { false }
            hitView = { _ in nil }
        }

        private func listenIfVisible() {
            guard token == nil, observed != nil, isWindowVisible() else { return }
            swipe = NDPageSwipe()
            token = monitors.add { [weak self] event in
                self?.route(event, in: event.window, at: event.locationInWindow) ?? false
            }
        }

        private func stopListening() {
            if let token { monitors.remove(token) }
            token = nil
            // A swipe held when the window closes is let go, so the pages settle on a tab instead of staying mid-way.
            if swipe.isHolding { onDrive?(.release, CACurrentMediaTime(), bounds.width) }
            swipe = NDPageSwipe()
        }

        /// Whether the event belongs to a page swipe and must not reach the page. `eventWindow` and `location` are the
        /// event's own (a test's synthetic scroll event carries neither).
        func route(_ event: NSEvent, in eventWindow: AnyObject?, at location: NSPoint) -> Bool {
            guard let observed, eventWindow === observed, event.type == .scrollWheel else { return false }
            var input = NDPageSwipe.Event(phase: event.phase, momentumPhase: event.momentumPhase,
                                          dx: event.scrollingDeltaX, dy: event.scrollingDeltaY)
            if event.phase.contains(.began) {
                input.inside = bounds.contains(convert(location, from: nil))
                input.overSidewaysScroller = NDPageSwipe.scrollsSideways(hitView(location))
            }
            let decision = swipe.route(input)
            // A synthesized event can carry no time; the clock's own is the same timebase.
            if let drive = decision.drive { onDrive?(drive, event.timestamp > 0 ? event.timestamp : CACurrentMediaTime(), bounds.width) }
            return decision.swallow
        }
    }
}
