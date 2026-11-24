import AppKit
import SwiftUI

/// Tab swipe (phase 14): a two-finger sideways swipe on the trackpad moves the window to the neighbouring page, the
/// way VoiceBar's History moves between Dictations, Ask and Speech. The gesture rules are VoiceBar's
/// `NotchHistoryCategoryPager`, unchanged:
///
/// - A gesture picks its axis once, from its first 6 pt: sideways only when it is 1.5 times more horizontal than
///   vertical. A vertical gesture stays the page's own scroll for its whole length and never changes page.
/// - It settles on release: past 0.22 of VoiceBar's 400 pt History list (88 pt, so the same finger travel works in
///   both apps), or a flick (a last step of 14 pt or more in the same direction), moves one page; anything shorter
///   moves none. However long the gesture, it moves at most one page.
/// - The ends stop: a swipe past Jobs stays on Jobs, a swipe before Overview stays on Overview (VoiceBar does not wrap).
///
/// The page then changes through `store.page`, exactly as a tab click does: the page swaps at once and only the
/// window's height moves (R1.10.1), so the swipe has no slide of its own and Reduce Motion keeps working as before.
struct NDPagePager: Equatable {
    enum Axis: Equatable { case undecided, horizontal, vertical }

    static let slop: CGFloat = 6
    static let horizontalBias: CGFloat = 1.5
    static let commitFraction: CGFloat = 0.22
    static let flickDelta: CGFloat = 14
    /// VoiceBar's History list width, the length `commitFraction` is a share of.
    static let travelWidth: CGFloat = 400

    private(set) var axis: Axis = .undecided
    private var travel: CGSize = .zero
    private var lastDeltaX: CGFloat = 0

    var isSwiping: Bool { axis == .horizontal }

    mutating func begin() {
        axis = .undecided
        travel = .zero
        lastDeltaX = 0
    }

    /// One more step of the gesture. Returns whether it is a page swipe (the caller then keeps the event from the page's
    /// vertical scroll).
    @discardableResult
    mutating func move(dx: CGFloat, dy: CGFloat) -> Bool {
        guard axis != .vertical else { return false }
        travel.width += dx
        travel.height += dy
        lastDeltaX = dx
        if axis == .undecided, max(abs(travel.width), abs(travel.height)) >= Self.slop {
            axis = abs(travel.width) > abs(travel.height) * Self.horizontalBias ? .horizontal : .vertical
        }
        return axis == .horizontal
    }

    /// The gesture ended: +1 for the next page (fingers moved left), -1 for the previous one, 0 for none.
    mutating func end() -> Int {
        defer { begin() }
        guard axis == .horizontal else { return 0 }
        let x = travel.width
        let flicked = abs(lastDeltaX) >= Self.flickDelta && (lastDeltaX > 0) == (x > 0)
        guard abs(x) > Self.travelWidth * Self.commitFraction || flicked else { return 0 }
        return x < 0 ? 1 : -1
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

    struct Decision: Equatable {
        /// Keep the event from the views under it.
        var swallow = false
        /// The page step to take now (only on the gesture's end).
        var step = 0
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
            // VoiceBar settles a cancelled gesture as an ended one.
            return Decision(swallow: false, step: pager.end()) // the page's scroll view sees its gesture end
        }
        guard phase.contains(.began) || phase.contains(.changed) else { return Decision() }
        sideways = pager.move(dx: event.dx, dy: event.dy)
        return Decision(swallow: sideways)
    }

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

extension NDPage {
    /// The page `step` pages along, stopping at the ends.
    func neighbour(_ step: Int) -> NDPage {
        let pages = Self.allCases
        let index = pages.firstIndex(of: self) ?? 0
        return pages[min(pages.count - 1, max(0, index + step))]
    }
}

extension NDStore {
    /// A finished page swipe: the same change a tab click makes.
    func swipePage(_ step: Int) {
        guard step != 0 else { return }
        page = page.neighbour(step)
    }
}

/// Feeds the window's two-finger scrolls to `NDPageSwipe`. The page's vertical `ScrollView` would take these events
/// whole, so a local monitor looks at each one first. It only listens while its view is in a window, so offscreen renders
/// and tests that host the root without a window install nothing.
struct NDPageSwipeMonitor: NSViewRepresentable {
    let onStep: @MainActor (Int) -> Void

    func makeNSView(context: Context) -> Probe { Probe() }
    func updateNSView(_ view: Probe, context: Context) { view.onStep = onStep }
    static func dismantleNSView(_ view: Probe, coordinator: ()) { view.removeMonitor() }

    final class Probe: NSView {
        var onStep: (@MainActor (Int) -> Void)?
        private var monitor: Any?
        private var swipe = NDPageSwipe()

        override func hitTest(_ point: NSPoint) -> NSView? { nil }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            removeMonitor()
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { [weak self] event in
                guard let self else { return event }
                return MainActor.assumeIsolated { self.handle(event) } ? nil : event
            }
        }

        func removeMonitor() {
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
            swipe = NDPageSwipe()
        }

        /// Whether the event belongs to a page swipe and must not reach the page.
        func handle(_ event: NSEvent) -> Bool {
            guard let window, event.window === window else { return false }
            var input = NDPageSwipe.Event(phase: event.phase, momentumPhase: event.momentumPhase,
                                          dx: event.scrollingDeltaX, dy: event.scrollingDeltaY)
            if event.phase.contains(.began) {
                let location = event.locationInWindow
                input.inside = bounds.contains(convert(location, from: nil))
                // `hitTest` takes a point in the content view's superview's coordinates.
                let hit = window.contentView.flatMap { $0.hitTest($0.superview?.convert(location, from: nil) ?? location) }
                input.overSidewaysScroller = NDPageSwipe.scrollsSideways(hit)
            }
            let decision = swipe.route(input)
            if decision.step != 0 { onStep?(decision.step) }
            return decision.swallow
        }
    }
}
