import AppKit
import SwiftUI

// MARK: - The chart

/// The pipeline as a loop (R1.3): the agents' group, six stations and the Memory hub, every fault in one language, each
/// fault's action at rest beside its station, and a popover per station on hover or click.
struct NDPipelineChart: View {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    let state: NDPipelineState
    /// The memory count in the hub ("297,412").
    let memories: String?
    /// Each station's red or amber item, for its action and its popover.
    var items: [NDStation: NDR1.Item] = [:]
    /// Google Drive's reconnect has started: its button gives way to the waiting line.
    var waiting = false
    var perform: (BrainBarMenuPresentation.ActionKind) -> Void = { _ in }
    var interactive = true
    var popover: (NDStation) -> AnyView? = { _ in nil }
    /// Render harness only: one station drawn hovered, its popover drawn where AppKit would open it.
    var previewHover: NDStation? = nil
    @State private var hovered: NDStation?
    @StateObject private var panels = NDPanelPresenter()

    /// Tab order (change 16): each station, then its action, then the next station.
    static let order: [NDStation] = [.agents, .read, .queue, .memory, .jobs, .search, .mac, .drive]

    struct Control: Equatable, Identifiable {
        var station: NDStation
        var action: BrainBarMenuPresentation.Action?
        var id: String { action == nil ? station.rawValue : station.rawValue + ".action" }
    }

    /// The controls in the key loop, in order: the stations only. R1.5 (L40): no action is on the chart at rest; each
    /// fault's action is in its station's panel (Return or Space opens it with focus on the action) and in the dropdown.
    static func controls(_ items: [NDStation: NDR1.Item]) -> [Control] {
        order.map { Control(station: $0, action: nil) }
    }

    struct LineHover {
        var title: String
        var line: String
        var rects: [CGRect]
        var edge: Edge
    }

    /// The rail, the return line and the Kept-safe frame, each with its title and line (L38).
    static let lineHovers: [LineHover] = [
        LineHover(title: Say.railTitle, line: Say.railLine, rects: NDPipelineLayout.hoverStrips(.rail), edge: .top),
        LineHover(title: Say.answersTitle, line: Say.answersLine, rects: NDPipelineLayout.hoverStrips(.answers), edge: .bottom),
        LineHover(title: Say.keptSafe, line: Say.keptSafeLine, rects: [NDPipelineLayout.keptSafe], edge: .top),
    ]

    /// R1.5: the Kept-safe frame no longer grows (no actions on the chart).
    var keptGrown: Bool { false }

    static func actionStations(_ items: [NDStation: NDR1.Item]) -> Set<NDStation> {
        Set(items.filter { $0.value.action != nil }.keys)
    }

    /// L32: inside the window an action has one short name, at the station and in its popover.
    static func shortLabel(_ action: BrainBarMenuPresentation.Action) -> String {
        switch action.kind {
        case .reconnectDrive, .connectDrive: Say.reconnectShort
        case .copyReport: Say.copyReportShort
        default: action.label
        }
    }

    static func actionWidth(_ action: BrainBarMenuPresentation.Action) -> CGFloat {
        let w = NSAttributedString(string: shortLabel(action), attributes: [.font: NSFont.systemFont(ofSize: 13, weight: .medium)]).size().width
        return ceil(w) + 20
    }

    var body: some View {
        let shownHover = previewHover ?? hovered
        ZStack(alignment: .topLeading) {
            Canvas { ctx, _ in Self.draw(ctx, state: state, memories: memories, hovered: shownHover, keptGrown: keptGrown) }
                .frame(width: NDPipelineLayout.drawWidth, height: NDPipelineLayout.size.height)
                .allowsHitTesting(false)
            // R1.4 changes 5, 8 and 9: the words that left the chart, on the things they name.
            ForEach(Self.lineHovers, id: \.title) { hover in
                ForEach(Array(hover.rects.enumerated()), id: \.offset) { _, rect in
                    NDHoverRegion(title: hover.title, line: hover.line, interactive: interactive, edge: hover.edge)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                }
            }
            ForEach(Self.controls(items)) { control in station(control.station) }
            // R1.4 change 2: each provider icon's hover and spoken name is its name.
            ForEach(NDPipelineLayout.agentRows, id: \.0.rawValue) { family, _ in
                let c = NDPipelineLayout.iconCentre(family)
                Color.clear.frame(width: 22, height: 22).contentShape(Rectangle())
                    .help(NDProvider.name(family))
                    .accessibilityElement()
                    .accessibilityLabel(NDProvider.name(family))
                    .position(c)
            }
            if let previewHover { preview(previewHover) }
        }
        .frame(width: NDPipelineLayout.size.width, height: NDPipelineLayout.size.height, alignment: .topLeading)
        // The panels are placed from this view's own coordinates: the same space the stations are drawn in.
        .background(alignment: .topLeading) {
            NDChartAnchor(presenter: panels).frame(width: NDPipelineLayout.size.width, height: NDPipelineLayout.size.height)
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Say.pipeline)
    }

    /// The station's panel at its content's measured height.
    func panel(_ s: NDStation, content: AnyView) -> NDPipelineLayout.Panel {
        let actions: Set<NDStation> = []  // R1.5: no action sits on the chart, so none for a panel to clear
        let widths: [NDStation: CGFloat] = [:]
        // The width first (a side panel narrows to stay inside the window), then the height the content needs at it.
        let width = NDPipelineLayout.panel(s, state: state, actions: actions, actionWidths: widths, height: 100).body.width
        let height = NDStationPopover.measure(content, width: width).height
        return NDPipelineLayout.panel(s, state: state, actions: actions, actionWidths: widths, height: height)
    }

    @ViewBuilder private func station(_ s: NDStation) -> some View {
        let shape = NDPipelineLayout.hitShape(s, state)
        let frame = shape.label.map { shape.circle.union($0) } ?? shape.circle
        NDStationButton(station: s, interactive: interactive, hovered: $hovered, presenter: panels,
                        shape: NDStationHitShape(circle: shape.circle.offsetBy(dx: -frame.minX, dy: -frame.minY),
                                                 label: shape.label?.offsetBy(dx: -frame.minX, dy: -frame.minY)),
                        content: { popover(s) }, layout: { content in panel(s, content: content) })
            .frame(width: frame.width, height: frame.height)
            .position(x: frame.midX, y: frame.midY)
    }

    /// The render harness's hover: the same panel chrome the window shows, at the frame the layout gives it.
    @ViewBuilder private func preview(_ s: NDStation) -> some View {
        if let content = popover(s) {
            let layout = panel(s, content: content)
            NDPanelChrome(layout: layout, content: content)
                .frame(width: layout.outer.width, height: layout.outer.height)
                .offset(x: layout.outer.minX, y: layout.outer.minY)
        }
    }

    // MARK: drawing

    static func stroke(for pipe: NDPipe, in state: NDPipelineState) -> NDPipeStroke {
        state.look(pipe) == .dashed ? .unconfirmed : .flowing
    }

    static func text(_ s: String, size: CGFloat, color: Color, weight: Font.Weight = .regular, mono: Bool = false) -> Text {
        let font = mono ? Font.system(size: size, weight: weight).monospacedDigit() : Font.system(size: size, weight: weight)
        return Text(s).font(font).foregroundColor(color)
    }

    enum Anchor { case start, middle, end }

    /// Text placed by its baseline, as the comp's SVG places it.
    static func put(_ ctx: GraphicsContext, _ t: Text, x: CGFloat, baseline: CGFloat, size: CGFloat, anchor: Anchor) {
        let y = baseline - size * 0.36
        let unit: UnitPoint = switch anchor {
        case .start: .leading
        case .middle: .center
        case .end: .trailing
        }
        ctx.draw(t, at: CGPoint(x: x, y: y), anchor: unit)
    }

    static func symbol(_ ctx: GraphicsContext, _ name: String, at p: CGPoint, size: CGFloat, color: Color) {
        var resolved = ctx.resolve(Image(systemName: name))
        resolved.shading = .color(color)
        ctx.draw(resolved, in: CGRect(x: p.x - size / 2, y: p.y - size / 2, width: size, height: size))
    }

    /// A 5 pt open chevron, its tip 1.5 pt before the line's end, along the last segment.
    static func arrowhead(_ ctx: GraphicsContext, end: CGPoint, before: CGPoint, shading: GraphicsContext.Shading) {
        let l = NDPolyline.seg(before, end)
        guard l > 0 else { return }
        let dx = (end.x - before.x) / l, dy = (end.y - before.y) / l
        let tip = CGPoint(x: end.x - dx * 1.5, y: end.y - dy * 1.5)
        let back = CGPoint(x: tip.x - dx * 5, y: tip.y - dy * 5)
        var head = Path()
        head.move(to: CGPoint(x: back.x - dy * 4, y: back.y + dx * 4))
        head.addLine(to: tip)
        head.addLine(to: CGPoint(x: back.x + dy * 4, y: back.y - dx * 4))
        ctx.stroke(head, with: shading, style: StrokeStyle(lineWidth: 1.6, lineCap: .round, lineJoin: .round))
    }

    /// Words set in a line on a plane mask.
    static func inLine(_ ctx: GraphicsContext, _ words: String, at p: CGPoint, color: Color) {
        let width = ceil(NSAttributedString(string: words, attributes: [.font: NSFont.systemFont(ofSize: 11)]).size().width) + 12
        ctx.fill(Path(CGRect(x: p.x - width / 2, y: p.y - 8, width: width, height: 16)), with: .color(ND.card))
        ctx.draw(text(words, size: 11, color: color), at: p, anchor: .center)
    }

    /// A caption set in a frame's top border, on a mask.
    static func caption(_ ctx: GraphicsContext, _ words: String, frame: CGRect, inset: CGFloat) {
        let w = ceil(NSAttributedString(string: words, attributes: [.font: NSFont.systemFont(ofSize: 11, weight: .medium)]).size().width)
        ctx.fill(Path(CGRect(x: frame.minX + inset - 6, y: frame.minY - 8, width: w + 12, height: 16)), with: .color(ND.card))
        put(ctx, text(words, size: 11, color: ND.t3, weight: .medium), x: frame.minX + inset, baseline: frame.minY + 4, size: 11, anchor: .start)
    }

    static func draw(_ ctx: GraphicsContext, state: NDPipelineState, memories: String?, hovered: NDStation? = nil, keptGrown: Bool = false) {
        drawPipes(ctx, state: state, keptGrown: keptGrown)
        drawAgents(ctx, state: state, hovered: hovered)
        drawHub(ctx, state: state, memories: memories, hovered: hovered)
        drawStations(ctx, state: state, hovered: hovered)
        if state.allUnconfirmed {
            let words = state.lastKnownAt.map(Say.valuesLastKnownAt) ?? Say.everyPipeNotConfirmed
            put(ctx, text(words, size: 11, color: ND.t2, weight: .medium), x: NDPipelineLayout.size.width, baseline: 10, size: 11, anchor: .end)
        }
    }

    /// R1.4 changes 16 and 17: every pipe whole, from line end to line end; the pipe after a faulty stage at 28 %; no
    /// stub, no gap, no switch; the arrow tip on the line's end point.
    static func drawPipes(_ ctx: GraphicsContext, state: NDPipelineState, keptGrown: Bool = false) {
        typealias L = NDPipelineLayout
        let dash: [CGFloat] = state.allUnconfirmed ? [2, 2] : []
        ctx.stroke(Path(roundedRect: L.keptFrame(grown: keptGrown), cornerRadius: 12, style: .continuous), with: .color(ND.hair2),
                   style: StrokeStyle(lineWidth: 1, dash: dash))
        for c in L.connections {
            let s: NDPipeStroke = switch state.look(c.pipe) {
            case .dashed: .unconfirmed
            case .faint: .downstream
            case .flowing: .flowing
            }
            ctx.stroke(c.line.path(radius: c.corner), with: s.shading, style: s.style)
            let pts = c.line.points
            if pts.count > 1 {
                arrowhead(ctx, end: pts[pts.count - 1], before: pts[pts.count - 2], shading: s.shading)
                if c.bothWays { arrowhead(ctx, end: pts[0], before: pts[1], shading: s.shading) }
            }
        }
    }

    /// The agents (R1.4 changes 1 to 3): four provider icons in one column, in the label colour; a provider a check names
    /// takes the fault treatment at icon scale (red icon, 1.5 pt ring at r 12, the badge on its top-right).
    static func drawAgents(_ ctx: GraphicsContext, state: NDPipelineState, hovered: NDStation?) {
        typealias L = NDPipelineLayout
        let unconf = state.allUnconfirmed
        let group = Path(roundedRect: L.agentsFrame, cornerRadius: 12, style: .continuous)
        ctx.fill(group, with: .color(ND.card))
        ctx.stroke(group, with: .color(ND.hair2), style: StrokeStyle(lineWidth: 1, dash: unconf ? [2, 2] : []))
        if hovered == .agents {
            ctx.stroke(Path(roundedRect: L.agentsFrame.insetBy(dx: -3.5, dy: -3.5), cornerRadius: 15, style: .continuous),
                       with: .color(ND.t2), lineWidth: 1)
        }
        for (family, _) in L.agentRows {
            let c = L.iconCentre(family)
            let sev = unconf ? nil : state.agentFaults[family]
            let color: Color = sev == .red ? ND.red : sev == .orange ? ND.orange : ND.t2
            let icon = NDProvider.icon(family)
            let rect = CGRect(x: c.x - L.iconSize / 2, y: c.y - L.iconSize / 2, width: L.iconSize, height: L.iconSize)
            ctx.fill(NDSVGPath.path(icon.path, viewBox: icon.viewBox, in: rect), with: .color(color), style: FillStyle(eoFill: icon.evenOdd))
            if let sev {
                let ring = sev == .red ? ND.red : ND.orange
                ctx.stroke(Path(ellipseIn: CGRect(x: c.x - 12, y: c.y - 12, width: 24, height: 24)), with: .color(ring), lineWidth: 1.5)
                let b = CGPoint(x: c.x + 8.5, y: c.y - 8.5)
                ctx.fill(Path(ellipseIn: CGRect(x: b.x - 7.2, y: b.y - 7.2, width: 14.4, height: 14.4)), with: .color(ND.card))
                symbol(ctx, sev == .red ? "exclamationmark.circle.fill" : "exclamationmark.triangle.fill", at: b, size: 12, color: ring)
            }
        }
    }

    /// Memory: an outline with its count.
    static func drawHub(_ ctx: GraphicsContext, state: NDPipelineState, memories: String?, hovered: NDStation?) {
        let hub = NDPipelineLayout.hub
        let path = Path(roundedRect: hub, cornerRadius: 12, style: .continuous)
        ctx.fill(path, with: .color(ND.card))
        ctx.stroke(path, with: .color(ND.hair2), style: StrokeStyle(lineWidth: 1, dash: state.allUnconfirmed ? [2, 2] : []))
        if hovered == .memory {
            ctx.stroke(Path(roundedRect: hub.insetBy(dx: -3.5, dy: -3.5), cornerRadius: 15, style: .continuous), with: .color(ND.t2), lineWidth: 1)
        }
        symbol(ctx, "brain", at: CGPoint(x: hub.midX, y: hub.minY + 27), size: 18, color: ND.t2)
        put(ctx, text(Say.stageMemory, size: 13, color: ND.t1, weight: .semibold), x: hub.midX, baseline: hub.minY + 57, size: 13, anchor: .middle)
        // R2A item 8: the count carries its unit ("297,412 memories"), on two lines when one would crowd the box.
        let count = memories ?? NDModel.Number.unknown.value
        let words = Say.memoriesCount(count)
        if NDPipelineLayout.textWidth(words, size: 11) <= hub.width - 16 {
            put(ctx, text(words, size: 11, color: ND.t2, mono: true), x: hub.midX, baseline: hub.minY + 72, size: 11, anchor: .middle)
        } else {
            put(ctx, text(count, size: 11, color: ND.t2, mono: true), x: hub.midX, baseline: hub.minY + 72, size: 11, anchor: .middle)
            put(ctx, text(Say.memoriesWord, size: 11, color: ND.t2), x: hub.midX, baseline: hub.minY + 86, size: 11, anchor: .middle)
        }
    }

    /// R1.4 change 18 (b): a Mac with an arrow down into its screen, drawn on a 16 pt grid at 14 pt, as comp 41 draws it.
    static func macWithArrow(_ ctx: GraphicsContext, at p: CGPoint, color: Color) {
        let k: CGFloat = 14 / 16
        func pt(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: p.x + (x - 8) * k, y: p.y + (y - 8) * k) }
        let screen = Path(roundedRect: CGRect(origin: pt(2.6, 2.6), size: CGSize(width: 10.8 * k, height: 7.8 * k)), cornerRadius: 1.3 * k)
        var base = Path()
        base.move(to: pt(1, 12.9)); base.addLine(to: pt(15, 12.9))
        var arrow = Path()
        arrow.move(to: pt(8, 4.4)); arrow.addLine(to: pt(8, 8.2))
        arrow.move(to: pt(6.3, 6.6)); arrow.addLine(to: pt(8, 8.3)); arrow.addLine(to: pt(9.7, 6.6))
        let style = StrokeStyle(lineWidth: 1.3, lineCap: .round, lineJoin: .round)
        ctx.stroke(screen, with: .color(color), style: style)
        ctx.stroke(base, with: .color(color), style: style)
        ctx.stroke(arrow, with: .color(color), style: StrokeStyle(lineWidth: 1.15, lineCap: .round, lineJoin: .round))
    }

    /// The badge's centre: on the ring at the station's top-right, at 45° (R1.4 change 16).
    static func badgeCentre(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x + 11, y: p.y - 11) }

    /// What a station draws for its treatment (one predicate; the chart, the rows and the icon all follow it).
    struct Marks: Equatable {
        var ring: Color?
        var ringDashed = false
        var badge: String?
        var nameRed = false
        var statusTone: NDModel.Tone = .fine
    }

    static func marks(_ t: NDPipelineState.Treatment) -> Marks {
        switch t {
        case .fault: Marks(ring: ND.red, badge: "exclamationmark.circle.fill", nameRed: true, statusTone: .fault)
        case .notice: Marks(ring: ND.orange, badge: "exclamationmark.triangle.fill", statusTone: .warn)
        case .notConfirmed: Marks(ring: ND.t3, ringDashed: true)
        case .calm: Marks()
        }
    }

    static func drawStations(_ ctx: GraphicsContext, state: NDPipelineState, hovered: NDStation?) {
        typealias L = NDPipelineLayout
        for station in NDStation.allCases where L.stations[station] != nil {
            let p = L.point(station)
            let t = state.treatment(station)
            let m = marks(t)
            if let ring = m.ring {
                let r = NDPipelineLayout.ringRadius
                let rect = CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)
                ctx.stroke(Path(ellipseIn: rect), with: .color(ring), style: StrokeStyle(lineWidth: m.ringDashed ? 1 : 1.5, dash: m.ringDashed ? [2, 2] : []))
            }
            if hovered == station {
                let rr: CGFloat = t == .fault || t == .notice ? 18.5 : 15.5
                ctx.stroke(Path(ellipseIn: CGRect(x: p.x - rr, y: p.y - rr, width: 2 * rr, height: 2 * rr)), with: .color(ND.t2), lineWidth: 1)
            }
            let node = Path(ellipseIn: CGRect(x: p.x - 13, y: p.y - 13, width: 26, height: 26))
            ctx.fill(node, with: .color(hovered == station ? ND.raised : ND.node))
            ctx.stroke(node, with: .color(ND.hair2), lineWidth: 1)
            if station == .mac {
                macWithArrow(ctx, at: p, color: m.nameRed ? ND.red : ND.t2)
            } else {
                symbol(ctx, L.symbols[station] ?? "circle", at: p, size: 14, color: m.nameRed ? ND.red : ND.t2)
            }
            if let badge = m.badge {
                let b = badgeCentre(p)
                ctx.fill(Path(ellipseIn: CGRect(x: b.x - 7.6, y: b.y - 7.6, width: 15.2, height: 15.2)), with: .color(ND.card))
                symbol(ctx, badge, at: b, size: 12, color: m.ring ?? ND.red)
            }
            put(ctx, text(L.name(station), size: 13, color: m.nameRed ? ND.redText : ND.t1, weight: m.nameRed ? .semibold : .regular),
                x: p.x, baseline: p.y + 29, size: 13, anchor: .middle)
            if let status = state.status[station] {
                // A status can carry its own severity on a calm station: Read's "Cursor not read" (R1.4 change 3).
                let own: NDModel.Tone = state.allUnconfirmed ? .fine : status.severity == .red ? .fault : status.severity == .orange ? .warn : .fine
                let tone = m.statusTone == .fine ? own : m.statusTone
                let color: Color = tone == .fault ? ND.redText : tone == .warn ? ND.orangeText : ND.t2
                put(ctx, text(status.text, size: 11, color: color, weight: tone == .fine ? .regular : .semibold, mono: true),
                    x: p.x, baseline: p.y + 43, size: 11, anchor: .middle)
            }
        }
    }
}

// MARK: - A station's hit area

/// The ring and the name, as one shape: what a pointer lands on, exactly where the station is drawn.
struct NDStationHitShape: Shape {
    let circle: CGRect
    let label: CGRect?
    /// A station's ring is a circle; the agents' group and the Memory box are rounded rectangles.
    var round: Bool? = nil
    func path(in rect: CGRect) -> Path {
        var p = Path()
        if round ?? (label != nil) { p.addEllipse(in: circle) } else { p.addRoundedRect(in: circle, cornerSize: CGSize(width: 12, height: 12)) }
        if let label { p.addRect(label) }
        return p
    }
}

/// Pointer over the ring or the name: one 1 pt `t2` ring (drawn by the chart), and after a short pause the station's
/// panel; moving into the panel keeps it open. A click pins it; Esc or a click elsewhere closes it (change 17). Keyboard
/// focus is the system's ring, and Space or Return pins the panel as a click does.
struct NDStationButton: View {
    let station: NDStation
    var interactive = true
    @Binding var hovered: NDStation?
    @ObservedObject var presenter: NDPanelPresenter
    let shape: NDStationHitShape
    var content: () -> AnyView?
    var layout: (AnyView) -> NDPipelineLayout.Panel
    @State private var overStation = false
    @FocusState private var focused: Bool

    var body: some View {
        Button { open(pinned: true, focusAction: false) } label: {
            Color.clear.contentShape(shape)
        }
        .buttonStyle(.plain)
        .contentShape(shape)
        // R1.5 change 4: Tab reaches every station; a 3 pt ring in the system focus colour shows where focus is.
        .focusable(interactions: .activate)
        .focused($focused)
        .focusEffectDisabled()
        .overlay {
            if focused {
                NDStationHitShape(circle: shape.circle.insetBy(dx: -4.5, dy: -4.5), label: nil, round: shape.label != nil)
                    .stroke(Color(nsColor: .keyboardFocusIndicatorColor), lineWidth: 3)
            }
        }
        .onKeyPress(keys: [.return, .space, .escape]) { press in
            Self.handle(press.key, station: station, presenter: presenter) { open(pinned: true, focusAction: true) } ? .handled : .ignored
        }
        .onContinuousHover(coordinateSpace: .local) { phase in
            guard interactive else { return }
            switch phase {
            case .active: if !overStation { overStation = true; changed() }
            case .ended: if overStation { overStation = false; changed() }
            }
        }
        .accessibilityLabel(NDPipelineLayout.name(station))
        .accessibilityAddTraits(.isButton)
    }

    /// R1.5 change 4, the station's own key handling: Return or Space opens the panel with focus on its action; Escape
    /// closes it and leaves focus on the station.
    @discardableResult
    static func handle(_ key: KeyEquivalent, station: NDStation, presenter: NDPanelPresenter, open: () -> Void) -> Bool {
        switch key {
        case .return, .space:
            open()
            return true
        case .escape:
            guard presenter.shown == station else { return false }
            presenter.escape()
            return true
        default:
            return false
        }
    }

    private func changed() {
        hovered = overStation ? station : (hovered == station ? nil : hovered)
        presenter.pointer(station, over: overStation) { pinned in open(pinned: pinned, focusAction: false) }
    }

    private func open(pinned: Bool, focusAction: Bool) {
        guard interactive || focusAction, let view = content() else { return }
        presenter.show(station, layout: layout(view), content: view, pinned: pinned, focusAction: focusAction)
    }
}

/// Places the chart's hover panels: one at a time, a borderless child window of the chart's own, positioned from the
/// chart's coordinates so the arrow lands on the station's circle (the lead's 1.4 rule). NSPopover centres its body on
/// the arrow; this panel lets the body slide along its edge while the arrow stays on the station.
@MainActor final class NDPanelPresenter: ObservableObject {
    weak var anchorView: NSView?
    private var panel: NSPanel?
    @Published private(set) var shown: NDStation?
    /// R1.5 change 4: where keyboard focus belongs: the open panel's action, or back on the station after Escape.
    enum FocusTarget: Equatable { case action(NDStation), station(NDStation) }
    @Published private(set) var focusTarget: FocusTarget?
    private var pinned = false
    private var overStation = false
    private var overPanel = false
    private var monitors: [Any] = []
    private var pending: Task<Void, Never>?

    /// The pointer arrived at or left a station: open after 300 ms, close 250 ms after it has left station and panel.
    func pointer(_ station: NDStation, over: Bool, open: @escaping (Bool) -> Void) {
        overStation = over
        pending?.cancel()
        pending = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(over ? 300 : 250))
            guard let self, !Task.isCancelled else { return }
            if over, self.overStation, self.shown != station { open(false) }
            if !over, !self.overStation, !self.overPanel, !self.pinned { self.hide() }
        }
    }

    func show(_ station: NDStation, layout: NDPipelineLayout.Panel, content: AnyView, pinned: Bool, focusAction: Bool = false) {
        hide()
        shown = station
        self.pinned = pinned
        focusTarget = focusAction ? .action(station) : nil
        guard let view = anchorView, let window = view.window else { return }
        // L39: inside the screen's visible frame (the window's own edge may be passed).
        var layout = layout
        if let visible = (window.screen ?? NSScreen.main)?.visibleFrame {
            let limit = view.convert(window.convertFromScreen(visible), from: nil)
            layout = NDPipelineLayout.fitted(layout, within: limit)
        }
        let rect = window.convertToScreen(view.convert(layout.outer, to: nil))
        let p = NDKeyPanel(contentRect: rect, styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: true)
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = true
        p.isReleasedWhenClosed = false
        p.becomesKeyOnlyIfNeeded = !focusAction
        p.appearance = window.effectiveAppearance
        let root = NDPanelChrome(layout: layout, content: content)
            .environment(\.ndPanelFocusAction, focusAction)
            .onHover { [weak self] inside in self?.overPanel = inside }
        p.contentView = NSHostingView(rootView: root)
        window.addChildWindow(p, ordered: .above)
        if focusAction { p.makeKey() }
        panel = p
        // Esc closes it; a click outside a pinned panel closes it. (The monitors run on the main thread.)
        monitors.append(NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            let isEscape = event.keyCode == 53
            let handled = MainActor.assumeIsolated { () -> Bool in
                guard isEscape, let self, self.panel != nil else { return false }
                self.escape()
                return true
            }
            return handled ? nil : event
        } as Any)
        monitors.append(NSEvent.addLocalMonitorForEvents(matching: .leftMouseDown) { [weak self] event in
            let number = event.windowNumber
            MainActor.assumeIsolated {
                if let self, let panel = self.panel, number != panel.windowNumber, self.pinned { self.hide() }
            }
            return event
        } as Any)
    }

    /// Escape: the panel closes and keyboard focus goes back to its station (the chart's window becomes key again).
    func escape() {
        let station = shown
        let parent = panel?.parent
        hide()
        if let station { focusTarget = .station(station) }
        parent?.makeKey()
    }

    func hide() {
        pending?.cancel()
        for m in monitors { NSEvent.removeMonitor(m) }
        monitors = []
        if let panel {
            panel.parent?.removeChildWindow(panel)
            panel.orderOut(nil)
        }
        panel = nil
        shown = nil
        focusTarget = nil
        pinned = false
        overPanel = false
    }
}

/// A borderless panel that can take keyboard focus when it is opened from the keyboard (R1.5 change 4).
final class NDKeyPanel: NSPanel {
    override var canBecomeKey: Bool { true }
}

private struct NDPanelFocusActionKey: EnvironmentKey { static let defaultValue = false }
extension EnvironmentValues {
    /// The panel was opened from the keyboard: its action takes focus.
    var ndPanelFocusAction: Bool {
        get { self[NDPanelFocusActionKey.self] }
        set { self[NDPanelFocusActionKey.self] = newValue }
    }
}

/// The chart's own NSView, in the drawing's coordinates (top-left origin), for placing panels.
struct NDChartAnchor: NSViewRepresentable {
    let presenter: NDPanelPresenter
    final class Flipped: NSView { override var isFlipped: Bool { true } }
    func makeNSView(context: Context) -> NSView {
        let view = Flipped()
        presenter.anchorView = view
        return view
    }
    func updateNSView(_ nsView: NSView, context: Context) { presenter.anchorView = nsView }
}

/// A hover panel's body and its arrow, the arrow at the layout's tip.
struct NDPanelChrome: View {
    let layout: NDPipelineLayout.Panel
    let content: AnyView

    var body: some View {
        let o = layout.outer
        let body = layout.body.offsetBy(dx: -o.minX, dy: -o.minY)
        let tip = CGPoint(x: layout.tip.x - o.minX, y: layout.tip.y - o.minY)
        ZStack(alignment: .topLeading) {
            NDPanelShape(body: body, tip: tip, edge: layout.edge)
                .fill(ND.pop)
            NDPanelShape(body: body, tip: tip, edge: layout.edge)
                .stroke(ND.hair2, lineWidth: 0.5)
            // A body the screen made shorter than its content scrolls (L39); otherwise it is exactly its content.
            ScrollView(.vertical) { content }
                .scrollIndicators(.automatic)
                .frame(width: body.width, height: body.height, alignment: .topLeading)
                .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                .offset(x: body.minX, y: body.minY)
        }
        .frame(width: o.width, height: o.height, alignment: .topLeading)
    }
}

struct NDPanelShape: Shape {
    let body: CGRect
    let tip: CGPoint
    let edge: Edge

    func path(in rect: CGRect) -> Path {
        var p = Path(roundedRect: body, cornerRadius: 8, style: .continuous)  // R1.5 change 2: radius 8
        let w = NDPipelineLayout.arrowHalfWidth
        var arrow = Path()
        switch edge {
        case .top:
            arrow.move(to: CGPoint(x: tip.x - w, y: body.maxY)); arrow.addLine(to: tip); arrow.addLine(to: CGPoint(x: tip.x + w, y: body.maxY))
        case .bottom:
            arrow.move(to: CGPoint(x: tip.x - w, y: body.minY)); arrow.addLine(to: tip); arrow.addLine(to: CGPoint(x: tip.x + w, y: body.minY))
        case .trailing:
            arrow.move(to: CGPoint(x: body.minX, y: tip.y - w)); arrow.addLine(to: tip); arrow.addLine(to: CGPoint(x: body.minX, y: tip.y + w))
        case .leading:
            arrow.move(to: CGPoint(x: body.maxX, y: tip.y - w)); arrow.addLine(to: tip); arrow.addLine(to: CGPoint(x: body.maxX, y: tip.y + w))
        }
        arrow.closeSubpath()
        p.addPath(arrow)
        return p
    }
}

// MARK: - A line's or a frame's hover (R1.4 changes 5, 8, 9)

/// The words a line or a frame used to carry at rest: its title and one line, in a popover after a short pause.
struct NDHoverRegion: View {
    let title: String
    let line: String
    var interactive = true
    var edge: Edge = .top
    @State private var open = false
    @State private var over = false

    var body: some View {
        Color.clear.contentShape(Rectangle())
            .onHover { inside in
                guard interactive else { return }
                over = inside
                Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(inside ? 300 : 200))
                    if over != open { open = over }
                }
            }
            .popover(isPresented: $open, arrowEdge: edge) { Self.card(title: title, line: line) }
            .accessibilityElement()
            .accessibilityLabel(title)
            .accessibilityHint(line)
    }

    static func card(title: String, line: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(NDType.bodyStrong).foregroundStyle(ND.t1)
            Text(line).font(NDType.body).foregroundStyle(ND.t2).fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 14).padding(.vertical, 12)
        .frame(width: 250, alignment: .leading)
    }
}

// MARK: - The station popover (§ F explanations; change 17 faults; L33 conditions)

struct NDStationPopover: View {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    @ObservedObject var store: NDStore
    let station: NDStation

    struct Line: Identifiable, Equatable {
        var id: String { label }
        var label: String
        var value: String
        var tone: NDModel.Tone = .fine
    }

    /// What a panel says (R1.5 change 2): its mark, title, lines (the first in `t1` under a fault), rows and action.
    struct Model: Equatable {
        enum Mark: Equatable { case fault, notice, symbol(String) }
        var mark: Mark
        var title: String
        var lines: [String]
        var firstLineStrong: Bool
        var rows: [Line]
        var action: BrainBarMenuPresentation.Action?
    }

    var body: some View {
        NDStationPanelBody(model: Self.model(station, item: store.r1.stationItems[station], store.inputs, store.model, store.r1.pipeline),
                           waiting: store.inputs.driveReconnecting) { store.perform($0) }
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    static func model(_ station: NDStation, item: NDR1.Item?, _ i: NDInputs, _ m: NDModel, _ p: NDPipelineState) -> Model {
        let symbol = station == .memory ? "brain" : (NDPipelineLayout.symbols[station] ?? "circle")
        let ownLines = lines(station, i, p)
        let ownRows = rows(station, i, m)
        guard let item else {
            return Model(mark: .symbol(symbol), title: title(station), lines: ownLines, firstLineStrong: false, rows: ownRows, action: nil)
        }
        let fault = faultLines(station, item: item, i, p)
        // A notice the station says in its own words (backup-late): its title, then what the station is and does.
        let showOwn = fault.isEmpty
        return Model(mark: item.severity == .red ? .fault : .notice, title: item.title,
                     lines: showOwn ? ownLines : fault, firstLineStrong: !showOwn,
                     rows: showOwn ? ownRows : [], action: item.action)
    }

    static func title(_ station: NDStation) -> String {
        station == .jobs ? Say.jobsPanelTitle : NDPipelineLayout.name(station)
    }

    /// A station's own lines: its explanation (§ F), the providers for the agents, the copies' truth for the Mac (L42),
    /// what the jobs do for Jobs.
    static func lines(_ station: NDStation, _ i: NDInputs, _ p: NDPipelineState) -> [String] {
        switch station {
        case .agents: [Say.agentsLine, Say.explainAgents]
        case .mac: [Say.macKeepsNewest(Say.countWord(NDBackupFacts.keepNewest)), Say.macOlderRemoved]
        case .jobs: [Say.jobsPanelLine]
        default: [explanation(station)]
        }
    }

    /// "Claude, Codex and Antigravity".
    static func list(_ names: [String]) -> String {
        guard names.count > 1 else { return names.first ?? "" }
        return names.dropLast().joined(separator: ", ") + " and " + names.last!
    }

    /// The eight station explanations, as written in CHANGES-R1.3 § F.
    static func explanation(_ station: NDStation) -> String {
        switch station {
        case .agents: Say.explainAgents
        case .read: Say.explainRead
        case .queue: Say.explainQueue
        case .memory: Say.explainMemory
        case .search: Say.explainSearch
        case .mac: Say.explainMac
        case .drive: Say.explainDrive
        case .jobs: Say.explainJobs
        }
    }

    /// A fault's lines under its title (change 17), each only under its condition (L33); otherwise the item's own detail.
    static func faultLines(_ station: NDStation, item: NDR1.Item?, _ i: NDInputs, _ p: NDPipelineState? = nil) -> [String] {
        guard let item else { return [] }
        let checks = i.capture.verdict?.checks ?? []
        func green(_ name: String) -> Bool { checks.first { $0.name == name }?.state == .green }
        switch (station, item.id) {
        case (.jobs, "nightly"):
            var out: [String] = []
            if let at = NDR1.lastNightly(i) {
                out.append(Say.lastGoodRun(NDR1.weekday(at, i.capture), NDR1.hhmm(at, i.capture)))
            }
            out.append(green("stores") && green("search") ? Say.memoryStillSaves : Say.notCheckedOrTidiedSince)
            return out
        case (.agents, "agent"):
            // L45: "{others} are still read": the providers no check names; left out when there are none.
            let named = Set((p ?? NDPipelineState.make(i)).agentFaults.keys)
            let others = NDPipelineLayout.agentRows.map(\.0).filter { !named.contains($0) }.map(NDProvider.name)
            return others.isEmpty ? [Say.agentNotReadDetail] : [Say.othersStillRead(list(others))]
        case (.drive, "drive"):
            var out: [String] = []
            if let newest = NDR1.driveAges(i.capture.verdict).min() { out.append(Say.lastDriveUpload(NDFormat.ageText(newest))) }
            if green("backup_local") { out.append(Say.macCopyStillMade) }
            return out
        default:
            return [item.detail].compactMap { $0 }.filter { !$0.isEmpty }
        }
    }

    /// The rows under an explanation (R1.4 changes 13 and 14): Save queue "Waiting: none" when it is measured empty;
    /// Search "Last check: working" when the search check is green, and the word index that has no check yet.
    static func rows(_ station: NDStation, _ i: NDInputs, _ m: NDModel) -> [Line] {
        switch station {
        case .queue:
            return i.savesWaiting == 0 ? [Line(label: Say.waitingRow, value: Say.none)] : []
        case .mac:
            // R1.5 change 8: Kept (the copies actually found) and Newest.
            var out: [Line] = []
            if let kept = i.localCopiesKept { out.append(Line(label: Say.keptRow, value: "\(kept)")) }
            if let at = i.newestLocalCopyAt {
                out.append(Line(label: Say.newestRow, value: Say.ago(NDFormat.ageText(max(0, i.capture.now.timeIntervalSince(at))))))
            }
            return out
        case .jobs:
            // R1.5 change 7: every job that works on memory, from the Jobs page, with its last run.
            let jobs: [BrainLayerLaunchdJob] = [.maintenanceNightly, .maintenanceWeekly, .walCheckpoint, .decay, .repairFTS]
            return jobs.map { job in
                let last: Date?
                switch job {
                case .maintenanceNightly: if case let .ran(at) = i.nightlyLastRun { last = at } else { last = nil }
                case .maintenanceWeekly: if case let .ran(at) = i.weeklyLastRun { last = at } else { last = nil }
                default: last = i.jobs?.first { $0.job == job }?.lastRun
                }
                let value = last.map { Say.ago(NDFormat.ageText(max(0, i.capture.now.timeIntervalSince($0)))) } ?? NDModel.Number.unknown.value
                return Line(label: Say.jobNames[job.rawValue] ?? job.title, value: value)
            }
        case .search:
            var out: [Line] = []
            if i.capture.verdict?.checks.first(where: { $0.name == "search" })?.state == .green {
                out.append(Line(label: Say.lastCheck, value: Say.working))
            }
            out.append(Line(label: Say.searchByWordsNotChecked, value: ""))
            return out
        default:
            return []
        }
    }

    /// The popover's size at its width, as SwiftUI lays it out (the render harness and the overlap test use it).
    @MainActor static func measure(_ view: AnyView, width: CGFloat) -> CGSize {
        let hosting = NSHostingController(rootView: view.frame(width: width))
        return hosting.sizeThatFits(in: CGSize(width: width, height: CGFloat.greatestFiniteMagnitude))
    }
}
