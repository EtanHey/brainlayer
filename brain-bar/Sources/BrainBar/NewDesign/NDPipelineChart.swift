import AppKit
import SwiftUI

/// Visible native static text supplies the four labels Canvas cannot put in the accessibility tree.
struct NDPipelineLabels: NSViewRepresentable {
    func makeNSView(context: Context) -> NDPipelineLabelView { NDPipelineLabelView() }
    func updateNSView(_ view: NDPipelineLabelView, context: Context) {}
}

final class NDPipelineLabelView: NSView {
    override var isFlipped: Bool { true }

    init() {
        super.init(frame: NSRect(origin: .zero, size: NDPipelineLayout.size))
        var labels: [NSTextField] = []
        func add(_ words: String, rect: CGRect, caption: Bool) {
            let field = NSTextField(labelWithString: words)
            field.font = .systemFont(ofSize: 11, weight: caption ? .medium : .regular)
            field.textColor = NDColor.t2
            field.alignment = .center
            field.frame = caption ? rect.insetBy(dx: -4, dy: -4) : rect
            field.drawsBackground = caption
            field.backgroundColor = NDColor.card
            field.setAccessibilityElement(true)
            field.setAccessibilityRole(.staticText)
            field.setAccessibilityLabel(words)
            addSubview(field)
            labels.append(field)
        }
        for (words, rect) in NDPipelineLayout.laneLabelRects() { add(words, rect: rect, caption: true) }
        for station: NDStation in [.jobs, .weekly] {
            if let words = NDPipelineLayout.purpose(station), let rect = NDPipelineLayout.purposeRect(station) {
                add(words, rect: rect, caption: false)
            }
        }
        setAccessibilityElement(false)
        setAccessibilityChildren(labels)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
}

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
    var backupProgress: BrainBarBackupProgress? = nil
    @State private var hovered: NDStation?
    @StateObject private var panels = NDPanelPresenter()

    /// Tab order (change 16): each station, then its action, then the next station.
    static let order: [NDStation] = [.agents, .read, .queue, .memory, .jobs, .weekly, .search, .mac, .drive]

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
        case .copyReport: action.label  // R1.12 (§ C): "Copy a report for your agent" on every surface, then "Copied"
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
            Canvas { ctx, _ in
                let drawn = Self.drawingState(state, progress: backupProgress)
                Self.draw(ctx, state: drawn, memories: memories, hovered: shownHover, keptGrown: keptGrown)
            }
                .frame(width: NDPipelineLayout.drawWidth, height: NDPipelineLayout.size.height)
                .allowsHitTesting(false)
                .accessibilityHidden(true)
            NDPipelineLabels().frame(width: NDPipelineLayout.size.width, height: NDPipelineLayout.size.height)
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
            // The upload's words are Drive's own status line (drawn above); its bar sits in the station's own space.
            if let backupProgress, Self.showsProgress(state, progress: backupProgress) {
                let frame = Self.progressFrame(state)
                BrainBarBackupProgressView(progress: backupProgress, compact: true, muted: state.status[.drive]?.severity != nil)
                    .frame(width: frame.width, height: frame.height)
                    .offset(x: frame.minX, y: frame.minY)
                    .allowsHitTesting(false)
            }
            // One pointer target for the whole column; provider names remain available to VoiceOver.
            ForEach(NDPipelineLayout.agentRows, id: \.0.rawValue) { family, _ in
                let c = NDPipelineLayout.iconCentre(family)
                Color.clear.frame(width: 22, height: 22)
                    .accessibilityElement()
                    .accessibilityLabel(NDProvider.name(family))
                    .allowsHitTesting(false)
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

    static func showsProgress(_ state: NDPipelineState, progress: BrainBarBackupProgress?) -> Bool {
        progress != nil
    }

    /// The bar: 96 × 3, 3 pt under Google Drive's status line box (y 93…96, whole points) and 5.5 pt clear of the
    /// Kept-safe border, so it belongs to its words rather than resting on the border. One place in every state: calm
    /// (the line says what the upload is doing) and faulted (the line keeps the fault's words, the bar turns grey).
    /// Nothing hangs below it, so it never reaches the Memory-to-Search lane.
    static func progressFrame(_ state: NDPipelineState) -> CGRect {
        let p = NDPipelineLayout.point(.drive)
        let lineBottom = NDPipelineLayout.statusTop(.drive) + NDPipelineLayout.statusHeight
        return CGRect(x: p.x - 48, y: lineBottom + 3, width: 96, height: 3)
    }

    /// A calm Drive's status line becomes the upload's line ("Uploading 45%", "Checking copy", "Stalled at 06:26");
    /// a fault keeps its own words.
    static func drawingState(_ state: NDPipelineState, progress: BrainBarBackupProgress?) -> NDPipelineState {
        var drawn = state
        if let progress, state.status[.drive]?.severity == nil {
            drawn.status[.drive] = .init(text: progress.stationLine())
        }
        return drawn
    }

    /// Words currently drawn on the chart: Drive's line as drawn (the upload's words or its fault's) and the bar under it.
    var displayedPanelWords: [NDStation: [CGRect]] {
        guard Self.showsProgress(state, progress: backupProgress) else { return [:] }
        let drawn = Self.drawingState(state, progress: backupProgress)
        return [.drive: NDPipelineLayout.labelRects(.drive, drawn) + [Self.progressFrame(state)]]
    }

    /// Measure each candidate at its own width before choosing an edge (L58b).
    func panel(_ s: NDStation, content: AnyView) -> NDPipelineLayout.Panel {
        let drawn = Self.drawingState(state, progress: backupProgress)
        let measure: (CGFloat) -> CGFloat = { NDStationPopover.measure(content, width: $0).height }
        return NDPipelineLayout.measuredPanel(s, state: drawn, displayedWords: displayedPanelWords,
                                              heightAtWidth: measure)
    }

    static func stationTarget(_ s: NDStation, state: NDPipelineState) -> (frame: CGRect, shape: NDStationHitShape) {
        let hit = NDPipelineLayout.hitShape(s, state)
        let frame = hit.label.map { hit.circle.union($0) } ?? hit.circle
        return (frame, NDStationHitShape(circle: hit.circle.offsetBy(dx: -frame.minX, dy: -frame.minY),
                                        label: hit.label?.offsetBy(dx: -frame.minX, dy: -frame.minY), rectangular: s == .agents))
    }

    @ViewBuilder private func station(_ s: NDStation) -> some View {
        let target = Self.stationTarget(s, state: state)
        let frame = target.frame
        NDStationButton(station: s, interactive: interactive, hovered: $hovered, presenter: panels,
                        shape: target.shape,
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

    /// R1.10 (comp 58): the nightly check-up is a crescent moon with a small check at its top right; the weekly clean a
    /// broom (SF Symbols has none, so it is drawn): a handle to the lower left and a fanned head with three bristles.
    static func jobGlyph(_ ctx: GraphicsContext, _ station: NDStation, at p: CGPoint, size: CGFloat, color: Color) {
        let u = size / 14
        if station == .jobs {
            symbol(ctx, "moon", at: CGPoint(x: p.x - 1.5 * u, y: p.y + 1.2 * u), size: 11.5 * u, color: color)
            symbol(ctx, "checkmark", at: CGPoint(x: p.x + 4.4 * u, y: p.y - 4.6 * u), size: 7.5 * u, color: color)
            return
        }
        func pt(_ x: CGFloat, _ y: CGFloat) -> CGPoint { CGPoint(x: p.x + (x - 7) * u, y: p.y + (y - 7) * u) }
        var handle = Path()
        handle.move(to: pt(12.5, 1.5)); handle.addLine(to: pt(7.6, 6.4))
        var head = Path()
        head.move(to: pt(6.2, 5.0)); head.addLine(to: pt(9.0, 7.8)); head.addLine(to: pt(5.6, 12.4))
        head.addLine(to: pt(1.6, 8.4)); head.closeSubpath()
        var bristles = Path()
        bristles.move(to: pt(3.2, 10.0)); bristles.addLine(to: pt(1.8, 12.2))
        bristles.move(to: pt(4.6, 11.4)); bristles.addLine(to: pt(3.6, 13.2))
        let style = StrokeStyle(lineWidth: 1.3 * u, lineCap: .round, lineJoin: .round)
        ctx.stroke(handle, with: .color(color), style: style)
        ctx.stroke(head, with: .color(color), style: style)
        ctx.stroke(bristles, with: .color(color), style: style)
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
            // CodeRabbit: the last-known caption masks the rail just like the other on-line captions.
            let width = ceil(NSAttributedString(string: words, attributes: [.font: NSFont.systemFont(ofSize: 11, weight: .medium)]).size().width)
            ctx.fill(Path(CGRect(x: NDPipelineLayout.size.width - width - 6, y: -2, width: width + 12, height: 16)),
                     with: .color(ND.card))
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
        symbol(ctx, "brain", at: CGPoint(x: hub.midX, y: hub.minY + 16), size: 16, color: ND.t2)
        put(ctx, text(Say.stageMemory, size: 13, color: ND.t1, weight: .semibold), x: hub.midX, baseline: hub.minY + 39, size: 13, anchor: .middle)
        // R2A item 8: the count carries its unit ("297,412 memories"), on two lines when one would crowd the box.
        let count = memories ?? NDModel.Number.unknown.value
        let words = Say.memoriesCount(count)
        let tint = state.allUnconfirmed ? ND.t3 : ND.t2  // R1.12 (E2): the Memory count follows the last-known rule
        if NDPipelineLayout.textWidth(words, size: 11) <= hub.width - 16 {
            put(ctx, text(words, size: 11, color: tint, mono: true), x: hub.midX, baseline: hub.minY + 60, size: 11, anchor: .middle)
        } else {
            put(ctx, text(count, size: 11, color: tint, mono: true), x: hub.midX, baseline: hub.minY + 54, size: 11, anchor: .middle)
            put(ctx, text(Say.memoriesWord, size: 11, color: tint), x: hub.midX, baseline: hub.minY + 68, size: 11, anchor: .middle)
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
        // R2 item 10: the name keeps its normal colour and weight; only the short secondary line is red.
        case .fault: Marks(ring: ND.red, badge: "exclamationmark.circle.fill", nameRed: false, statusTone: .fault)
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
            } else if station == .jobs || station == .weekly {
                Self.jobGlyph(ctx, station, at: p, size: 14, color: m.nameRed ? ND.red : ND.t2)
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
                // R1.12 (E2): not confirmed, the stations keep their last-known values in `t3`, as clock times.
                let color: Color = state.allUnconfirmed ? ND.t3 : tone == .fault ? ND.redText : tone == .warn ? ND.orangeText : ND.t2
                put(ctx, text(status.text, size: 11, color: color, weight: tone == .fine ? .regular : .semibold, mono: true),
                    x: p.x, baseline: L.statusTop(station) + 10, size: 11, anchor: .middle)
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
    var rectangular = false
    func path(in rect: CGRect) -> Path {
        if rectangular { return Path(circle) }
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
    /// The spoken name, when it is not the station's (a failing provider's icon names the provider).
    var spoken: String? = nil
    var content: () -> AnyView?
    var layout: (AnyView) -> NDPipelineLayout.Panel
    @State private var overStation = false
    @FocusState private var focused: Bool

    var body: some View {
        // R1.8 item 4: one keyboard stop per station. It was a Button inside a focusable wrapper: with Full Keyboard
        // Access the inner button was a second stop that drew no ring (every other Tab looked like nothing happened).
        Color.clear
            .contentShape(shape)
            .onTapGesture { open(pinned: true, focusAction: false) }
        // R1.5 change 4: Tab reaches every station; a 3 pt ring in the system focus colour shows where focus is.
        .focusable(interactions: .activate)
        .focused($focused)
        .focusEffectDisabled()
        .ndDecoration(Group {
            if focused {
                NDStationHitShape(circle: shape.circle.insetBy(dx: -4.5, dy: -4.5), label: nil, round: shape.label != nil)
                    .stroke(Color(nsColor: .keyboardFocusIndicatorColor), lineWidth: 3)
            }
        })
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
        .accessibilityLabel(spoken ?? NDPipelineLayout.name(station))
        .accessibilityAddTraits(.isButton)
        .accessibilityAction { open(pinned: true, focusAction: false) }
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

    /// The pointer arrived at or left a station: open after 300 ms.
    /// R1.6 item 1 (R2 decision 1): a panel closes the moment the pointer leaves (no delay; a 100 ms fade, called off if the pointer
    /// reaches the panel within it), and never stays up while another station is highlighted.
    static let closeDelay: Double = 0
    static let closeFade: Double = 0.1

    func pointer(_ station: NDStation, over: Bool, open: @escaping (Bool) -> Void) {
        overStation = over
        pending?.cancel()
        if over, let shown, shown != station { hide() }  // another station is highlighted: the old panel goes at once
        if over {
            if panel?.alphaValue ?? 1 < 1 { panel?.animator().alphaValue = 1 }
            pending = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .milliseconds(300))
                guard let self, !Task.isCancelled, self.overStation, self.shown != station else { return }
                open(false)
            }
        } else {
            leave()
        }
    }

    /// The pointer left the station or the panel: fade out at once; if it came back over either, stay.
    private func leave() {
        guard !overStation, !overPanel, !pinned, shown != nil else { return }
        guard let panel else { hide(); return }
        NSAnimationContext.runAnimationGroup({ context in
            context.duration = Self.closeFade
            panel.animator().alphaValue = 0
        }, completionHandler: { [weak self] in
            MainActor.assumeIsolated {
                guard let self, self.panel === panel else { return }
                if self.overStation || self.overPanel || self.pinned { panel.animator().alphaValue = 1 } else { self.hide() }
            }
        })
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
            .onHover { [weak self] inside in
                guard let self else { return }
                self.overPanel = inside
                if inside { self.panel?.animator().alphaValue = 1 } else { self.leave() }
            }
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
            // R2 item 2: exactly its content; it scrolls only when the screen made it shorter (L39).
            Group {
                if layout.scrolls {
                    ScrollView(.vertical) { content }.scrollIndicators(.automatic)
                } else {
                    content
                }
            }
            .frame(width: body.width, height: body.height, alignment: .topLeading)
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .circular))
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
        var p = Path()
        let r = min(8, body.width / 2, body.height / 2)
        let w = NDPipelineLayout.arrowHalfWidth
        p.move(to: CGPoint(x: body.minX + r, y: body.minY))
        if edge == .bottom {
            p.addLine(to: CGPoint(x: tip.x - w, y: body.minY))
            p.addLine(to: tip)
            p.addLine(to: CGPoint(x: tip.x + w, y: body.minY))
        }
        p.addLine(to: CGPoint(x: body.maxX - r, y: body.minY))
        p.addArc(tangent1End: CGPoint(x: body.maxX, y: body.minY), tangent2End: CGPoint(x: body.maxX, y: body.minY + r), radius: r)
        if edge == .leading {
            p.addLine(to: CGPoint(x: body.maxX, y: tip.y - w))
            p.addLine(to: tip)
            p.addLine(to: CGPoint(x: body.maxX, y: tip.y + w))
        }
        p.addLine(to: CGPoint(x: body.maxX, y: body.maxY - r))
        p.addArc(tangent1End: CGPoint(x: body.maxX, y: body.maxY), tangent2End: CGPoint(x: body.maxX - r, y: body.maxY), radius: r)
        if edge == .top {
            p.addLine(to: CGPoint(x: tip.x + w, y: body.maxY))
            p.addLine(to: tip)
            p.addLine(to: CGPoint(x: tip.x - w, y: body.maxY))
        }
        p.addLine(to: CGPoint(x: body.minX + r, y: body.maxY))
        p.addArc(tangent1End: CGPoint(x: body.minX, y: body.maxY), tangent2End: CGPoint(x: body.minX, y: body.maxY - r), radius: r)
        if edge == .trailing {
            p.addLine(to: CGPoint(x: body.minX, y: tip.y + w))
            p.addLine(to: tip)
            p.addLine(to: CGPoint(x: body.minX, y: tip.y - w))
        }
        p.addLine(to: CGPoint(x: body.minX, y: body.minY + r))
        p.addArc(tangent1End: CGPoint(x: body.minX, y: body.minY), tangent2End: CGPoint(x: body.minX + r, y: body.minY), radius: r)
        p.closeSubpath()
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
                           waiting: NDDriveWaiting.shows(store.inputs, for: store.r1.stationItems[station]?.action),
                           compactRows: station == .search,  // w303 review: Drive keeps its 8 pt above the rows while uploading
                           compactAction: station == .search,
                           allowsWideLayout: station == .read) { store.perform($0) }  // w303: not the copies
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
        // R1.12 (5e): Drive's fault panel is its title, the two rows and Reconnect, so it fits the room above it.
        let showOwn = fault.isEmpty && !(station == .drive && item.id == "drive")
        return Model(mark: item.severity == .red ? .fault : .notice, title: faultTitle(station, item, i),
                     lines: showOwn ? ownLines : fault, firstLineStrong: !showOwn,
                     rows: showOwn || station == .drive ? ownRows : [], action: item.action)
    }

    /// R2 item 11: the jobs fault panel's title, short (the station is named above it). R1.10 keeps it short: comp 59's
    /// two-line "Nightly check-up failed 3 nights in a row" makes the panel too tall for the room under Save queue.
    /// R1.12 (§ C): one wording per fault: the panel's title is the item's, the same sentence as its row, its page and its
    /// dropdown block ("Nightly check-up failed 3 nights").
    static func faultTitle(_ station: NDStation, _ item: NDR1.Item, _ i: NDInputs) -> String { item.title }

    static func title(_ station: NDStation) -> String { NDPipelineLayout.name(station) }  // R1.10: each job its own panel

    /// A station's own lines: its explanation (§ F), the providers for the agents, the copies' truth for the Mac (L42),
    /// what the jobs do for Jobs.
    static func lines(_ station: NDStation, _ i: NDInputs, _ p: NDPipelineState) -> [String] {
        let checks = i.capture.verdict?.checks ?? []
        switch station {
        case .agents: return [Say.agentsLine, Say.explainAgents]
        // R1.12 (I 12): shorter, the same facts as L45.
        case .mac: return [Say.localCopyPanel(Say.countWord(NDBackupFacts.keepNewest))]
        // R1.12 (E3, I 11): with the local copy late, nothing new was made to upload.
        case .drive where checks.contains(where: { $0.name == "backup_local" && $0.state == .amber }):
            return [Say.nothingNewToUpload]
        // R1.12 (F, I 10): the second sentence only while the search check is green.
        case .search:
            let worked = checks.contains { $0.name == "search" && $0.state == .green }
            return [Say.searchFinds + (worked ? " " + Say.searchLastWorked : "")]
        default: return [explanation(station)]
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
        case .jobs: Say.explainNightly  // R1.10 (L51)
        case .weekly: Say.explainWeekly
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
            return []  // R1.12 (5e): the title, the two rows (Database, Sessions) and Reconnect
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
        case .drive:
            // R2 item 4: the two uploads Drive receives, each its own age (the station shows the database copy's).
            let checks = i.capture.verdict?.checks ?? []
            func age(_ name: String) -> String? {
                checks.first { $0.name == name }?.ageS.flatMap { $0.isFinite && $0 >= 0 ? Say.ago(NDFormat.ageText($0)) : nil }
            }
            // The panel is a short strip above the station with no room for another row: a running upload's start and its
            // last checkpoint are on the Backups page; Drive's line says when it stopped moving ("Stalled at 06:26").
            return [age("backup_offsite").map { Line(label: Say.databaseCopy, value: $0) },
                    age("transcripts").map { Line(label: Say.conversationsCopy, value: $0) }].compactMap { $0 }
        case .mac:
            // R1.5 change 8: Kept (the copies actually found) and Newest.
            var out: [Line] = []
            if let kept = i.localCopiesKept { out.append(Line(label: Say.keptRow, value: "\(kept)")) }
            // R1.10 (L50): the newest copy as a clock time on its day.
            if let at = i.newestLocalCopyAt { out.append(Line(label: Say.newestRow, value: NDR1.dayAt(at, i.capture))) }
            return out
        case .weekly:
            return []  // R1.12 (5h): its rows go; its status shows the last run and the Jobs page the next
        case .jobs:
            // R1.10 (comp 59): the job's own last and next run, from what the Jobs page reads.
            let job: BrainLayerLaunchdJob = station == .jobs ? .maintenanceNightly : .maintenanceWeekly
            let input = i.jobs?.first { $0.job == job }
            let ran: NDRun = station == .jobs ? i.nightlyLastRun : i.weeklyLastRun
            let now = i.capture.now
            var last: Date?
            if case let .ran(at) = ran { last = at } else { last = input?.lastRun }
            let lastValue = last.map { Say.ago(NDFormat.ageText(max(0, now.timeIntervalSince($0)))) } ?? NDModel.Number.unknown.value
            let nextValue = input?.nextRun.map { Say.inAge(NDFormat.ageText(max(60, $0.timeIntervalSince(now)))) } ?? NDModel.Number.unknown.value
            return [Line(label: Say.lastRun, value: lastValue), Line(label: Say.nextRunHead, value: nextValue)]
        case .search:
            // R1.12 (F; Fable 11): the row the station and the dropdown say, "By meaning 99% searchable"; no row for the
            // word index under "All good".
            return i.searchable.map { [Line(label: Say.stageMeaning, value: Say.searchablePercent(Int(($0 * 100).rounded(.down))))] } ?? []
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
