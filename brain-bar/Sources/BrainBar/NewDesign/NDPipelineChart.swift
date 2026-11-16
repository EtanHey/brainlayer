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

    /// Tab order (change 16): each station, then its action, then the next station.
    static let order: [NDStation] = [.agents, .read, .queue, .memory, .jobs, .search, .mac, .drive]

    struct Control: Equatable, Identifiable {
        var station: NDStation
        var action: BrainBarMenuPresentation.Action?
        var id: String { action == nil ? station.rawValue : station.rawValue + ".action" }
    }

    /// The controls in the key loop, in order: what the view lays out, and what the tests check.
    static func controls(_ items: [NDStation: NDR1.Item]) -> [Control] {
        order.flatMap { s -> [Control] in
            var out = [Control(station: s, action: nil)]
            if let action = items[s]?.action { out.append(Control(station: s, action: action)) }
            return out
        }
    }

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
            Canvas { ctx, _ in Self.draw(ctx, state: state, memories: memories, hovered: shownHover) }
                .frame(width: NDPipelineLayout.drawWidth, height: NDPipelineLayout.size.height)
                .allowsHitTesting(false)
            ForEach(Self.controls(items)) { control in
                if let action = control.action {
                    actionButton(control.station, action)
                } else {
                    station(control.station)
                }
            }
            if let previewHover { preview(previewHover) }
        }
        .frame(width: NDPipelineLayout.size.width, height: NDPipelineLayout.size.height, alignment: .topLeading)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Say.pipeline)
    }

    @ViewBuilder private func station(_ s: NDStation) -> some View {
        let hit = NDPipelineLayout.hitRect(s)
        let spec = NDPipelineLayout.popover(s, actions: Self.actionStations(items))
        NDStationButton(station: s, interactive: interactive, hovered: $hovered, size: hit.size,
                        anchor: spec.anchor.offsetBy(dx: -hit.minX, dy: -hit.minY), edge: spec.edge, popover: { popover(s) })
            .position(x: hit.midX, y: hit.midY)
    }

    @ViewBuilder private func actionButton(_ s: NDStation, _ action: BrainBarMenuPresentation.Action) -> some View {
        let frame = NDPipelineLayout.actionFrame(s, width: Self.actionWidth(action))
        let isReconnect = [.reconnectDrive, .connectDrive].contains(action.kind)
        // R1.2 change 24: the waiting line is right-aligned to the button's right edge.
        NDWaitingSwap(waiting: waiting && isReconnect, alignment: .trailing) {
            NDButton(label: Self.shortLabel(action), filled: NDButtonPolicy.filled(action), compact: true,
                     accessibility: action.label) { perform(action.kind) }
        } line: {
            Text(Say.waitingForBrowser).font(NDType.caption).foregroundStyle(ND.t2).fixedSize()
        }
        .frame(width: frame.width, height: frame.height, alignment: .trailing)
        .position(x: frame.midX, y: frame.midY)
    }

    /// The render harness's hover: the popover drawn at the frame the layout gives it.
    @ViewBuilder private func preview(_ s: NDStation) -> some View {
        if let content = popover(s) {
            let spec = NDPipelineLayout.popover(s, actions: Self.actionStations(items))
            let size = NDStationPopover.measure(content, width: spec.width)
            let frame = NDPipelineLayout.popoverFrame(spec, height: size.height)
            content
                .frame(width: frame.width, height: frame.height, alignment: .topLeading)
                .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(ND.raised).shadow(color: .black.opacity(0.25), radius: 12, y: 4))
                .overlay(RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(ND.hair2, lineWidth: 0.5))
                .offset(x: frame.minX, y: frame.minY)
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

    static func draw(_ ctx: GraphicsContext, state: NDPipelineState, memories: String?, hovered: NDStation? = nil) {
        drawPipes(ctx, state: state)
        drawAgents(ctx, state: state, hovered: hovered)
        drawHub(ctx, state: state, memories: memories, hovered: hovered)
        drawStations(ctx, state: state, hovered: hovered)
        if state.allUnconfirmed {
            let words = state.lastKnownAt.map(Say.valuesLastKnownAt) ?? Say.everyPipeNotConfirmed
            put(ctx, text(words, size: 11, color: ND.t2, weight: .medium), x: NDPipelineLayout.size.width, baseline: 9, size: 11, anchor: .end)
        }
    }

    static func drawPipes(_ ctx: GraphicsContext, state: NDPipelineState) {
        typealias L = NDPipelineLayout
        // Kept safe: the hairline frame around the two copies (it never grows; actions sit under it).
        ctx.stroke(Path(roundedRect: L.keptSafe, cornerRadius: 12, style: .continuous), with: .color(ND.hair2), lineWidth: 1)
        var labels: [(String, CGPoint)] = []
        for c in L.connections {
            let line = c.line
            let total = line.length
            var endShading = stroke(for: c.pipe, in: state).shading
            var startShading = endShading
            switch state.look(c.pipe) {
            case .breakOut:
                // Change 9: a red stretch touching the faulty stage, a 12 pt gap, then everything after it at 28 %.
                let (red, rest) = line.split(18)
                let (_, after) = rest.split(12)
                let tint = NDPipeStroke.severity(.red)
                ctx.stroke(red.path(radius: 8), with: tint.shading, style: tint.style)
                ctx.stroke(after.path(radius: c.corner), with: NDPipeStroke.downstream.shading, style: NDPipeStroke.downstream.style)
                endShading = NDPipeStroke.downstream.shading
                startShading = endShading
            case .breakIn:
                // A stage that refuses what arrives: the pipe, the gap, then a red stretch touching the stage.
                let (up, rest) = line.split(max(0, total - 30))
                let (_, red) = rest.split(12)
                ctx.stroke(up.path(radius: c.corner), with: NDPipeStroke.flowing.shading, style: NDPipeStroke.flowing.style)
                let tint = NDPipeStroke.severity(.red)
                ctx.stroke(red.path(radius: 8), with: tint.shading, style: tint.style)
                endShading = tint.shading
            case .notice:
                // Change 10: an amber stretch into the stage; nothing downstream is quieted.
                let (up, amber) = line.split(max(0, total - 26))
                ctx.stroke(up.path(radius: c.corner), with: NDPipeStroke.flowing.shading, style: NDPipeStroke.flowing.style)
                let tint = NDPipeStroke.severity(.orange)
                ctx.stroke(amber.path(radius: 8), with: tint.shading, style: tint.style)
                endShading = tint.shading
            case .flowing, .dashed:
                let s = stroke(for: c.pipe, in: state)
                ctx.stroke(line.path(radius: c.corner), with: s.shading, style: s.style)
            }
            let pts = line.points
            if pts.count > 1 {
                arrowhead(ctx, end: pts[pts.count - 1], before: pts[pts.count - 2], shading: endShading)
                if c.bothWays { arrowhead(ctx, end: pts[0], before: pts[1], shading: startShading) }
            }
            if let label = c.label { labels.append((label, line.point(at: total * c.labelAt))) }
        }
        for (words, p) in labels { inLine(ctx, words, at: p, color: ND.t2) }
        put(ctx, text(Say.answersToAgentsLane, size: 11, color: ND.t2), x: L.answersLabel.x, baseline: L.answersLabel.y, size: 11, anchor: .start)
        caption(ctx, Say.keptSafe, frame: L.keptSafe, inset: 16)
    }

    /// The agents' group (L1): its frame, its caption, one row per agent; an agent's fault is its row.
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
        caption(ctx, Say.yourAgents, frame: L.agentsFrame, inset: 14)
        var phrase: NDSeverity?
        for (family, y) in L.agentRows {
            let sev = unconf ? nil : state.agentFaults[family]
            if let sev {
                phrase = sev == .red ? .red : (phrase ?? .orange)
                symbol(ctx, sev == .red ? "exclamationmark.circle.fill" : "exclamationmark.triangle.fill",
                       at: CGPoint(x: L.agentsFrame.minX + 14, y: y), size: 12, color: sev == .red ? ND.red : ND.orange)
            }
            let color = sev == .red ? ND.redText : sev == .orange ? ND.orangeText : ND.t2
            put(ctx, text(family.label, size: 11, color: color, weight: sev == nil ? .regular : .semibold),
                x: L.agentsFrame.minX + 24, baseline: y + 4, size: 11, anchor: .start)
        }
        if let phrase {
            put(ctx, text(Say.notBeingRead, size: 11, color: phrase == .red ? ND.redText : ND.orangeText, weight: .semibold),
                x: L.agentPhrase.x, baseline: L.agentPhrase.y, size: 11, anchor: .start)
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
        symbol(ctx, "brain", at: CGPoint(x: hub.midX, y: hub.minY + 25), size: 18, color: ND.t2)
        put(ctx, text(Say.stageMemory, size: 13, color: ND.t1, weight: .semibold), x: hub.midX, baseline: hub.minY + 55, size: 13, anchor: .middle)
        put(ctx, text(memories ?? NDModel.Number.unknown.value, size: 11, color: ND.t2, mono: true), x: hub.midX, baseline: hub.minY + 70,
            size: 11, anchor: .middle)
    }

    /// The badge's centre: on the ring at the station's top-right, at 45° (change 9).
    static func badgeCentre(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x + 11.7, y: p.y - 11.7) }

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
                let rect = CGRect(x: p.x - 16.5, y: p.y - 16.5, width: 33, height: 33)
                ctx.stroke(Path(ellipseIn: rect), with: .color(ring), style: StrokeStyle(lineWidth: m.ringDashed ? 1 : 1.5, dash: m.ringDashed ? [2, 2] : []))
            }
            if hovered == station {
                let rr: CGFloat = t == .fault || t == .notice ? 20 : 16.5
                ctx.stroke(Path(ellipseIn: CGRect(x: p.x - rr, y: p.y - rr, width: 2 * rr, height: 2 * rr)), with: .color(ND.t2), lineWidth: 1)
            }
            let node = Path(ellipseIn: CGRect(x: p.x - 13, y: p.y - 13, width: 26, height: 26))
            ctx.fill(node, with: .color(hovered == station ? ND.raised : ND.node))
            ctx.stroke(node, with: .color(ND.hair2), lineWidth: 1)
            symbol(ctx, L.symbols[station] ?? "circle", at: p, size: 14, color: m.nameRed ? ND.red : ND.t2)
            if let badge = m.badge {
                let b = badgeCentre(p)
                ctx.fill(Path(ellipseIn: CGRect(x: b.x - 7.6, y: b.y - 7.6, width: 15.2, height: 15.2)), with: .color(ND.card))
                symbol(ctx, badge, at: b, size: 12, color: m.ring ?? ND.red)
            }
            put(ctx, text(L.name(station), size: 13, color: m.nameRed ? ND.redText : ND.t1, weight: m.nameRed ? .semibold : .regular),
                x: p.x, baseline: p.y + 29, size: 13, anchor: .middle)
            if let status = state.status[station] {
                let color: Color = m.statusTone == .fault ? ND.redText : m.statusTone == .warn ? ND.orangeText : ND.t2
                put(ctx, text(status.text, size: 11, color: color, weight: m.statusTone == .fine ? .regular : .semibold, mono: true),
                    x: p.x, baseline: p.y + 43, size: 11, anchor: .middle)
            }
        }
    }
}

// MARK: - A station's hit area

/// Pointer over: one 1 pt `t2` ring (drawn by the chart), and after a short pause the station's popover; moving into the
/// popover keeps it open. A click pins it; Esc or a click elsewhere closes it (change 17). Keyboard focus is the
/// system's ring, and Space or Return pins the popover as a click does.
struct NDStationButton: View {
    let station: NDStation
    var interactive = true
    @Binding var hovered: NDStation?
    let size: CGSize
    /// The popover's anchor, in this button's coordinates.
    let anchor: CGRect
    let edge: Edge
    var popover: () -> AnyView?
    @State private var open = false
    @State private var pinned = false
    @State private var overStation = false
    @State private var overPopover = false

    var body: some View {
        Button { pin() } label: {
            Color.clear.frame(width: size.width, height: size.height).contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .ndKeyboardFocus { pin() }
        .onHover { inside in
            guard interactive else { return }
            overStation = inside
            hovered = inside ? station : (hovered == station ? nil : hovered)
            schedule()
        }
        .popover(isPresented: $open, attachmentAnchor: .rect(.rect(anchor)), arrowEdge: edge) {
            (popover() ?? AnyView(EmptyView()))
                .onHover { inside in
                    overPopover = inside
                    schedule()
                }
        }
        .onChange(of: open) { _, now in if !now { pinned = false } }
        .accessibilityLabel(NDPipelineLayout.name(station))
        .accessibilityAddTraits(.isButton)
    }

    private func pin() {
        guard interactive else { return }
        pinned = true
        open = true
    }

    /// Open 300 ms after the pointer arrives; close 250 ms after it has left both the station and the popover.
    private func schedule() {
        let wantOpen = overStation || overPopover
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(wantOpen ? 300 : 250))
            let still = overStation || overPopover
            if wantOpen, still, !open { open = true }
            if !wantOpen, !still, !pinned, open { open = false }
        }
    }
}

// MARK: - The station popover (§ F explanations; change 17 faults; L33 conditions)

struct NDStationPopover: View {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    @ObservedObject var store: NDStore
    let station: NDStation

    struct Line: Identifiable {
        var id: String { label }
        var label: String
        var value: String
        var tone: NDModel.Tone = .fine
    }

    var body: some View {
        let item = store.r1.stationItems[station]
        let spec = NDPipelineLayout.popover(station, actions: NDPipelineChart.actionStations(store.r1.stationItems))
        Self.content(station: station, item: item, faultLines: Self.faultLines(station, item: item, store.inputs),
                     rows: station == .search ? Self.searchRows(store.model) : [],
                     waiting: store.inputs.driveReconnecting) { store.perform($0) }
            .frame(width: spec.width, alignment: .leading)
    }

    @ViewBuilder static func content(station: NDStation, item: NDR1.Item?, faultLines: [String], rows: [Line], waiting: Bool,
                                     perform: @escaping (BrainBarMenuPresentation.ActionKind) -> Void) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            if let item {
                Text(item.title).font(NDType.bodyStrong).foregroundStyle(ND.t1).fixedSize(horizontal: false, vertical: true)
                    .padding(.bottom, 6)
                ForEach(Array(faultLines.enumerated()), id: \.offset) { index, line in
                    Text(keepingUnits(line)).font(NDType.body).foregroundStyle(index == 0 ? ND.t1 : ND.t2)
                        .fixedSize(horizontal: false, vertical: true).padding(.top, index == 0 ? 0 : 4)
                }
                if let action = item.action {
                    HStack {
                        Spacer()
                        NDWaitingSwap(waiting: waiting && [.reconnectDrive, .connectDrive].contains(action.kind)) {
                            NDButton(label: NDPipelineChart.shortLabel(action), filled: NDButtonPolicy.filled(action), small: true,
                                     accessibility: action.label) { perform(action.kind) }
                        } line: {
                            Text(Say.waitingForBrowser).font(NDType.body).foregroundStyle(ND.t2)
                        }
                    }
                    .padding(.top, 10)
                }
            } else {
                Text(NDPipelineLayout.name(station)).font(NDType.bodyStrong).foregroundStyle(ND.t1).padding(.bottom, 6)
                Text(explanation(station)).font(NDType.body).foregroundStyle(ND.t2).fixedSize(horizontal: false, vertical: true)
                if !rows.isEmpty {
                    VStack(spacing: 0) {
                        ForEach(rows) { line in
                            HStack {
                                Text(line.label).font(NDType.body).foregroundStyle(ND.t2)
                                Spacer(minLength: 12)
                                Text(line.value).font(NDType.number)
                                    .foregroundStyle(line.tone == .fault ? ND.redText : line.tone == .warn ? ND.orangeText : ND.t1)
                            }
                            .frame(height: NDGrid.row)
                        }
                    }
                    .padding(.top, 8)
                }
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
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
    static func faultLines(_ station: NDStation, item: NDR1.Item?, _ i: NDInputs) -> [String] {
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
        case (.drive, "drive"):
            var out: [String] = []
            if let newest = NDR1.driveAges(i.capture.verdict).min() { out.append(Say.lastDriveUpload(NDFormat.ageText(newest))) }
            if green("backup_local") { out.append(Say.macCopyStillMade) }
            return out
        default:
            return [item.detail].compactMap { $0 }.filter { !$0.isEmpty }
        }
    }

    /// Search's details: the canary, and the word index that has no check yet (a row here, never on the chart).
    static func searchRows(_ m: NDModel) -> [Line] {
        let rows = m.areas.flatMap(\.rows)
        var out: [Line] = []
        if let r = rows.first(where: { $0.id == "search" }) { out.append(Line(label: r.name, value: r.number.text, tone: r.number.tone)) }
        out.append(Line(label: Say.searchByWordsNoCheck, value: ""))
        return out
    }

    /// The popover's size at its width, as SwiftUI lays it out (the render harness and the overlap test use it).
    @MainActor static func measure(_ view: AnyView, width: CGFloat) -> CGSize {
        let hosting = NSHostingController(rootView: view.frame(width: width))
        return hosting.sizeThatFits(in: CGSize(width: width, height: CGFloat.greatestFiniteMagnitude))
    }
}
