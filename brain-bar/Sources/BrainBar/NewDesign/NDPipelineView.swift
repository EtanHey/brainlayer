import AppKit
import SwiftUI

// MARK: - Polylines with rounded corners (the comp's `rpath`, `split`, `at`)

struct NDPolyline: Equatable, Sendable {
    var points: [CGPoint]

    static func seg(_ a: CGPoint, _ b: CGPoint) -> CGFloat { hypot(b.x - a.x, b.y - a.y) }

    var length: CGFloat {
        guard points.count > 1 else { return 0 }
        var total: CGFloat = 0
        for i in 1..<points.count { total += Self.seg(points[i - 1], points[i]) }
        return total
    }

    /// The polyline cut at distance `d` from its start: the part before and the part after.
    func split(_ d: CGFloat) -> (NDPolyline, NDPolyline) {
        guard points.count > 1 else { return (self, self) }
        var walked: CGFloat = 0
        for i in 1..<points.count {
            let a = points[i - 1], b = points[i]
            let l = Self.seg(a, b)
            if walked + l >= d, l > 0 {
                let t = max(0, min(1, (d - walked) / l))
                let p = CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t)
                return (NDPolyline(points: Array(points[..<i]) + [p]), NDPolyline(points: [p] + Array(points[i...])))
            }
            walked += l
        }
        return (self, NDPolyline(points: [points[points.count - 1], points[points.count - 1]]))
    }

    func point(at d: CGFloat) -> CGPoint { split(d).1.points.first ?? .zero }

    /// The path with corners rounded by quadratic curves of radius `r` (never more than half a segment).
    func path(radius r: CGFloat) -> Path {
        var p = Path()
        guard let first = points.first, points.count > 1 else { return p }
        p.move(to: first)
        if points.count > 2 {
            for i in 1..<(points.count - 1) {
                let a = points[i - 1], c = points[i], b = points[i + 1]
                let la = Self.seg(a, c), lb = Self.seg(c, b)
                guard la > 0, lb > 0 else { continue }
                let rr = min(r, la / 2, lb / 2)
                let p1 = CGPoint(x: c.x + (a.x - c.x) / la * rr, y: c.y + (a.y - c.y) / la * rr)
                let p2 = CGPoint(x: c.x + (b.x - c.x) / lb * rr, y: c.y + (b.y - c.y) / lb * rr)
                p.addLine(to: p1)
                p.addQuadCurve(to: p2, control: c)
            }
        }
        p.addLine(to: points[points.count - 1])
        return p
    }

    /// Trimmed by `start` at the start and `end` at the end (a station's radius).
    func trimmed(start: CGFloat = 0, end: CGFloat = 0) -> NDPolyline {
        var line = self
        if start > 0 { line = line.split(start).1 }
        if end > 0 { line = line.split(line.length - end).0 }
        return line
    }
}

// MARK: - Layout (852 × 262 pt, DESIGN-SPEC-R1 § 3.2 as revised by R1.1 change 1), as data

enum NDPipelineLayout {
    /// R1.2 change 18: 820 wide inside its surface, lanes 90 pt apart.
    static let size = CGSize(width: 820, height: 246)
    static let yRail: CGFloat = 24, yTop: CGFloat = 70, yBottom: CGFloat = 160, yWords: CGFloat = 137
    static let junction = CGPoint(x: 132, y: 70)
    static let stationRadius: CGFloat = 13
    /// The hub: an outline at x 338, y 54, 108 × 138 (change 9).
    static let hub = CGRect(x: 338, y: 54, width: 108, height: 122)
    /// "Kept safe": a hairline frame around the two copies (change 6; R1.2 x 486).
    static let keptSafe = CGRect(x: 486, y: 44, width: 292, height: 82)
    /// Agents: a list at a 16 pt pitch centred on the top lane (change 17a); names right-aligned at x 64, at most 60 pt.
    static let agentPitch: CGFloat = 16
    static let agents: [(AgentFamily, CGFloat)] = {
        let all = AgentFamily.allCases
        let first = yTop - agentPitch * CGFloat(all.count - 1) / 2
        return all.enumerated().map { ($0.element, first + CGFloat($0.offset) * agentPitch) }
    }()

    static let stations: [NDStation: CGPoint] = [
        .read: CGPoint(x: 204, y: 70), .queue: CGPoint(x: 204, y: 160), .mac: CGPoint(x: 552, y: 70),
        .drive: CGPoint(x: 700, y: 70), .meaning: CGPoint(x: 552, y: 160), .search: CGPoint(x: 700, y: 160),
        .jobs: CGPoint(x: 392, y: 222),
    ]

    static let symbols: [NDStation: String] = [
        .read: "text.bubble", .queue: "tray.full", .memory: "brain", .mac: "laptopcomputer", .drive: "icloud",
        .meaning: "sparkle.magnifyingglass", .search: "magnifyingglass", .jobs: "clock",
    ]

    static func name(_ station: NDStation) -> String {
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        switch station {
        case .read: return Say.stageRead
        case .queue: return Say.stageQueue
        case .memory: return Say.stageMemory
        case .mac: return Say.stageMac
        case .drive: return Say.stageDrive
        case .meaning: return Say.stageMeaning
        case .search: return Say.search
        case .jobs: return Say.jobs
        }
    }

    /// One drawn connection: its points, an optional label set in the line (at a fraction of its length), and an
    /// arrowhead before its destination (change 4). Data, not drawing: revision files change these, not the renderer.
    struct Connection: Sendable {
        var pipe: NDPipe
        var line: NDPolyline
        var corner: CGFloat = 8
        var label: String? = nil
        var labelAt: CGFloat = 0.5
        var arrowhead = true
    }

    static func point(_ s: NDStation) -> CGPoint { stations[s] ?? .zero }

    static let connections: [Connection] = {
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        let r = stationRadius
        let hubRight = hub.maxX
        var out: [Connection] = []
        for (family, y) in agents {
            out.append(Connection(pipe: NDPipe.agent(family),
                                  line: NDPolyline(points: [CGPoint(x: 72, y: y), CGPoint(x: 100, y: y), junction]), arrowhead: false))
        }
        out.append(Connection(pipe: .junctionRead, line: NDPolyline(points: [junction, point(.read)]).trimmed(end: r)))
        let rail = NDPolyline(points: [CGPoint(x: 146, y: yTop), CGPoint(x: 158, y: yTop), CGPoint(x: 158, y: yRail),
                                       CGPoint(x: point(.drive).x, y: yRail), point(.drive)]).trimmed(end: r)
        out.append(Connection(pipe: .conversationsCopy, line: rail, corner: 10, label: Say.conversationsCopiedAsTheyAre, labelAt: 0.42))
        out.append(Connection(pipe: .readMemory, line: NDPolyline(points: [point(.read), CGPoint(x: hub.minX, y: yTop)]).trimmed(start: r)))
        out.append(Connection(pipe: .inSaves, line: NDPolyline(points: [CGPoint(x: 136, y: yBottom), CGPoint(x: point(.queue).x - r, y: yBottom)])))
        out.append(Connection(pipe: .queueMemory, line: NDPolyline(points: [point(.queue), CGPoint(x: hub.minX, y: yBottom)]).trimmed(start: r)))
        out.append(Connection(pipe: .memoryMac, line: NDPolyline(points: [CGPoint(x: hubRight, y: yTop), point(.mac)]).trimmed(end: r)))
        out.append(Connection(pipe: .macDrive, line: NDPolyline(points: [point(.mac), point(.drive)]).trimmed(start: r, end: r)))
        out.append(Connection(pipe: .memoryMeaning, line: NDPolyline(points: [CGPoint(x: hubRight, y: yBottom), point(.meaning)]).trimmed(end: r)))
        out.append(Connection(pipe: .meaningSearch, line: NDPolyline(points: [point(.meaning), point(.search)]).trimmed(start: r, end: r)))
        // Change 7: a straight track from Memory at y 148 to x 712, then a 12 pt corner down into Search's top.
        let words = NDPolyline(points: [CGPoint(x: hubRight, y: yWords), CGPoint(x: point(.search).x, y: yWords),
                                        point(.search)]).trimmed(end: r)
        out.append(Connection(pipe: .words, line: words, corner: 12, label: Say.byWordsNotConfirmed, labelAt: 0.4))
        out.append(Connection(pipe: .out, line: NDPolyline(points: [CGPoint(x: point(.search).x + r, y: yBottom),
                                                                    CGPoint(x: size.width - 6, y: yBottom)])))
        out.append(Connection(pipe: .jobsMemory, line: NDPolyline(points: [point(.jobs), CGPoint(x: point(.jobs).x, y: hub.maxY)]).trimmed(start: r)))
        return out
    }()
}

// MARK: - Strokes as tokens per state (change 8: one meaning per line style)

struct NDPipeStroke: Equatable {
    var color: Color
    var width: CGFloat
    var dash: [CGFloat]
    var opacity: Double = 1

    /// Flowing, and a check confirms it.
    /// R1.2 change 18: routine lighter (1.5 pt), breaks louder (2.5 pt).
    static let flowing = NDPipeStroke(color: ND.pipe, width: 1.5, dash: [])
    /// Not confirmed: the only meaning of a dash, always with words beside it.
    static let unconfirmed = NDPipeStroke(color: ND.pipe, width: 1.5, dash: [3, 4])
    /// Downstream of a break: solid at 28 %.
    static let downstream = NDPipeStroke(color: ND.pipe, width: 1.5, dash: [], opacity: 0.28)
    /// The word path: nothing checks it.
    static let unguarded = NDPipeStroke(color: ND.pipe, width: 1.6, dash: [3, 4])
    static func severity(_ s: NDSeverity) -> NDPipeStroke {
        NDPipeStroke(color: s == .red ? ND.red : ND.orange, width: 2.5, dash: [])
    }

    var shading: GraphicsContext.Shading { .color(color.opacity(opacity)) }
    var style: StrokeStyle { StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round, dash: dash) }
}

// MARK: - The chart

/// The pipeline (H3): eight stages on lines, each with a status line; breaks drawn at their side with one mark. One
/// `Canvas` for the pipes, the hub and the words; the stations are buttons over it (hover ring, click opens details).
struct NDPipelineChart: View {
    let state: NDPipelineState
    /// The memory count in the hub ("297,412").
    let memories: String?
    var interactive = true
    var popover: (NDStation) -> AnyView? = { _ in nil }
    /// The station under the pointer: its fill becomes `raised` (change 15).
    @State private var hovered: NDStation?

    var body: some View {
        ZStack(alignment: .topLeading) {
            Canvas { ctx, _ in Self.draw(ctx, state: state, memories: memories, hovered: hovered) }
                .frame(width: NDPipelineLayout.size.width, height: NDPipelineLayout.size.height)
            ForEach(NDStation.allCases.filter { $0 != .memory }) { station in
                NDStationButton(station: station, interactive: interactive, hovered: $hovered,
                                fault: state.faultStations.contains(station) && !state.allUnconfirmed, popover: { popover(station) })
                    .position(NDPipelineLayout.point(station))
            }
            NDStationButton(station: .memory, interactive: interactive, hovered: $hovered, hubSize: NDPipelineLayout.hub.size,
                            popover: { popover(.memory) })
                .position(x: NDPipelineLayout.hub.midX, y: NDPipelineLayout.hub.midY)
        }
        .frame(width: NDPipelineLayout.size.width, height: NDPipelineLayout.size.height, alignment: .topLeading)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(BrainBarMenuPresenter.SurfaceSay.pipeline)
    }

    // MARK: drawing

    static func stroke(for pipe: NDPipe, in state: NDPipelineState) -> NDPipeStroke {
        if pipe == .words { return .unguarded }
        if state.allUnconfirmed { return .unconfirmed }
        if state.quiet[pipe] != nil { return .downstream }
        if state.unconfirmed.contains(pipe) { return .unconfirmed }
        return .flowing
    }

    static func text(_ s: String, size: CGFloat, color: Color, weight: Font.Weight = .regular, mono: Bool = false) -> Text {
        let font = mono ? Font.system(size: size, weight: weight).monospacedDigit() : Font.system(size: size, weight: weight)
        return Text(s).font(font).foregroundColor(color)
    }

    /// Text placed by its baseline, as the comp's SVG places it.
    static func put(_ ctx: GraphicsContext, _ t: Text, x: CGFloat, baseline: CGFloat, size: CGFloat,
                    anchor: NDPipelineState.Phrase.Anchor) {
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

    /// A 5 pt open chevron, its tip 1.5 pt before the line's end, along the last segment (change 4).
    static func arrowhead(_ ctx: GraphicsContext, _ line: NDPolyline, shading: GraphicsContext.Shading) {
        guard line.points.count > 1 else { return }
        let end = line.points[line.points.count - 1], before = line.points[line.points.count - 2]
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

    /// A word set in a line on a plane mask.
    static func inLine(_ ctx: GraphicsContext, _ words: String, at p: CGPoint, color: Color) {
        let width = ceil(NSAttributedString(string: words, attributes: [.font: NSFont.systemFont(ofSize: 11)]).size().width) + 12
        ctx.fill(Path(CGRect(x: p.x - width / 2, y: p.y - 8, width: width, height: 16)), with: .color(ND.card))
        ctx.draw(text(words, size: 11, color: color), at: p, anchor: .center)
    }

    static func draw(_ ctx: GraphicsContext, state: NDPipelineState, memories: String?, hovered: NDStation? = nil) {
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        typealias L = NDPipelineLayout
        var marks: [(CGPoint, NDSeverity)] = []
        var labels: [(String, CGPoint, Color)] = []
        // "Kept safe": the hairline frame around the two copies, its caption in the top border.
        ctx.stroke(Path(roundedRect: L.keptSafe, cornerRadius: 12, style: .continuous), with: .color(ND.hair2), lineWidth: 1)
        // R1.2 change 15: the word path is drawn only where it explains what still works (search by meaning behind).
        let wordsShown = state.breaks[.memoryMeaning] != nil && !state.allUnconfirmed
        let ordered = (wordsShown ? L.connections.filter { $0.pipe == .words } : []) + L.connections.filter { $0.pipe != .words }
        for c in ordered {
            let line = c.line
            var last: GraphicsContext.Shading = stroke(for: c.pipe, in: state).shading
            if !state.allUnconfirmed, state.stationPipes.contains(c.pipe) {
                // The stage refuses: the pipe arrives in `pipe`, its last 30 pt in red.
                let (up, red) = line.split(max(0, line.length - 30))
                ctx.stroke(up.path(radius: c.corner), with: NDPipeStroke.flowing.shading, style: NDPipeStroke.flowing.style)
                ctx.stroke(red.path(radius: c.corner), with: .color(ND.red), style: NDPipeStroke.severity(.red).style)
                last = .color(ND.red)
            } else if !state.allUnconfirmed, let br = state.breaks[c.pipe] {
                let total = line.length, gap: CGFloat = 20
                let start: CGFloat = br.side == .source ? 5 : max(0, total - gap - 6)
                let (a, rest) = line.split(start)
                let (_, b) = rest.split(gap)
                let tint = NDPipeStroke.severity(br.severity)
                if br.side == .source {
                    ctx.stroke(a.path(radius: c.corner), with: tint.shading, style: tint.style)
                    ctx.stroke(b.path(radius: c.corner), with: NDPipeStroke.downstream.shading, style: NDPipeStroke.downstream.style)
                    last = NDPipeStroke.downstream.shading
                } else {
                    let (up, end) = a.split(max(0, a.length - 26))
                    ctx.stroke(up.path(radius: c.corner), with: NDPipeStroke.flowing.shading, style: NDPipeStroke.flowing.style)
                    ctx.stroke(end.path(radius: c.corner), with: tint.shading, style: tint.style)
                    ctx.stroke(b.path(radius: c.corner), with: tint.shading, style: tint.style)
                    last = tint.shading
                }
                marks.append((line.point(at: start + gap / 2), br.severity))
            } else {
                let s = stroke(for: c.pipe, in: state)
                ctx.stroke(line.path(radius: c.corner), with: s.shading, style: s.style)
                // Downstream of a break, long enough: the word in the line ("delayed" after a late stage, "stopped").
                if let sev = state.quiet[c.pipe], !state.allUnconfirmed, line.length >= 80 {
                    labels.append((sev == .orange ? Say.delayed : Say.stopped, line.point(at: line.length / 2), ND.t3))
                }
            }
            if c.arrowhead, !(state.breaks[c.pipe]?.side == .target && !state.allUnconfirmed) {
                arrowhead(ctx, line, shading: last)
            }
            if let label = c.label { labels.append((label, line.point(at: line.length * c.labelAt), c.pipe == .words ? ND.t3 : ND.t2)) }
        }
        ctx.fill(Path(ellipseIn: CGRect(x: L.junction.x - 3, y: L.junction.y - 3, width: 6, height: 6)), with: .color(ND.pipe))
        for (words, p, color) in labels { inLine(ctx, words, at: p, color: color) }
        // The frame's caption, set in its top border.
        let caption = Say.keptSafe
        let cw = ceil(NSAttributedString(string: caption, attributes: [.font: NSFont.systemFont(ofSize: 11, weight: .medium)]).size().width)
        ctx.fill(Path(CGRect(x: L.keptSafe.minX + 12, y: L.keptSafe.minY - 8, width: cw + 8, height: 16)), with: .color(ND.card))
        put(ctx, text(caption, size: 11, color: ND.t3, weight: .medium), x: L.keptSafe.minX + 16, baseline: L.keptSafe.minY + 4, size: 11, anchor: .start)
        // Memory: an outline (change 9).
        let hub = L.hub
        ctx.stroke(Path(roundedRect: hub, cornerRadius: 12, style: .continuous), with: .color(ND.hair2), lineWidth: 1)
        symbol(ctx, "brain", at: CGPoint(x: hub.midX, y: hub.minY + 44), size: 20, color: ND.t2)
        put(ctx, text(Say.stageMemory, size: 13, color: ND.t1, weight: .semibold), x: hub.midX, baseline: hub.minY + 74, size: 13, anchor: .middle)
        put(ctx, text(memories ?? NDModel.Number.unknown.value, size: 11, color: ND.t2, mono: true), x: hub.midX, baseline: hub.minY + 89,
            size: 11, anchor: .middle)
        // Stations, their names (t1) and status lines.
        // Stale values stay readable (full `t2`); R1.2 change 12 says "last known" once, in the top-right caption.
        let faded = 1.0
        for station in NDStation.allCases where station != .memory {
            let p = L.point(station)
            let fault = state.faultStations.contains(station) && !state.allUnconfirmed
            if fault {
                ctx.stroke(Path(ellipseIn: CGRect(x: p.x - 16.5, y: p.y - 16.5, width: 33, height: 33)), with: .color(ND.red), lineWidth: 1.5)
            }
            let node = Path(ellipseIn: CGRect(x: p.x - 13, y: p.y - 13, width: 26, height: 26))
            ctx.fill(node, with: .color(hovered == station ? ND.raised : ND.node))
            ctx.stroke(node, with: .color(ND.hair2), lineWidth: 1)
            symbol(ctx, L.symbols[station] ?? "circle", at: p, size: 14, color: fault ? ND.red : ND.t2)
            let nameColor = fault ? ND.redText : ND.t1
            let weight: Font.Weight = fault ? .semibold : .regular
            let status = state.status[station]
            let statusColor: Color = state.allUnconfirmed ? ND.t2 : status?.severity == .red ? ND.redText
                : status?.severity == .orange ? ND.orangeText : ND.t2
            let statusWeight: Font.Weight = status?.severity != nil && !state.allUnconfirmed ? .semibold : .regular
            if station == .jobs {
                put(ctx, text(L.name(station), size: 13, color: nameColor, weight: weight), x: p.x + 22, baseline: p.y - 1, size: 13, anchor: .start)
                if let status {
                    var c = ctx
                    c.opacity = faded
                    statusText(c, status.text, color: statusColor, weight: statusWeight, x: p.x + 22, baseline: p.y + 13, anchor: .start)
                }
            } else {
                put(ctx, text(L.name(station), size: 13, color: nameColor, weight: weight), x: p.x, baseline: p.y + 29, size: 13, anchor: .middle)
                if let status {
                    var c = ctx
                    c.opacity = faded
                    statusText(c, status.text, color: statusColor, weight: statusWeight, x: p.x, baseline: p.y + 43, anchor: .middle)
                }
            }
        }
        // Inputs and the output.
        for (family, y) in L.agents {
            let red = state.redAgents.contains(family) && !state.allUnconfirmed
            put(ctx, text(family.label, size: 11, color: red ? ND.redText : ND.t2, weight: red ? .semibold : .regular),
                x: 64, baseline: y + 4, size: 11, anchor: .end)
        }
        put(ctx, text(Say.savedByAgentsInput, size: 11, color: ND.t2), x: 128, baseline: L.yBottom + 4, size: 11, anchor: .end)
        put(ctx, text(Say.answersToAgents, size: 11, color: ND.t2), x: L.size.width, baseline: L.yBottom - 8, size: 11, anchor: .end)
        put(ctx, text(Say.goesIn, size: 11, color: ND.t3, weight: .medium), x: 0, baseline: 10, size: 11, anchor: .start)
        if state.allUnconfirmed {
            let caption = state.lastKnownAt.map(Say.valuesLastKnownAt) ?? Say.everyPipeNotConfirmed
            put(ctx, text(caption, size: 11, color: ND.t2, weight: .medium), x: L.size.width, baseline: 10, size: 11, anchor: .end)
        }
        for (m, sev) in marks {
            ctx.fill(Path(ellipseIn: CGRect(x: m.x - 8.5, y: m.y - 8.5, width: 17, height: 17)), with: .color(ND.card))
            symbol(ctx, sev == .red ? "exclamationmark.circle.fill" : "exclamationmark.triangle.fill", at: m, size: 14,
                   color: sev == .red ? ND.red : ND.orange)
        }
        if !state.allUnconfirmed {
            for p in state.phrases {
                put(ctx, text(p.text, size: 11, color: p.severity == .red ? ND.redText : ND.orangeText, weight: .semibold),
                    x: p.x, baseline: p.y, size: 11, anchor: p.anchor)
            }
        }
    }

    /// A status line; a long one (the Jobs failure) wraps to a second 14 pt line at 200 pt (change 17d).
    static func statusText(_ ctx: GraphicsContext, _ s: String, color: Color, weight: Font.Weight, x: CGFloat, baseline: CGFloat,
                           anchor: NDPipelineState.Phrase.Anchor) {
        let font = NSFont.systemFont(ofSize: 11, weight: weight == .semibold ? .semibold : .regular)
        let width = NSAttributedString(string: s, attributes: [.font: font]).size().width
        guard width > 200, anchor == .start else {
            put(ctx, text(s, size: 11, color: color, weight: weight, mono: true), x: x, baseline: baseline, size: 11, anchor: anchor)
            return
        }
        var first = "", rest = ""
        for word in s.split(separator: " ") {
            let candidate = first.isEmpty ? String(word) : first + " " + word
            if rest.isEmpty, NSAttributedString(string: candidate, attributes: [.font: font]).size().width <= 200 {
                first = candidate
            } else {
                rest = rest.isEmpty ? String(word) : rest + " " + word
            }
        }
        put(ctx, text(first, size: 11, color: color, weight: weight, mono: true), x: x, baseline: baseline, size: 11, anchor: anchor)
        put(ctx, text(rest, size: 11, color: color, weight: weight, mono: true), x: x, baseline: baseline + 14, size: 11, anchor: anchor)
    }
}

/// A station's hit area (R1.2 change 25): pointer over draws one 1 pt `t2` ring 3 pt outside the station, or 3 pt outside
/// the red ring on a red station (which keeps its red ring); the station never changes size; the fill becomes `raised`; keyboard focus is the system's ring. A click opens the details in
/// a popover below the station (D4).
struct NDStationButton: View {
    let station: NDStation
    var interactive = true
    @Binding var hovered: NDStation?
    /// A red station already has its ring: hover adds none (D7.6), only the raised fill.
    var fault = false
    var hubSize: CGSize? = nil
    var popover: () -> AnyView?
    @State private var hovering = false
    @State private var open = false

    /// D4 (real screen, round 1): every popover opens below its station, so none covers the Activity chart or the upper
    /// route; below the bottom lane there is only the footer and the window edge. Only the hub's covers a neighbour (Jobs,
    /// directly under it) while it is open: every edge of the hub has a pipeline neighbour, and below costs least.
    static func arrowEdge(_ station: NDStation) -> Edge { .bottom }

    var body: some View {
        let size = hubSize ?? CGSize(width: 40, height: 40)
        Button { if interactive { open = true } } label: {
            ZStack {
                if hovering || open {
                    if let hubSize {
                        RoundedRectangle(cornerRadius: 15, style: .continuous).stroke(ND.t2, lineWidth: 1.5)
                            .frame(width: hubSize.width + 7, height: hubSize.height + 7)
                    } else {
                        Circle().stroke(ND.t2, lineWidth: 1).frame(width: fault ? 41 : 33, height: fault ? 41 : 33)
                    }
                }
                Color.clear.frame(width: size.width, height: size.height).contentShape(Rectangle())
            }
        }
        .buttonStyle(.plain)
        .onHover { inside in
            hovering = interactive && inside
            if interactive { hovered = inside ? station : (hovered == station ? nil : hovered) }
        }
        .popover(isPresented: $open, arrowEdge: Self.arrowEdge(station)) {
            popover() ?? AnyView(EmptyView())
        }
        .accessibilityLabel(NDPipelineLayout.name(station))
        .accessibilityAddTraits(.isButton)
    }
}

// MARK: - Station details (the popover, § 3.2 "Clicking a stage")

/// The old panel's detail rows, behind a click on a stage: a title, 24 pt rows of label and value (red when faulty), and
/// one bordered button to the page that owns it. 276 pt wide, 12 pt padding.
struct NDStationDetails: View {
    @ObservedObject var store: NDStore
    let station: NDStation

    struct Line: Identifiable {
        var id: String { label }
        var label: String
        var value: String
        var tone: NDModel.Tone = .fine
    }

    var body: some View {
        let lines = Self.lines(station, store.model, store.inputs)
        VStack(alignment: .leading, spacing: 0) {
            Text(NDPipelineLayout.name(station)).font(NDType.bodyStrong).foregroundStyle(ND.t1).padding(.bottom, 6)
            ForEach(lines) { line in
                HStack {
                    Text(line.label).font(NDType.body).foregroundStyle(ND.t2)
                    Spacer(minLength: 12)
                    Text(line.value).font(NDType.number).foregroundStyle(line.tone == .fault ? ND.redText : line.tone == .warn ? ND.orangeText : ND.t1)
                }
                .frame(height: NDGrid.row)
            }
            if let page = Self.page(station) {
                HStack {
                    Spacer()
                    NDButton(label: page == .backups ? BrainBarMenuPresenter.SurfaceSay.openBackups : BrainBarMenuPresenter.SurfaceSay.openJobs,
                             small: true) { store.page = page }
                }
                .padding(.top, 10)
            }
        }
        .padding(12)
        .frame(width: 276)
    }

    static func page(_ station: NDStation) -> NDPage? {
        switch station {
        case .mac, .drive: .backups
        case .jobs: .jobs
        default: nil
        }
    }

    static func lines(_ station: NDStation, _ m: NDModel, _ i: NDInputs) -> [Line] {
        typealias Say = BrainBarMenuPresenter.SurfaceSay
        let rows = m.areas.flatMap(\.rows)
        func row(_ id: String, as label: String? = nil) -> Line? {
            guard let r = rows.first(where: { $0.id == id }) else { return nil }
            let value: String
            if case .gauge = r.visual, !r.number.unit.isEmpty {
                value = Say.ago(r.number.text)
            } else {
                value = r.number.text
            }
            return Line(label: label ?? r.name, value: value, tone: r.number.tone)
        }
        switch station {
        case .read:
            var out = [row("ingest")].compactMap { $0 }
            for a in i.agents.agents {
                out.append(Line(label: a.family.label, value: a.sessions.map { Say.running($0) } ?? NDModel.Number.unknown.value))
            }
            return out
        case .queue:
            var out = [row("stores")].compactMap { $0 }
            if let queued = i.oldestSaveQueuedAt {
                out.append(Line(label: Say.stageQueue, value: Say.waiting(NDFormat.ageText(max(0, i.capture.now.timeIntervalSince(queued))))))
            }
            return out
        case .memory:
            return [Line(label: Say.stageMemory, value: m.memories ?? NDModel.Number.unknown.value), row("integrity")].compactMap { $0 }
        case .meaning:
            return [row("searchable"), row("embedding")].compactMap { $0 }
        case .search:
            return [row("search"), Line(label: Say.searchByWordsNoCheck, value: "")].compactMap { $0 }
        case .mac:
            return [row("backup_local", as: Say.newestCopyOnMac), row("disk")].compactMap { $0 }
        case .drive:
            let auth = rows.first { $0.id == "drive_auth" }
            let account = auth.map { r in
                Line(label: Say.account, value: r.number.tone == .fault ? Say.signedOutValue : r.number.value == "—" ? "—" : Say.signedIn,
                     tone: r.number.tone)
            }
            return [row("backup_offsite", as: Say.databaseCopy), row("transcripts", as: Say.conversationsCopy), account].compactMap { $0 }
        case .jobs:
            return [row("nightly", as: Say.jobNames[.maintenanceNightly]), row("weekly", as: Say.jobNames[.maintenanceWeekly])].compactMap { $0 }
        }
    }
}
