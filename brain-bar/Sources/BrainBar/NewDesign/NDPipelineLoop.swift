import AppKit
import SwiftUI

// MARK: - Layout (R1.3 with 1.3.1; R1.4 changes 1, 17 and 19), as data

enum NDPipelineLayout {
    typealias Say = BrainBarMenuPresenter.SurfaceSay

    /// 820 × 270 inside its surface (R1.4 change 19). The drawing may run 12 pt into the surface's right padding: the
    /// Kept-safe frame keeps 12 pt between its edge and any text (lead, 1.3 build brief).
    /// R2A item 5: the answers lane runs below Jobs' action (route the lines around the button), so 820 × 306.
    static let size = CGSize(width: 820, height: 306)
    static let drawWidth: CGFloat = 832
    static let yTop: CGFloat = 64, ySaves: CGFloat = 134, yAnswers: CGFloat = 272, yRail: CGFloat = 18
    /// The Memory-to-Search lane's level: 8 pt below the grown Kept-safe frame.
    static let ySearchLane: CGFloat = 162
    static let stationRadius: CGFloat = 13
    /// The fault ring's radius (R1.4 change 16).
    static let ringRadius: CGFloat = 15.5
    /// R1.4 change 17: every line ends 3 pt outside the outermost ring a station can have, in every state, arrow tips
    /// included; at the Memory box and the agents' frame, 3 pt from the edge.
    static let lineEnd: CGFloat = 18.5
    static let edgeGap: CGFloat = 3
    /// The agents: four provider icons in one column (R1.4 change 1).
    static let agentsFrame = CGRect(x: 16, y: 38, width: 44, height: 108)
    static let agentRows: [(AgentFamily, CGFloat)] = AgentFamily.allCases.enumerated().map { ($0.element, 56 + 24 * CGFloat($0.offset)) }
    static let iconSize: CGFloat = 16
    static func iconCentre(_ family: AgentFamily) -> CGPoint {
        CGPoint(x: agentsFrame.midX, y: agentRows.first { $0.0 == family }?.1 ?? agentsFrame.midY)
    }
    /// Memory: an outline, 100 × 102 (L5).
    static let hub = CGRect(x: 400, y: 44, width: 100, height: 102)
    /// "Kept safe" hugs its two stations (L5), widened by the lead so the two text blocks keep 16 pt between them and
    /// 12 pt to the frame at their widest statuses ("last backup 27 h ago" in Semibold): Copy on this Mac moves to x 630.
    static let keptSafe = CGRect(x: 561, y: 38, width: 269, height: 80)

    static let stations: [NDStation: CGPoint] = [
        .read: CGPoint(x: 230, y: 64), .queue: CGPoint(x: 230, y: 134), .search: CGPoint(x: 580, y: 272),
        .mac: CGPoint(x: 630, y: 64), .drive: CGPoint(x: 760, y: 64), .jobs: CGPoint(x: 450, y: 186),
    ]
    static let symbols: [NDStation: String] = [
        .read: "text.bubble", .queue: "tray.full", .memory: "brain", .mac: "laptopcomputer.and.arrow.down", .drive: "icloud",
        .search: "magnifyingglass", .jobs: "clock", .agents: "person.2",
    ]

    static func name(_ station: NDStation) -> String {
        switch station {
        case .agents: return Say.yourAgents
        case .read: return Say.stageRead
        case .queue: return Say.stageQueue
        case .memory: return Say.stageMemory
        case .mac: return Say.stageMac
        case .drive: return Say.stageDrive
        case .search: return Say.search
        case .jobs: return Say.jobs
        }
    }

    static func point(_ s: NDStation) -> CGPoint {
        switch s {
        case .agents: CGPoint(x: agentsFrame.midX, y: agentsFrame.midY)
        case .memory: CGPoint(x: hub.midX, y: hub.midY)
        default: stations[s] ?? .zero
        }
    }

    /// What a pointer or a click lands on: the circle, the hub, or the agents' group.
    static func hitRect(_ s: NDStation) -> CGRect {
        switch s {
        case .agents: agentsFrame
        case .memory: hub
        default: CGRect(x: point(s).x - 20, y: point(s).y - 20, width: 40, height: 40)
        }
    }

    /// One drawn connection: its points and its arrowheads (Jobs has one at each end). R1.4 change 5 to 8: no words.
    struct Connection: Sendable {
        var pipe: NDPipe
        var line: NDPolyline
        var corner: CGFloat = 10
        var bothWays = false
    }

    static let connections: [Connection] = {
        let e = lineEnd, g = edgeGap
        let read = point(.read), queue = point(.queue), search = point(.search), mac = point(.mac), drive = point(.drive)
        let jobs = point(.jobs), ax = agentsFrame.midX
        var out: [Connection] = []
        out.append(Connection(pipe: .sessions, line: NDPolyline(points: [CGPoint(x: agentsFrame.maxX + g, y: yTop), read]).trimmed(end: e)))
        out.append(Connection(pipe: .readMemory, line: NDPolyline(points: [read, CGPoint(x: hub.minX - g, y: yTop)]).trimmed(start: e)))
        out.append(Connection(pipe: .saves, line: NDPolyline(points: [CGPoint(x: agentsFrame.maxX + g, y: ySaves), queue]).trimmed(end: e)))
        out.append(Connection(pipe: .queueMemory, line: NDPolyline(points: [queue, CGPoint(x: hub.minX - g, y: ySaves)]).trimmed(start: e)))
        // R2A item 5: the Kept-safe frame grows under its actions, so the Memory-to-Search lane leaves the box's bottom
        // edge and runs below the grown frame (route the lines around the button, never the button around the lines).
        out.append(Connection(pipe: .memorySearch, line: NDPolyline(points: [CGPoint(x: hub.maxX - 14, y: hub.maxY + g),
                                                                              CGPoint(x: hub.maxX - 14, y: ySearchLane),
                                                                              CGPoint(x: search.x, y: ySearchLane), search]).trimmed(end: e),
                              corner: 8))
        out.append(Connection(pipe: .answers, line: NDPolyline(points: [search, CGPoint(x: ax, y: yAnswers),
                                                                         CGPoint(x: ax, y: agentsFrame.maxY + g)]).trimmed(start: e), corner: 12))
        out.append(Connection(pipe: .memoryMac, line: NDPolyline(points: [CGPoint(x: hub.maxX + g, y: yTop), mac]).trimmed(end: e)))
        out.append(Connection(pipe: .macDrive, line: NDPolyline(points: [mac, drive]).trimmed(start: e, end: e)))
        out.append(Connection(pipe: .rail, line: NDPolyline(points: [CGPoint(x: ax, y: agentsFrame.minY - g), CGPoint(x: ax, y: yRail),
                                                                     CGPoint(x: drive.x, y: yRail), drive]).trimmed(end: e)))
        out.append(Connection(pipe: .jobsMemory, line: NDPolyline(points: [jobs, CGPoint(x: jobs.x, y: hub.maxY + g)]).trimmed(start: e),
                              bothWays: true))
        return out
    }()

    /// What a line ends on: a station's centre (with the line-end radius) or a frame (with the 3 pt gap).
    enum End: Equatable { case station(NDStation), frame(CGRect) }
    static func ends(_ pipe: NDPipe) -> (start: End, end: End) {
        switch pipe {
        case .sessions: (.frame(agentsFrame), .station(.read))
        case .readMemory: (.station(.read), .frame(hub))
        case .saves: (.frame(agentsFrame), .station(.queue))
        case .queueMemory: (.station(.queue), .frame(hub))
        case .memorySearch: (.frame(hub), .station(.search))  // leaves the box's bottom edge
        case .answers: (.station(.search), .frame(agentsFrame))
        case .memoryMac: (.frame(hub), .station(.mac))
        case .macDrive: (.station(.mac), .station(.drive))
        case .rail: (.frame(agentsFrame), .station(.drive))
        case .jobsMemory: (.station(.jobs), .frame(hub))
        }
    }

    /// The hover strips along the rail and the return line (R1.4 changes 5 and 8): each segment, 12 pt thick.
    static func hoverStrips(_ pipe: NDPipe) -> [CGRect] {
        guard let line = connections.first(where: { $0.pipe == pipe })?.line, line.points.count > 1 else { return [] }
        return (1..<line.points.count).map { i in
            let a = line.points[i - 1], b = line.points[i]
            return CGRect(x: min(a.x, b.x) - 6, y: min(a.y, b.y) - 6, width: abs(b.x - a.x) + 12, height: abs(b.y - a.y) + 12)
        }
    }

    // MARK: actions at rest (change 16, L3)

    /// The action's frame for a button of `width` (20 pt high). Under the status line, or on a free side where a lane
    /// runs below: Jobs on its left (L3), Read and Search on their right. Google Drive's and the Mac's sit just under the
    /// Kept-safe frame, which no longer grows: the lead's wider frame would otherwise grow into the Memory-to-Search lane.
    static func actionFrame(_ s: NDStation, width: CGFloat, hasStatus: Bool = true) -> CGRect {
        let p = point(s)
        switch s {
        // Exceptions, each forced: Read's would land on Save queue's circle 70 pt below, so it sits right of the status;
        // the agents have no status line and the answers rise into their frame's centre, so theirs sits right of that line.
        case .read: return CGRect(x: p.x + 50, y: p.y + 33, width: width, height: 20)
        case .agents: return CGRect(x: agentsFrame.midX + 8, y: agentsFrame.maxY + 10, width: width, height: 20)
        case .memory: return CGRect(x: p.x - width / 2, y: hub.maxY + 6, width: width, height: 20)
        // R2A item 5: centred under the station's status line (or its name, when it has none), 6 pt below, 20 pt high.
        default: return CGRect(x: p.x - width / 2, y: p.y + (hasStatus ? 46 : 33) + 6, width: width, height: 20)
        }
    }

    /// The Kept-safe frame: it hugs its two stations, and grows to hold an action (and the waiting line that replaces
    /// Reconnect) under either copy, so its edge never runs through a button.
    static func keptFrame(grown: Bool) -> CGRect {
        guard grown else { return keptSafe }
        let bottom = point(.drive).y + 46 + 6 + 30 + 8
        return CGRect(x: keptSafe.minX, y: keptSafe.minY, width: keptSafe.width, height: bottom - keptSafe.minY)
    }

    /// Council r2a finding 4: the waiting line's own area, right-aligned to Reconnect's right edge under Google Drive,
    /// in the place the button leaves (8 pt under the Kept-safe frame), clear of every connector.
    /// It wraps at `waitingWidth` so it never reaches the Memory-to-Search lane that turns down at Search's x.
    static let waitingWidth: CGFloat = 200
    static func waitingArea(buttonWidth: CGFloat, lineWidth: CGFloat) -> CGRect {
        let button = actionFrame(.drive, width: buttonWidth, hasStatus: true)
        let width = min(lineWidth, waitingWidth)
        let lines = ceil(lineWidth / waitingWidth)
        return CGRect(x: button.maxX - width, y: button.minY, width: width, height: max(button.height, lines * 14 + 2))
    }

    // MARK: hover panels (L2, and the lead's 1.4 rule: every arrow on its own station)

    /// A station's hover panel: the edge it opens on, the rect its arrow points at, the arrow's tip, the body, and the
    /// whole window (body and arrow). The arrow always sits on the station's own circle (its frame for the agents, its box
    /// for Memory); the body slides along the edge to keep clear of every neighbour's words and actions.
    struct Panel: Equatable {
        var edge: Edge
        var anchor: CGRect
        var tip: CGPoint
        var body: CGRect
        var outer: CGRect
    }

    static let arrowLength: CGFloat = 10
    /// R2A item 1: a panel keeps 12 pt inside the window's content edge. In chart coordinates the window runs 40 pt past
    /// each side of the chart (page side 24, surface padding 16) and 60 pt below it (surface 12, page 20, footer 28).
    static let windowInside = (minX: CGFloat(-28), maxX: CGFloat(848), maxY: CGFloat(306 + 60 - 12))
    static let arrowHalfWidth: CGFloat = 9
    /// The arrow keeps at least this far from the body's ends (its corner radius plus the arrow's half width).
    static let arrowInset: CGFloat = 18

    /// The station's own shape: its circle, the agents' frame, or the Memory box.
    static func circle(_ s: NDStation) -> CGRect {
        switch s {
        case .agents: agentsFrame
        case .memory: hub
        default: CGRect(x: point(s).x - stationRadius, y: point(s).y - stationRadius, width: 2 * stationRadius, height: 2 * stationRadius)
        }
    }

    /// Where each station's panel opens, its width, and how far the body slides along the edge from centred.
    static func panelSide(_ s: NDStation, hasAction: Bool) -> (edge: Edge, width: CGFloat, slide: CGFloat) {
        switch s {
        case .agents: (.top, 240, 100)
        case .read: (.top, 260, 0)
        case .memory: (.top, 208, 0)
        case .mac: (.top, 220, 0)
        case .drive: (.top, 220, -34)
        // Search opens to its right, beside its own words, inside the window (R2A item 1).
        case .search: (.trailing, 214, 0)
        // Between an agent's action (left) and Jobs' action (right).
        case .queue: (.bottom, 180, 0)
        case .jobs: (.bottom, 230, -48)
        }
    }

    static func panel(_ s: NDStation, state: NDPipelineState, actions: Set<NDStation>, actionWidths: [NDStation: CGFloat] = [:],
                      height: CGFloat) -> Panel {
        let side = panelSide(s, hasAction: actions.contains(s))
        let own = circle(s)
        var anchor = own
        if side.edge == .trailing || side.edge == .leading {
            // Beside the station's words as well as its circle, so the body never covers them.
            for r in labelRects(s, state) { anchor = anchor.union(CGRect(x: r.minX, y: own.minY, width: r.width, height: own.height)) }
        }
        if side.edge == .bottom {
            // Below the station's own words (and its action, when it sits under them), never over them.
            for r in labelRects(s, state) { anchor = anchor.union(r) }
            if actions.contains(s), let w = actionWidths[s] {
                let f = actionFrame(s, width: w, hasStatus: state.status[s] != nil)
                if f.minY >= own.maxY { anchor = anchor.union(f) }
            }
        }
        let a = arrowLength
        var w = side.width
        if side.edge == .trailing { w = min(w, windowInside.maxX - (anchor.maxX + a)) }
        let tip: CGPoint
        var body: CGRect
        switch side.edge {
        case .top:
            tip = CGPoint(x: own.midX, y: anchor.minY)
            body = CGRect(x: tip.x - w / 2 + side.slide, y: tip.y - a - height, width: w, height: height)
        case .bottom:
            tip = CGPoint(x: own.midX, y: anchor.maxY)
            body = CGRect(x: tip.x - w / 2 + side.slide, y: tip.y + a, width: w, height: height)
        case .trailing:
            tip = CGPoint(x: anchor.maxX, y: own.midY)
            body = CGRect(x: tip.x + a, y: tip.y - height / 2 + side.slide, width: w, height: height)
        case .leading:
            tip = CGPoint(x: anchor.minX, y: own.midY)
            body = CGRect(x: tip.x - a - w, y: tip.y - height / 2 + side.slide, width: w, height: height)
        }
        // Inside the window (R2A item 1), then the arrow never leaves the body's straight edge.
        switch side.edge {
        case .top, .bottom:
            body.origin.x = min(max(body.minX, windowInside.minX), windowInside.maxX - w)
            body.origin.x = min(max(body.minX, tip.x - w + arrowInset), tip.x - arrowInset)
        case .leading, .trailing:
            body.origin.y = min(body.minY, windowInside.maxY - height)
            body.origin.y = min(max(body.minY, tip.y - height + arrowInset), tip.y - arrowInset)
        }
        let arrowBox: CGRect = switch side.edge {
        case .top: CGRect(x: tip.x - arrowHalfWidth, y: tip.y - a, width: 2 * arrowHalfWidth, height: a)
        case .bottom: CGRect(x: tip.x - arrowHalfWidth, y: tip.y, width: 2 * arrowHalfWidth, height: a)
        case .trailing: CGRect(x: tip.x, y: tip.y - arrowHalfWidth, width: a, height: 2 * arrowHalfWidth)
        case .leading: CGRect(x: tip.x - a, y: tip.y - arrowHalfWidth, width: a, height: 2 * arrowHalfWidth)
        }
        return Panel(edge: side.edge, anchor: anchor, tip: tip, body: body, outer: body.union(arrowBox))
    }

    /// What a pointer lands on (the lead's 1.4 rule): the station's ring and its name, in the drawing's own coordinates.
    static func hitShape(_ s: NDStation, _ state: NDPipelineState) -> (circle: CGRect, label: CGRect?) {
        switch s {
        case .agents, .memory: return (circle(s), nil)
        default:
            let p = point(s), r = ringRadius
            return (CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r), labelRects(s, state).first)
        }
    }

    // MARK: text boxes (the overlap rules and their tests)

    static func textWidth(_ s: String, size: CGFloat, weight: NSFont.Weight = .regular) -> CGFloat {
        ceil(NSAttributedString(string: s, attributes: [.font: NSFont.monospacedDigitSystemFont(ofSize: size, weight: weight)]).size().width)
    }

    /// A station's name and status boxes as drawn for `state`.
    static func labelRects(_ s: NDStation, _ state: NDPipelineState) -> [CGRect] {
        guard stations[s] != nil else { return [] }
        let p = point(s)
        let t = state.treatment(s)
        let nameW = textWidth(name(s), size: 13, weight: t == .fault ? .semibold : .regular)
        var out = [CGRect(x: p.x - nameW / 2, y: p.y + 17, width: nameW, height: 16)]
        if let status = state.status[s] {
            let w = textWidth(status.text, size: 11, weight: t == .fault || t == .notice ? .semibold : .regular)
            out.append(CGRect(x: p.x - w / 2, y: p.y + 33, width: w, height: 13))
        }
        return out
    }

    /// R1.4 change 5 to 10: no words are set in or beside lanes, and the two frames have no caption.
    static func laneLabelRects() -> [(String, CGRect)] { [] }
}

// MARK: - Strokes as tokens (change 9: one meaning per line style)

struct NDPipeStroke: Equatable {
    var color: Color
    var width: CGFloat
    var dash: [CGFloat]
    var opacity: Double = 1

    static let flowing = NDPipeStroke(color: ND.pipe, width: 1.5, dash: [])
    /// Not confirmed: the only meaning of a dash.
    static let unconfirmed = NDPipeStroke(color: ND.pipe, width: 1.5, dash: [3, 4])
    /// Downstream of a break: solid at 28 %.
    static let downstream = NDPipeStroke(color: ND.pipe, width: 1.5, dash: [], opacity: 0.28)
    static func severity(_ s: NDSeverity) -> NDPipeStroke {
        NDPipeStroke(color: s == .red ? ND.red : ND.orange, width: 2.5, dash: [])
    }

    var shading: GraphicsContext.Shading { .color(color.opacity(opacity)) }
    var style: StrokeStyle { StrokeStyle(lineWidth: width, lineCap: .round, lineJoin: .round, dash: dash) }
}
