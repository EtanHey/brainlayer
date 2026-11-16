import AppKit
import SwiftUI

// MARK: - Layout (R1.3 changes 1 to 6 with the 1.3.1 corrections L1 to L5), as data

enum NDPipelineLayout {
    typealias Say = BrainBarMenuPresenter.SurfaceSay

    /// 820 × 294 inside its surface. The drawing may run 12 pt into the surface's right padding: the Kept-safe frame
    /// keeps 12 pt between its edge and any text (lead, 1.3 build brief).
    static let size = CGSize(width: 820, height: 294)
    static let drawWidth: CGFloat = 832
    static let yTop: CGFloat = 64, ySaves: CGFloat = 134, yAnswers: CGFloat = 248, yRail: CGFloat = 18
    static let stationRadius: CGFloat = 13
    /// "Your agents": the group the lanes leave and the answers return to (L1).
    static let agentsFrame = CGRect(x: 8, y: 38, width: 92, height: 104)
    static let agentRows: [(AgentFamily, CGFloat)] = AgentFamily.allCases.enumerated().map { ($0.element, 60 + 22 * CGFloat($0.offset)) }
    /// Memory: an outline, 100 × 102 (L5).
    static let hub = CGRect(x: 400, y: 44, width: 100, height: 102)
    /// "Kept safe" hugs its two stations (L5), widened by the lead so the two text blocks keep 16 pt between them and
    /// 12 pt to the frame at their widest statuses ("last backup 27 h ago" in Semibold): Copy on this Mac moves to x 630.
    static let keptSafe = CGRect(x: 561, y: 38, width: 269, height: 80)

    static let stations: [NDStation: CGPoint] = [
        .read: CGPoint(x: 230, y: 64), .queue: CGPoint(x: 230, y: 134), .search: CGPoint(x: 580, y: 248),
        .mac: CGPoint(x: 630, y: 64), .drive: CGPoint(x: 760, y: 64), .jobs: CGPoint(x: 450, y: 186),
    ]
    static let symbols: [NDStation: String] = [
        .read: "text.bubble", .queue: "tray.full", .memory: "brain", .mac: "laptopcomputer", .drive: "icloud",
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

    /// One drawn connection: its points, a label set in the line, and its arrowheads (Jobs has one at each end).
    struct Connection: Sendable {
        var pipe: NDPipe
        var line: NDPolyline
        var corner: CGFloat = 8
        var label: String? = nil
        var labelAt: CGFloat = 0.5
        var bothWays = false
    }

    static let connections: [Connection] = {
        let r = stationRadius
        let read = point(.read), queue = point(.queue), search = point(.search), mac = point(.mac), drive = point(.drive)
        let jobs = point(.jobs)
        var out: [Connection] = []
        out.append(Connection(pipe: .sessions, line: NDPolyline(points: [CGPoint(x: agentsFrame.maxX, y: yTop), read]).trimmed(end: r),
                              label: Say.lanesSessions))
        out.append(Connection(pipe: .readMemory, line: NDPolyline(points: [read, CGPoint(x: hub.minX, y: yTop)]).trimmed(start: r)))
        out.append(Connection(pipe: .saves, line: NDPolyline(points: [CGPoint(x: agentsFrame.maxX, y: ySaves), queue]).trimmed(end: r),
                              label: Say.lanesSaves))
        out.append(Connection(pipe: .queueMemory, line: NDPolyline(points: [queue, CGPoint(x: hub.minX, y: ySaves)]).trimmed(start: r)))
        out.append(Connection(pipe: .memorySearch, line: NDPolyline(points: [CGPoint(x: hub.maxX, y: ySaves), CGPoint(x: search.x, y: ySaves),
                                                                              search]).trimmed(end: r), corner: 12))
        // L1: the answers run left along the bottom, then up into the group's bottom edge, the arrowhead touching it.
        out.append(Connection(pipe: .answers, line: NDPolyline(points: [search, CGPoint(x: 54, y: yAnswers),
                                                                         CGPoint(x: 54, y: agentsFrame.maxY)]).trimmed(start: r), corner: 12))
        out.append(Connection(pipe: .memoryMac, line: NDPolyline(points: [CGPoint(x: hub.maxX, y: yTop), mac]).trimmed(end: r)))
        out.append(Connection(pipe: .macDrive, line: NDPolyline(points: [mac, drive]).trimmed(start: r, end: r)))
        // The copy rail leaves the group's top at x 70; its label sits at 78 % so Memory's popover never covers it.
        out.append(Connection(pipe: .rail, line: NDPolyline(points: [CGPoint(x: 70, y: agentsFrame.minY), CGPoint(x: 70, y: yRail),
                                                                     CGPoint(x: drive.x, y: yRail), drive]).trimmed(end: r),
                              corner: 10, label: Say.sessionFilesCopied, labelAt: 0.78))
        // L3: straight and vertical, hub bottom to Jobs, an arrowhead at both ends.
        out.append(Connection(pipe: .jobsMemory, line: NDPolyline(points: [jobs, CGPoint(x: jobs.x, y: hub.maxY)]).trimmed(start: r),
                              bothWays: true))
        return out
    }()

    /// "answers to agents" beside the rising leg (L1), and the agent fault's phrase under the group.
    static let answersLabel = CGPoint(x: 62, y: 204)
    static let agentPhrase = CGPoint(x: 62, y: 158)

    // MARK: actions at rest (change 16, L3)

    /// The action's frame for a button of `width` (20 pt high). Under the status line, or on a free side where a lane
    /// runs below: Jobs on its left (L3), Read and Search on their right. Google Drive's and the Mac's sit just under the
    /// Kept-safe frame, which no longer grows: the lead's wider frame would otherwise grow into the Memory-to-Search lane.
    static func actionFrame(_ s: NDStation, width: CGFloat) -> CGRect {
        let p = point(s), r = stationRadius
        switch s {
        case .drive, .mac: return CGRect(x: p.x - width / 2, y: keptSafe.maxY + 6, width: width, height: 20)
        case .queue: return CGRect(x: p.x - width / 2, y: p.y + 51, width: width, height: 20)
        case .jobs: return CGRect(x: p.x - r - 12 - width, y: p.y - 10, width: width, height: 20)
        case .read: return CGRect(x: p.x + 50, y: p.y + 33, width: width, height: 20)
        case .search: return CGRect(x: p.x + r + 12, y: p.y - 10, width: width, height: 20)
        case .agents: return CGRect(x: 62, y: agentsFrame.maxY + 24, width: width, height: 20)
        case .memory: return CGRect(x: p.x - width / 2, y: hub.maxY + 6, width: width, height: 20)
        }
    }

    // MARK: popovers (L2)

    struct PopoverSpec: Equatable {
        var edge: Edge
        /// The rect the popover's arrow points at, in chart coordinates.
        var anchor: CGRect
        var width: CGFloat
    }

    /// Popover sides per station: never over its own station, a neighbour's name or status, a lane label or an action.
    /// `actions` names the stations showing an action at rest (a Kept-safe popover opens under either copy's button).
    static func popover(_ s: NDStation, actions: Set<NDStation>) -> PopoverSpec {
        let p = point(s)
        let hasAction = actions.contains(s)
        let keptAction = !actions.isDisjoint(with: [.mac, .drive])
        let circle = CGRect(x: p.x - 13, y: p.y - 13, width: 26, height: 26)
        switch s {
        case .agents: return PopoverSpec(edge: .top, anchor: agentsFrame, width: 260)
        case .read: return PopoverSpec(edge: .top, anchor: circle, width: 260)
        case .memory: return PopoverSpec(edge: .top, anchor: CGRect(x: hub.minX, y: hub.minY, width: 32, height: hub.height), width: 208)
        case .mac: return PopoverSpec(edge: .bottom, anchor: CGRect(x: 620, y: p.y - 13, width: 60, height: keptAction ? 97 : 73), width: 220)
        case .drive: return PopoverSpec(edge: .bottom, anchor: CGRect(x: p.x - 13, y: p.y - 13, width: 26, height: keptAction ? 97 : 73), width: 220)
        case .search:
            if hasAction { return PopoverSpec(edge: .bottom, anchor: CGRect(x: p.x - 13, y: p.y - 13, width: 26, height: 62), width: 214) }
            return PopoverSpec(edge: .trailing, anchor: circle, width: 214)
        case .queue: return PopoverSpec(edge: .bottom, anchor: CGRect(x: 217, y: p.y - 13, width: 88, height: hasAction ? 88 : 67), width: 200)
        case .jobs: return PopoverSpec(edge: .bottom, anchor: CGRect(x: 370, y: p.y - 13, width: 60, height: 63), width: 230)
        }
    }

    /// Where the popover lands for a content `height` (NSPopover centres on the anchor; the arrow is 10 pt long).
    static func popoverFrame(_ spec: PopoverSpec, height: CGFloat) -> CGRect {
        let arrow: CGFloat = 10
        switch spec.edge {
        case .top: return CGRect(x: spec.anchor.midX - spec.width / 2, y: spec.anchor.minY - arrow - height, width: spec.width, height: height)
        case .bottom: return CGRect(x: spec.anchor.midX - spec.width / 2, y: spec.anchor.maxY + arrow, width: spec.width, height: height)
        case .trailing: return CGRect(x: spec.anchor.maxX + arrow, y: spec.anchor.midY - height / 2, width: spec.width, height: height)
        case .leading: return CGRect(x: spec.anchor.minX - arrow - spec.width, y: spec.anchor.midY - height / 2, width: spec.width, height: height)
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

    /// The words set in or beside lanes, and the two group captions.
    static func laneLabelRects() -> [(String, CGRect)] {
        var out: [(String, CGRect)] = []
        for c in connections {
            guard let label = c.label else { continue }
            let m = c.line.point(at: c.line.length * c.labelAt)
            let w = textWidth(label, size: 11) + 12
            out.append((label, CGRect(x: m.x - w / 2, y: m.y - 8, width: w, height: 16)))
        }
        out.append((Say.answersToAgentsLane, CGRect(x: answersLabel.x, y: answersLabel.y - 10,
                                                    width: textWidth(Say.answersToAgentsLane, size: 11), height: 13)))
        out.append((Say.keptSafe, CGRect(x: keptSafe.minX + 10, y: keptSafe.minY - 8,
                                         width: textWidth(Say.keptSafe, size: 11, weight: .medium) + 12, height: 16)))
        out.append((Say.yourAgents, CGRect(x: agentsFrame.minX + 8, y: agentsFrame.minY - 8,
                                           width: textWidth(Say.yourAgents, size: 11, weight: .medium) + 12, height: 16)))
        return out
    }
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
