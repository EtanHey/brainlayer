import AppKit
import SwiftUI

// MARK: - Layout (R1.3 with 1.3.1; R1.4 changes 1, 17 and 19), as data

enum NDPipelineLayout {
    typealias Say = BrainBarMenuPresenter.SurfaceSay

    /// 820 × 270 inside its surface (R1.4 change 19). The drawing may run 12 pt into the surface's right padding: the
    /// Kept-safe frame keeps 12 pt between its edge and any text (lead, 1.3 build brief).
    /// R1.5 change 23: 820 × 282; the answers lane at y 248 clears Jobs' longer name and its status.
    /// R1.12 change 6: 820 × 296; Search sits at the loop's bottom-right corner with its status line under it.
    static let size = CGSize(width: 820, height: 296)
    static let drawWidth: CGFloat = 832
    static let yTop: CGFloat = 64, ySaves: CGFloat = 134, yAnswers: CGFloat = 248, yRail: CGFloat = 18
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
    /// R2 item 8: the diagram sits centred in its card, 32 pt from each side: everything right of the agents moved 26 pt
    /// left (the long first-row lines are shorter).
    /// R1.10 (L49): 140 wide, so the two job stations hang under it with straight links. R1.10.4: everything right of the
    /// agents moved 15 pt left (the first-row lines are 15 pt shorter), so the diagram is 32 pt from each side again.
    static let hub = CGRect(x: 365, y: 44, width: 140, height: 102)
    /// "Kept safe" hugs its two stations (L5), widened by the lead so the two text blocks keep 16 pt between them and
    /// 12 pt to the frame at their widest statuses ("last backup 27 h ago" in Semibold): Copy on this Mac moves to x 630.
    /// R1.5: Copy on this Mac's status grew ("3 kept, newest 30 h ago"), so it moves to x 622 and the frame starts at x 544.
    /// R1.10: Memory grew 40 pt, so the frame starts 16 pt right of it; its two stations sit 137 pt apart, the least that
    /// keeps their widest statuses ("2 kept, newest 30 h ago", "last upload 30 h ago") 16 pt apart (the lead's 1.3 rule;
    /// the comp's 110 let them overlap), and the frame keeps 12 pt to them.
    /// R1.12 change 6: the frame is 84 tall. The copies stay 137 pt apart (1.10's, already more than the 110 at which their
    /// statuses ran into each other) and the frame 283 wide: the designer's 140 apart in a 262 frame, at 1.10.4's 32 pt
    /// insets, would leave the widest statuses ("3 kept, newest 30 h ago", "last upload 27 h ago", Semibold) 2.5 and
    /// −1.5 pt from the frame, against the lead's 12 pt rule (testKeptSafeTextKeepsItsSpacing).
    static let keptSafe = CGRect(x: 521, y: 38, width: 283, height: 84)

    /// R1.12 change 6: Search moves under Google Drive, to the loop's bottom-right corner; Memory → Search runs right
    /// along the saves lane and down, and the answers return straight along y 248.
    static let stations: [NDStation: CGPoint] = [
        .read: CGPoint(x: 215, y: 64), .queue: CGPoint(x: 215, y: 134), .search: CGPoint(x: 736, y: 248),
        .mac: CGPoint(x: 599, y: 64), .drive: CGPoint(x: 736, y: 64), .jobs: CGPoint(x: 377, y: 186), .weekly: CGPoint(x: 493, y: 186),
    ]
    static let symbols: [NDStation: String] = [
        .read: "text.bubble", .queue: "tray.full", .memory: "brain", .mac: "laptopcomputer.and.arrow.down", .drive: "icloud",
        .search: "magnifyingglass", .jobs: "moon", .weekly: "broom", .agents: "person.2",
    ]

    static func name(_ station: NDStation) -> String {
        switch station {
        case .agents: return Say.yourAgents
        case .read: return Say.stageRead
        case .queue: return Say.stageQueue
        case .memory: return Say.stageMemory
        case .mac: return Say.localCopy  // R1.10 (L50)
        case .drive: return Say.stageDrive
        case .search: return Say.search
        case .jobs: return Say.stationNightly  // R1.10 (L49)
        case .weekly: return Say.stationWeekly
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
        let jobs = point(.jobs), weekly = point(.weekly), ax = agentsFrame.midX
        var out: [Connection] = []
        out.append(Connection(pipe: .sessions, line: NDPolyline(points: [CGPoint(x: agentsFrame.maxX + g, y: yTop), read]).trimmed(end: e)))
        out.append(Connection(pipe: .readMemory, line: NDPolyline(points: [read, CGPoint(x: hub.minX - g, y: yTop)]).trimmed(start: e)))
        out.append(Connection(pipe: .saves, line: NDPolyline(points: [CGPoint(x: agentsFrame.maxX + g, y: ySaves), queue]).trimmed(end: e)))
        out.append(Connection(pipe: .queueMemory, line: NDPolyline(points: [queue, CGPoint(x: hub.minX - g, y: ySaves)]).trimmed(start: e)))
        // R2A item 5: the Kept-safe frame grows under its actions, so the Memory-to-Search lane leaves the box's bottom
        // edge and runs below the grown frame (route the lines around the button, never the button around the lines).
        // R1.5: the Kept-safe frame no longer grows, so the line leaves the box's right edge at the saves lane and turns
        // once, down into Search's top, at the lines' common 10 pt radius.
        out.append(Connection(pipe: .memorySearch, line: NDPolyline(points: [CGPoint(x: hub.maxX + g, y: ySaves),
                                                                              CGPoint(x: search.x, y: ySaves), search]).trimmed(end: e),
                              corner: 10))
        out.append(Connection(pipe: .answers, line: NDPolyline(points: [search, CGPoint(x: ax, y: yAnswers),
                                                                         CGPoint(x: ax, y: agentsFrame.maxY + g)]).trimmed(start: e), corner: 12))
        out.append(Connection(pipe: .memoryMac, line: NDPolyline(points: [CGPoint(x: hub.maxX + g, y: yTop), mac]).trimmed(end: e)))
        out.append(Connection(pipe: .macDrive, line: NDPolyline(points: [mac, drive]).trimmed(start: e, end: e)))
        out.append(Connection(pipe: .rail, line: NDPolyline(points: [CGPoint(x: ax, y: agentsFrame.minY - g), CGPoint(x: ax, y: yRail),
                                                                     CGPoint(x: drive.x, y: yRail), drive]).trimmed(end: e)))
        out.append(Connection(pipe: .jobsMemory, line: NDPolyline(points: [jobs, CGPoint(x: jobs.x, y: hub.maxY + g)]).trimmed(start: e),
                              bothWays: true))
        out.append(Connection(pipe: .weeklyMemory, line: NDPolyline(points: [weekly, CGPoint(x: weekly.x, y: hub.maxY + g)]).trimmed(start: e),
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
        case .memorySearch: (.frame(hub), .station(.search))
        case .answers: (.station(.search), .frame(agentsFrame))
        case .memoryMac: (.frame(hub), .station(.mac))
        case .macDrive: (.station(.mac), .station(.drive))
        case .rail: (.frame(agentsFrame), .station(.drive))
        case .jobsMemory: (.station(.jobs), .frame(hub))
        case .weeklyMemory: (.station(.weekly), .frame(hub))
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
    static func keptFrame(grown: Bool) -> CGRect { keptSafe }  // R1.5 change 23: it no longer grows (no actions on the chart)

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

    /// L39: a panel may pass the window's bottom edge but stays inside the screen's visible frame (`limit`, in chart
    /// coordinates). The body slides along its edge and, if the screen is too short, gives up height (its content
    /// scrolls); the arrow stays on the station.
    static func fitted(_ panel: Panel, within limit: CGRect) -> Panel {
        var p = panel
        let w = p.body.width
        switch p.edge {
        case .top, .bottom:
            p.body.origin.x = min(max(p.body.minX, limit.minX), limit.maxX - w)
            p.body.origin.x = min(max(p.body.minX, p.tip.x - w + arrowInset), p.tip.x - arrowInset)
            // At least one scrolling line of body (32 pt), however short the screen.
            if p.edge == .bottom, p.body.maxY > limit.maxY {
                p.body.size.height = max(32, limit.maxY - p.body.minY)
                p.scrolls = true
            }
            if p.edge == .top, p.body.minY < limit.minY {
                let bottom = p.body.maxY
                p.body.origin.y = limit.minY
                p.body.size.height = max(32, bottom - limit.minY)
                p.scrolls = true
            }
        case .leading, .trailing:
            p.body.origin.y = min(max(p.body.minY, limit.minY), limit.maxY - p.body.height)
            p.body.origin.y = min(max(p.body.minY, p.tip.y - p.body.height + arrowInset), p.tip.y - arrowInset)
        }
        let arrowBox: CGRect = switch p.edge {
        case .top: CGRect(x: p.tip.x - arrowHalfWidth, y: p.tip.y - arrowLength, width: 2 * arrowHalfWidth, height: arrowLength)
        case .bottom: CGRect(x: p.tip.x - arrowHalfWidth, y: p.tip.y, width: 2 * arrowHalfWidth, height: arrowLength)
        case .trailing: CGRect(x: p.tip.x, y: p.tip.y - arrowHalfWidth, width: arrowLength, height: 2 * arrowHalfWidth)
        case .leading: CGRect(x: p.tip.x - arrowLength, y: p.tip.y - arrowHalfWidth, width: arrowLength, height: 2 * arrowHalfWidth)
        }
        p.outer = p.body.union(arrowBox)
        return p
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
        /// R2 item 2: the body scrolls only when the screen made it shorter than its content.
        var scrolls = false
    }

    static let arrowLength: CGFloat = 10
    /// 1.5.1: what lies outside the chart's surface that no panel may cover: the "Pipeline" block heading above it and the
    /// footer's two lines below it, in chart coordinates (surface padding 16 / 12, heading row 22 + 8, page padding 20,
    /// footer 28, page sides 24).
    static var footerTop: CGFloat { size.height + 12 + 20 }
    /// R1.12 (§ B): the Activity card's bottom edge in chart coordinates (surface padding 16, heading row 22 + 8, the
    /// 32 pt gap): no panel reaches into the card, its bars or its axis labels.
    static var activityBottom: CGFloat { -16 - 8 - 22 - NDGrid.activityToPipeline }
    /// The Activity card's axis labels end its 16 pt bottom padding above its edge: a panel may cover that empty padding
    /// (the comps do), never the bars or the labels.
    static var activityLabelsBottom: CGFloat { activityBottom - 16 }
    static var pipelineHeading: CGRect {
        let w = textWidth(Say.pipeline, size: 13, weight: .semibold)
        return CGRect(x: -16 + 2, y: -16 - 8 - 22, width: w, height: 22)
    }
    static var footerTexts: [CGRect] {
        let left = [Say.statusChecked("59 min"), Say.waitingForNextCheck].map { textWidth($0, size: 11) }.max() ?? 0
        let right = textWidth(Say.nextStatusCheck("59 min"), size: 11)
        let y = footerTop + 7
        return [CGRect(x: -16, y: y, width: left, height: 14), CGRect(x: size.width + 16 - right, y: y, width: right, height: 14)]
    }
    /// R2A item 1: a panel keeps 12 pt inside the window's content edge. In chart coordinates the window runs 40 pt past
    /// each side of the chart (page side 24, surface padding 16) and 60 pt below it (surface 12, page 20, footer 28).
    static let windowInside = (minX: CGFloat(-28), maxX: CGFloat(848), maxY: size.height + 60 - 12)
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
    /// R1.12 (CHANGES-R1.12 § B, 9 of 9): every panel inside the window, covering no station's icon, name or status, no
    /// Activity bar or axis label and no footer text. The five top-lane panels open above into the 136 pt the wider
    /// Activity-to-Pipeline gap gives them; the three inside the loop open into it; Search's opens left under the jobs.
    /// `slide` moves the body along its edge from centred on the arrow (the designer's x, relative to the station).
    static func panelSide(_ s: NDStation, hasAction: Bool) -> (edge: Edge, width: CGFloat, slide: CGFloat) {
        switch s {
        case .agents: (.top, 300, 0)  // 5a: right of the "Pipeline" heading (agentsPanelMinX), notch down at the agents
        case .read: (.top, 300, 10)  // 5b: 140 left of the station, 160 right
        case .memory: (.top, 300, 0)  // 5c
        case .mac: (.top, 320, 0)  // 5d
        case .drive: (.top, 320, -108)  // 5e: ends 52 right of the station, inside the window
        case .queue: (.bottom, 262, -35)  // 5f: below-left inside the loop, its right edge clear of the nightly check-up's name
        case .jobs: (.leading, 262, 0)  // 5g: left, inside the loop, under Save queue's status
        case .weekly: (.trailing, 264, 0)  // 5h: up-right, between the Kept-safe frame and Search
        // 5i, as built: the designer's place (left, under the job stations' words, 238 to the footer) holds 86 pt and the
        // panel needs about 98, and its corner meets the footer's right line. One panel shows at a time, so Search's opens
        // above it instead, in the weekly clean's room: right of the job stations' words, under the Kept-safe frame.
        case .search: (.top, 264, searchPanelMinX + 132 - point(.search).x)
        }
    }

    /// R1.13.1: the agents' panel's left edge, 2 pt right of the "Pipeline" heading's words, so it covers no heading.
    static var agentsPanelMinX: CGFloat { pipelineHeading.maxX + 2 }

    /// 5i, as built: the Search panel's left edge, clear of the weekly clean's widest words ("failed 2 weeks").
    static var searchPanelMinX: CGFloat { point(.weekly).x + 52 }

    /// R1.12 5g, 5h, 5i: the side panels' tops, and where their arrows meet the station (inside its ring, 18 pt below the
    /// body's top so the arrow keeps off the rounded corner).
    static func sideTop(_ s: NDStation) -> CGFloat? {
        switch s {
        case .jobs: point(.queue).y + 48  // 182: under Save queue's status line
        case .weekly: keptSafe.maxY + 2  // 124: under the Kept-safe frame
        default: nil
        }
    }

    static func panel(_ s: NDStation, state: NDPipelineState, actions: Set<NDStation>, actionWidths: [NDStation: CGFloat] = [:],
                      height: CGFloat, width: CGFloat? = nil) -> Panel {
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
        var w = width ?? side.width
        if side.edge == .trailing { w = min(w, windowInside.maxX - (anchor.maxX + a)) }
        let tip: CGPoint
        var body: CGRect
        switch side.edge {
        case .top where s == .agents:
            // 5a: the notch points down at the agents' column's top edge; the body covers the rail line above it (a pipe).
            // R1.13.1: the body starts 2 pt right of the "Pipeline" heading (from the chart's left edge it covered the
            // heading's left), and the notch moves right along the column's top edge to stay under the body's straight
            // edge, still on the column.
            let minX = agentsPanelMinX
            tip = CGPoint(x: min(max(own.midX, minX + arrowInset), own.maxX), y: own.minY)
            body = CGRect(x: minX, y: tip.y - a - height, width: w, height: height)
        case .top:
            tip = CGPoint(x: own.midX, y: anchor.minY)
            body = CGRect(x: tip.x - w / 2 + side.slide, y: tip.y - a - height, width: w, height: height)
        case .leading where sideTop(s) != nil, .trailing where sideTop(s) != nil:
            let top = sideTop(s) ?? 0
            let y = s == .weekly ? own.midY : min(own.maxY, top + arrowInset)
            tip = CGPoint(x: side.edge == .leading ? anchor.minX : anchor.maxX, y: y)
            let x = side.edge == .leading ? tip.x - a - w : tip.x + a
            body = CGRect(x: x, y: top, width: w, height: height)
        case .bottom:
            // The agents' arrow sits left of the answers line rising into the group's bottom edge at its centre.
            tip = CGPoint(x: s == .agents ? own.minX + 10 : own.midX, y: anchor.maxY)
            body = CGRect(x: tip.x - w / 2 + side.slide, y: tip.y + a, width: w, height: height)
        case .trailing where s == .agents:
            // The arrow at the column's first icon; the body ends 4 pt above Read's words and rises from there.
            tip = CGPoint(x: anchor.maxX, y: point(.read).y - 8)
            let ceiling = (labelRects(.read, state).first?.minY ?? (point(.read).y + ringRadius)) - 4
            body = CGRect(x: tip.x + a, y: ceiling - height, width: w, height: height)
        case .trailing:
            tip = CGPoint(x: anchor.maxX, y: own.midY)
            body = CGRect(x: tip.x + a, y: tip.y - height / 2 + side.slide, width: w, height: height)
        case .leading:
            tip = CGPoint(x: anchor.minX, y: own.midY)
            body = CGRect(x: tip.x - a - w, y: tip.y - height / 2 + side.slide, width: w, height: height)
        }
        // Inside the window's sides (R2A item 1; the agents' panel passes the left edge, L39), above the footer for a side
        // panel (1.5.1), then the arrow never leaves the body's straight edge.
        switch side.edge {
        case .top, .bottom:
            if s != .agents { body.origin.x = min(max(body.minX, windowInside.minX), windowInside.maxX - w) }
            body.origin.x = min(max(body.minX, tip.x - w + arrowInset), tip.x - arrowInset)
        case .leading, .trailing:
            // R1.12: a side panel may cover the footer's empty middle, never its texts, and stays inside the window.
            let clearsFooter = !footerTexts.contains { $0.minX < body.maxX && body.minX < $0.maxX }
            body.origin.y = min(body.minY, (clearsFooter ? windowInside.maxY : footerTop - 4) - height)
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
        let nameW = textWidth(name(s), size: 13)  // R2 item 10: the name keeps its normal weight
        _ = t
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
