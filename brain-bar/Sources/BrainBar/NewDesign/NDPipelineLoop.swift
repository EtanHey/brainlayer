import AppKit
import SwiftUI

// MARK: - Layout (R1.3 with 1.3.1; R1.4 changes 1, 17 and 19), as data

enum NDPipelineLayout {
    typealias Say = BrainBarMenuPresenter.SurfaceSay

    /// L63: the return caption clears the lowest status by 32 pt; Search finishes the card.
    static let nameOffset: CGFloat = 17, purposeOffset: CGFloat = 37
    static let statusOffset: CGFloat = 33, purposeStatusOffset: CGFloat = 57, statusHeight: CGFloat = 13
    static let yJobs: CGFloat = 138
    static let yAnswers: CGFloat = yJobs + purposeStatusOffset + statusHeight + 32 + 8
    static let size = CGSize(width: 820, height: yAnswers + statusOffset + statusHeight + 2)
    static let verticalPadding: CGFloat = 4
    static let drawWidth: CGFloat = 832
    static let yTop: CGFloat = 44, ySaves: CGFloat = 114, yRail: CGFloat = 8
    static let stationRadius: CGFloat = 13
    /// The fault ring's radius (R1.4 change 16).
    static let ringRadius: CGFloat = 15.5
    /// R1.4 change 17: every line ends 3 pt outside the outermost ring a station can have, in every state, arrow tips
    /// included; at the Memory box and the agents' frame, 3 pt from the edge.
    static let lineEnd: CGFloat = 18.5
    static let edgeGap: CGFloat = 3
    /// The agents: four provider icons in one column (R1.4 change 1).
    static let agentsFrame = CGRect(x: 16, y: 18, width: 44, height: 108)
    static let agentRows: [(AgentFamily, CGFloat)] = AgentFamily.allCases.enumerated().map { ($0.element, 36 + 24 * CGFloat($0.offset)) }
    static let iconSize: CGFloat = 16
    static func iconCentre(_ family: AgentFamily) -> CGPoint {
        CGPoint(x: agentsFrame.midX, y: agentRows.first { $0.0 == family }?.1 ?? agentsFrame.midY)
    }
    /// R14: The lower Memory port sits 3 pt below the box, 8 pt below the Kept-safe frame.
    /// R2 item 8: the diagram sits centred in its card, 32 pt from each side: everything right of the agents moved 26 pt
    /// left (the long first-row lines are shorter).
    /// R1.10 (L49): 140 wide, so the two job stations hang under it with straight links. R1.10.4: everything right of the
    /// agents moved 15 pt left (the first-row lines are 15 pt shorter), so the diagram is 32 pt from each side again.
    static let hub = CGRect(x: 365, y: 24, width: 140, height: 83)
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
    static let keptSafe = CGRect(x: 521, y: 18, width: 283, height: 84)

    /// R1.12 change 6: Search moves under Google Drive, to the loop's bottom-right corner; Memory → Search runs right
    /// along the copies' lower edge and down, and the answers return straight along yAnswers (gap 79 / L56).
    static let stations: [NDStation: CGPoint] = [
        .read: CGPoint(x: 215, y: yTop), .queue: CGPoint(x: 215, y: ySaves), .search: CGPoint(x: 736, y: yAnswers),
        .mac: CGPoint(x: 599, y: yTop), .drive: CGPoint(x: 736, y: yTop), .jobs: CGPoint(x: 313, y: yJobs), .weekly: CGPoint(x: 650, y: yJobs),
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
        out.append(Connection(pipe: .queueMemory, line: NDPolyline(points: [queue, CGPoint(x: 280, y: ySaves),
                    CGPoint(x: 280, y: 88), CGPoint(x: hub.minX - g, y: 88)]).trimmed(start: e)))
        // R14: leave the lower Memory port 8 pt below Kept-safe, then turn once into Search.
        // The hub ends 3 pt above the port, preserving the endpoint gap and its bottom text padding.
        let searchLaneY = keptSafe.maxY + 8
        out.append(Connection(pipe: .memorySearch, line: NDPolyline(points: [CGPoint(x: hub.maxX - 12, y: searchLaneY),
                    CGPoint(x: search.x, y: searchLaneY), search]).trimmed(end: e), corner: 10))
        out.append(Connection(pipe: .answers, line: NDPolyline(points: [search, CGPoint(x: ax, y: yAnswers),
                                                                         CGPoint(x: ax, y: agentsFrame.maxY + g)]).trimmed(start: e), corner: 12))
        out.append(Connection(pipe: .memoryMac, line: NDPolyline(points: [CGPoint(x: hub.maxX + g, y: yTop), mac]).trimmed(end: e)))
        out.append(Connection(pipe: .macDrive, line: NDPolyline(points: [mac, drive]).trimmed(start: e, end: e)))
        out.append(Connection(pipe: .rail, line: NDPolyline(points: [CGPoint(x: ax, y: agentsFrame.minY - g), CGPoint(x: ax, y: yRail),
                                                                     CGPoint(x: drive.x, y: yRail), drive]).trimmed(end: e)))
        out.append(Connection(pipe: .jobsMemory, line: NDPolyline(points: [jobs,
                    CGPoint(x: hub.minX + 12, y: jobs.y), CGPoint(x: hub.minX + 12, y: hub.maxY + g)]).trimmed(start: e),
                              bothWays: true))
        out.append(Connection(pipe: .weeklyMemory, line: NDPolyline(points: [weekly,
                    CGPoint(x: hub.maxX - 12, y: weekly.y), CGPoint(x: hub.maxX - 12, y: hub.maxY + g)]).trimmed(start: e),
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
            // L63: clip in place; sliding a side panel onto the screen can cover another station.
            if p.body.minY < limit.minY {
                let bottom = p.body.maxY
                p.body.origin.y = limit.minY
                p.body.size.height = max(32, bottom - limit.minY)
                p.scrolls = true
            }
            if p.body.maxY > limit.maxY {
                p.body.size.height = max(32, limit.maxY - p.body.minY)
                p.scrolls = true
            }
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
        enum ClearanceException: Equatable { case topRowBelowBothSidesBlocked }
        var clearanceException: ClearanceException? = nil
    }

    static let arrowLength: CGFloat = 10
    /// 1.5.1: what lies outside the chart's surface that no panel may cover: the "Pipeline" block heading above it and the
    /// footer's two lines below it, in chart coordinates (surface padding 16 / 12, heading row 22 + 8, page padding 20,
    /// footer 28, page sides 24).
    static var footerTop: CGFloat { size.height + verticalPadding + 20 }
    /// R1.12 (§ B): the Activity card's bottom edge in chart coordinates (surface padding 16, heading row 22 + 8, the
    /// 32 pt gap): no panel reaches into the card, its bars or its axis labels.
    static var activityBottom: CGFloat { -verticalPadding - 8 - 22 - NDGrid.activityToPipeline }
    /// The Activity card's axis labels end its 16 pt bottom padding above its edge: a panel may cover that empty padding
    /// (the comps do), never the bars or the labels.
    static var activityLabelsBottom: CGFloat { activityBottom - 16 }
    static var pipelineHeading: CGRect {
        let w = textWidth(Say.pipeline, size: 13, weight: .semibold)
        return CGRect(x: -16 + 2, y: -verticalPadding - 8 - 22, width: w, height: 22)
    }
    static var footerTexts: [CGRect] {
        let left = [Say.statusChecked("59 min"), Say.waitingForNextCheck].map { textWidth($0, size: 11) }.max() ?? 0
        let right = textWidth(Say.nextStatusCheck("59 min"), size: 11)
        let y = footerTop + 7
        return [CGRect(x: -16, y: y, width: left, height: 14), CGRect(x: size.width + 16 - right, y: y, width: right, height: 14)]
    }
    /// R2A item 1: a panel keeps 12 pt inside the window's content edge. In chart coordinates the window runs 40 pt past
    /// each side of the chart (page side 24, surface padding 16) and 60 pt below it (surface 12, page 20, footer 28).
    static let windowInside = (minX: CGFloat(-28), maxX: CGFloat(848), maxY: size.height + verticalPadding + 20 + 28 - 12)
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

    /// Preferred widths and initial edge; placement restores clearance without moving the notch.
    static func panelSide(_ s: NDStation, hasAction: Bool) -> (edge: Edge, width: CGFloat, slide: CGFloat) {
        switch s {
        case .agents, .read, .memory: (.top, 300, 0)
        case .mac, .drive: (.top, 320, 0)
        case .queue: (.bottom, 262, 0)
        case .jobs, .weekly: (.bottom, 400, 0)
        case .search: (.top, searchPanelWidth, 0)
        }
    }

    static let searchPanelWidth: CGFloat = 264

    static let topRowPanels: Set<NDStation> = [.agents, .read, .memory, .mac, .drive]

    static func panelWords(_ s: NDStation, state: NDPipelineState, displayedWords: [NDStation: [CGRect]]) -> [CGRect] {
        displayedWords[s] ?? labelRects(s, state)
    }

    static func panelBodyObstacles(_ s: NDStation, interior: Bool) -> [CGRect] {
        interior ? NDStation.allCases.filter { $0 != s }.map { circle($0) } : []
    }

    /// L58: body clearance can move the body, never the target-aligned notch.
    static func panelObstacles(_ s: NDStation, state: NDPipelineState, actions: Set<NDStation>,
                               actionWidths: [NDStation: CGFloat], interior: Bool, displayedWords: [NDStation: [CGRect]] = [:]) -> [CGRect] {
        var obstacles = [pipelineHeading] + footerTexts + panelWords(s, state: state, displayedWords: displayedWords) + [circle(s).insetBy(dx: 1, dy: 1)]
        if interior {
            for other in NDStation.allCases where other != s {
                let words = panelWords(other, state: state, displayedWords: displayedWords)
                // L63b: Memory uses the pocket between the jobs, with 4 pt clear of their words.
                obstacles += s == .memory && (other == .jobs || other == .weekly || other == .queue)
                    ? words.map { $0.insetBy(dx: -4, dy: -4) } : words
            }
            // R13: the two job panels clear the native caption fields, including their drawing insets.
            if s == .jobs || s == .weekly {
                obstacles += laneLabelRects().map { $0.1.insetBy(dx: -4, dy: -4) }
            }
        }
        for station in actions {
            if let width = actionWidths[station], interior || station == s {
                obstacles.append(actionFrame(station, width: width, hasStatus: state.status[station] != nil))
            }
        }
        return obstacles
    }

    /// A full-height candidate at one edge. No narrowing: content is measured at this same width for every candidate.
    static func panelCandidate(_ s: NDStation, edge: Edge, state: NDPipelineState, actions: Set<NDStation>,
                               actionWidths: [NDStation: CGFloat] = [:], height: CGFloat,
                               width overrideWidth: CGFloat? = nil, allowsInteriorOverlap: Bool = false,
                               displayedWords: [NDStation: [CGRect]] = [:]) -> Panel? {
        let own = circle(s), width = overrideWidth ?? panelSide(s, hasAction: actions.contains(s)).width
        let words = panelWords(s, state: state, displayedWords: displayedWords)
        let anchor = words.reduce(own) { $0.union($1) }
        let horizontal = edge == .top || edge == .bottom
        let tip: CGPoint
        let body: CGRect
        switch edge {
        case .top:
            tip = CGPoint(x: own.midX, y: own.minY)
            body = CGRect(x: tip.x - width / 2, y: tip.y - arrowLength - height, width: width, height: height)
        case .bottom:
            tip = CGPoint(x: own.midX, y: anchor.maxY)
            body = CGRect(x: tip.x - width / 2, y: tip.y + arrowLength, width: width, height: height)
        case .trailing:
            tip = CGPoint(x: anchor.maxX, y: own.midY)
            body = CGRect(x: tip.x + arrowLength, y: tip.y - height / 2, width: width, height: height)
        case .leading:
            tip = CGPoint(x: anchor.minX, y: own.midY)
            body = CGRect(x: tip.x - arrowLength - width, y: tip.y - height / 2, width: width, height: height)
        }
        let ceiling = s == .search ? keptSafe.maxY + 4 : activityLabelsBottom + 4
        let low = horizontal ? max(windowInside.minX, tip.x - width + arrowInset) : max(ceiling, tip.y - height + arrowInset)
        let high = horizontal ? min(windowInside.maxX - width, tip.x - arrowInset) : min(footerTop - 4 - height, tip.y - arrowInset)
        guard low <= high else { return nil }
        let arrowBox = horizontal
            ? CGRect(x: tip.x - arrowHalfWidth, y: min(tip.y, edge == .top ? body.maxY : body.minY), width: 2 * arrowHalfWidth, height: arrowLength)
            : CGRect(x: min(tip.x, edge == .leading ? body.maxX : body.minX), y: tip.y - arrowHalfWidth, width: arrowLength, height: 2 * arrowHalfWidth)
        let obstacles = panelObstacles(s, state: state, actions: actions, actionWidths: actionWidths, interior: true, displayedWords: displayedWords)
        let bodyObstacles = panelBodyObstacles(s, interior: true)
        let desired = horizontal ? body.minX : body.minY
        // The nearest legal point is either the centre, a window limit or an obstacle boundary.
        var positions = [min(max(desired, low), high), low, high]
        for rect in obstacles + bodyObstacles {
            positions += horizontal ? [rect.minX - width, rect.maxX] : [rect.minY - height, rect.maxY]
        }
        for position in positions.sorted(by: { abs($0 - desired) < abs($1 - desired) }) where position >= low && position <= high {
            var placed = body
            if horizontal { placed.origin.x = position } else { placed.origin.y = position }
            let outer = placed.union(arrowBox)
            guard placed.minX >= windowInside.minX, placed.maxX <= windowInside.maxX,
                  placed.minY >= ceiling, placed.maxY <= footerTop - 4,
                  !obstacles.contains(where: { outer.intersects($0) }),
                  !bodyObstacles.contains(where: { placed.intersects($0) }) else { continue }
            return Panel(edge: edge, anchor: anchor, tip: tip, body: placed, outer: outer,
                         clearanceException: nil)
        }
        return nil
    }

    /// L58b: progressively widen above, preserving content measurement and the same window margins.
    static func widenedPanelWidths(_ s: NDStation) -> [CGFloat] {
        let normal = panelSide(s, hasAction: false).width
        let limit = windowInside.maxX - windowInside.minX
        guard normal < limit else { return [] }
        let widths = stride(from: normal + 40, to: limit, by: 40).map { $0 }
        return Array(Set(widths + [800, limit])).filter { $0 > normal && $0 <= limit }.sorted()
    }

    @MainActor static func measuredPanel(_ s: NDStation, state: NDPipelineState, displayedWords: [NDStation: [CGRect]] = [:],
                              heightAtWidth: (CGFloat) -> CGFloat) -> Panel {
        let normal = panelSide(s, hasAction: false).width
        let height = heightAtWidth(normal)
        if s == .memory {
            let left = [.jobs, .queue].flatMap { panelWords($0, state: state, displayedWords: displayedWords) }.map(\.maxX).max()! + 4
            let right = panelWords(.weekly, state: state, displayedWords: displayedWords).map(\.minX).min()! - 4
            let width = min(normal, right - left)
            if width >= 2 * arrowInset,
               let below = panelCandidate(s, edge: .bottom, state: state, actions: [], height: heightAtWidth(width),
                                          width: width, displayedWords: displayedWords) { return below }
        }
        if topRowPanels.contains(s) {
            if let above = panelCandidate(s, edge: .top, state: state, actions: [], height: height,
                                          width: normal, displayedWords: displayedWords) { return above }
            for width in widenedPanelWidths(s) {
                if let above = panelCandidate(s, edge: .top, state: state, actions: [], height: heightAtWidth(width),
                                              width: width, displayedWords: displayedWords) { return above }
            }
        }
        // L63: use free space beside the spread stations before giving up content height.
        let edges: [Edge] = topRowPanels.contains(s) ? [.trailing, .leading, .bottom] : [panelSide(s, hasAction: false).edge, .trailing, .leading, .top, .bottom]
        for edge in edges {
            if let candidate = panelCandidate(s, edge: edge, state: state, actions: [], height: height,
                                              width: normal, displayedWords: displayedWords) { return candidate }
        }
        for width: CGFloat in [normal + 40, normal + 80, normal + 160, normal + 240, 280, 264, 240, 220, 200, 180, 160] where width != normal && width <= windowInside.maxX - windowInside.minX {
            for edge in edges {
                if let candidate = panelCandidate(s, edge: edge, state: state, actions: [], height: heightAtWidth(width),
                                                  width: width, displayedWords: displayedWords) { return candidate }
            }
        }
        return panel(s, state: state, actions: [], height: height, width: normal, displayedWords: displayedWords)
    }

    static func panel(_ s: NDStation, state: NDPipelineState, actions: Set<NDStation>, actionWidths: [NDStation: CGFloat] = [:],
                      height: CGFloat, width overrideWidth: CGFloat? = nil, displayedWords: [NDStation: [CGRect]] = [:]) -> Panel {
        let preferred = panelSide(s, hasAction: actions.contains(s)).edge
        if let panel = panelCandidate(s, edge: preferred, state: state, actions: actions, actionWidths: actionWidths, height: height, width: overrideWidth, displayedWords: displayedWords) {
            return panel
        }
        if topRowPanels.contains(s) {
            for side: Edge in [.trailing, .leading] {
                if let panel = panelCandidate(s, edge: side, state: state, actions: actions, actionWidths: actionWidths, height: height, width: overrideWidth, displayedWords: displayedWords) {
                    return panel
                }
            }
            if let panel = panelCandidate(s, edge: .bottom, state: state, actions: actions, actionWidths: actionWidths,
                                          height: height, width: overrideWidth, displayedWords: displayedWords) {
                return panel
            }
        }
        // Content beyond the window capacity retains the screen-fit scrolling behaviour.
        let ownBottom = panelWords(s, state: state, displayedWords: displayedWords).reduce(circle(s).maxY) { max($0, $1.maxY) }
        let bottomHeight = max(32, min(height, footerTop - 4 - ownBottom - arrowLength))
        if topRowPanels.contains(s), var panel = panelCandidate(s, edge: .bottom, state: state, actions: actions,
                                                                actionWidths: actionWidths, height: bottomHeight, width: overrideWidth,
                                                                displayedWords: displayedWords) {
            panel.scrolls = bottomHeight < height
            return panel
        }
        let room = preferred == .top ? circle(s).minY - arrowLength - (keptSafe.maxY + 4) : footerTop - 4 - ownBottom - arrowLength
        let fittedHeight = max(32, min(height, room))
        if var panel = panelCandidate(s, edge: preferred, state: state, actions: actions, actionWidths: actionWidths, height: fittedHeight, width: overrideWidth, displayedWords: displayedWords) {
            panel.scrolls = fittedHeight < height
            return panel
        }
        // Even oversized content must use a checked body, never the old unchecked overlap fallback.
        // 40 pt keeps the notch clear of both rounded corners on either axis (2 * arrowInset).
        for candidateWidth in [overrideWidth ?? panelSide(s, hasAction: actions.contains(s)).width, 264, 220, 180, 160, 120, 80, 40] {
            for edge: Edge in [preferred, .top, .trailing, .leading, .bottom] {
                if var candidate = panelCandidate(s, edge: edge, state: state, actions: actions,
                    actionWidths: actionWidths, height: 40, width: candidateWidth, displayedWords: displayedWords) {
                    candidate.scrolls = true
                    return candidate
                }
            }
        }
        preconditionFailure("Pipeline geometry must provide a clear minimum hover body")
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

    static func purpose(_ s: NDStation) -> String? {
        switch s {
        case .jobs: Say.nightlyPurpose
        case .weekly: Say.weeklyPurpose
        default: nil
        }
    }

    /// The job purposes occupy separate columns; timestamps remain on their own line, 4 pt below.
    static func purposeRect(_ s: NDStation) -> CGRect? {
        guard let words = purpose(s) else { return nil }
        let p = point(s)
        let w = textWidth(words, size: 11) + 8  // R13: NSTextField adds 4 pt of drawing inset on each side.
        return CGRect(x: p.x - w / 2, y: p.y + purposeOffset, width: w, height: 16)
    }

    static func statusTop(_ s: NDStation) -> CGFloat { point(s).y + (purpose(s) == nil ? statusOffset : purposeStatusOffset) }

    /// A station's name and status boxes as drawn for `state`.
    static func labelRects(_ s: NDStation, _ state: NDPipelineState) -> [CGRect] {
        guard stations[s] != nil else { return [] }
        let p = point(s)
        let t = state.treatment(s)
        let nameW = textWidth(name(s), size: 13)  // R2 item 10: the name keeps its normal weight
        _ = t
        var out = [CGRect(x: p.x - nameW / 2, y: p.y + nameOffset, width: nameW, height: 16)]
        if let purpose = purposeRect(s) { out.append(purpose) }
        if let status = state.status[s] {
            let w = textWidth(status.text, size: 11, weight: t == .fault || t == .notice ? .semibold : .regular)
            out.append(CGRect(x: p.x - w / 2, y: statusTop(s), width: w, height: statusHeight))
        }
        return out
    }

    /// Only the two otherwise ambiguous routes get a short payload caption at rest.
    static func laneLabelRects() -> [(String, CGRect)] {
        [(Say.sessionBackups, yRail), (Say.searchAnswers, yAnswers)].map { words, y in
            let w = textWidth(words, size: 11, weight: .medium)  // Gap 74 captions keep L54's type scale.
            return (words, CGRect(x: 215 - w / 2, y: y - 8, width: w, height: 16))
        }
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
