import AppKit
import SwiftUI

private typealias Say = BrainBarMenuPresenter.SurfaceSay

/// The window (DESIGN-SPEC-R1 § 2.3 to 2.6): a unified 52 pt title bar holding only the centred tabs, one page, and a
/// 28 pt footer that says when it was checked. Each page takes its own height (L17); nothing animates on a tab switch.
struct NDWindowRoot: View {
    @ObservedObject var store: NDStore
    /// Headless renders have no toolbar, so they draw the same row themselves, with stand-in traffic lights.
    var drawsTrafficLights = false

    var body: some View {
        let r1 = store.r1
        VStack(spacing: 0) {
            if drawsTrafficLights {
                NDTitleBar(store: store, drawsTrafficLights: true)
            } else {
                ND.bar.frame(height: NDGrid.titleBar).overlay(alignment: .bottom) { ND.hair.frame(height: 0.5) }
            }
            ScrollView(.vertical) {
                Group {
                    switch store.shownPage {
                    case .overview: NDOverviewPage(store: store).onAppear { drawn(.overview) }
                    case .backups: NDBackupsPage(store: store).onAppear { drawn(.backups) }
                    case .jobs: NDJobsPage(store: store).onAppear { drawn(.jobs) }
                    }
                }
                .padding(.horizontal, NDGrid.pageSide)
                .padding(.top, NDGrid.pageTop)
                .padding(.bottom, 20)
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .scrollIndicators(.automatic)
            // A page longer than its view scrolls under a 28 pt fade with the system scroller (two or more faults).
            .overlay(alignment: .bottom) {
                LinearGradient(colors: [ND.bg.opacity(0), ND.bg], startPoint: .top, endPoint: .bottom)
                    .frame(height: 28)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
            NDFooter(left: r1.footerLeft, right: r1.footerRight)
        }
        .ignoresSafeArea(.container, edges: .top)
        .background(ND.bg)
        .foregroundStyle(ND.t1)
        .environment(\.ndPalette, .window)
    }
}

extension NDWindowRoot {
    /// The page this root last drew (D1's test reads it; nothing else does).
    func drawn(_ page: NDPage) {
#if DEBUG
        store.drawnPageForTesting = page
#endif
    }
}

/// "Checked 1 min ago" · "Next check in 4 min" (finding 8); stale: "Last checked 2 h ago" alone.
struct NDFooter: View {
    let left: String
    let right: String?
    var body: some View {
        HStack {
            Text(left)
            Spacer()
            if let right { Text(right) }
        }
        .font(NDType.captionNumber)
        .foregroundStyle(ND.t3)
        .padding(.horizontal, NDGrid.pageSide)
        .frame(height: NDGrid.footer)
        .background(ND.bar)
        .overlay(alignment: .top) { ND.hair.frame(height: 0.5) }
    }
}

/// The window's unified toolbar: only the centred tabs (the memory count moved into the Memory hub, § 2.3).
@MainActor final class NDToolbar: NSObject, NSToolbarDelegate {
    static let pages = NSToolbarItem.Identifier("brainbar.nd.pages")
    let toolbar: NSToolbar
    private let store: NDStore

    init(store: NDStore) {
        self.store = store
        toolbar = NSToolbar(identifier: "brainbar.nd.toolbar")
        super.init()
        toolbar.delegate = self
        toolbar.displayMode = .iconOnly
        toolbar.allowsUserCustomization = false
        toolbar.centeredItemIdentifiers = [Self.pages]
    }

    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        [.flexibleSpace, Self.pages, .flexibleSpace]
    }

    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        toolbarDefaultItemIdentifiers(toolbar)
    }

    func toolbar(_ toolbar: NSToolbar, itemForItemIdentifier id: NSToolbarItem.Identifier,
                 willBeInsertedIntoToolbar flag: Bool) -> NSToolbarItem? {
        guard id == Self.pages else { return nil }
        let item = NSToolbarItem(itemIdentifier: id)
        item.view = NSHostingView(rootView: NDSegmentedPages(store: store).fixedSize().environment(\.ndPalette, .window))
        item.label = "Pages"
        return item
    }
}

// MARK: - Title bar (headless renders only)

struct NDTitleBar: View {
    @ObservedObject var store: NDStore
    var drawsTrafficLights = false

    var body: some View {
        ZStack {
            HStack {
                if drawsTrafficLights {
                    HStack(spacing: 8) {
                        Circle().fill(Color(nsColor: NDColor.rgb(0xFF5F57))).frame(width: 12, height: 12)
                        Circle().fill(Color(nsColor: NDColor.rgb(0xFEBC2E))).frame(width: 12, height: 12)
                        Circle().fill(Color(nsColor: NDColor.rgb(0x28C840))).frame(width: 12, height: 12)
                    }
                    .padding(.leading, 20)
                }
                Spacer()
            }
            NDSegmentedPages(store: store).fixedSize()
        }
        .frame(height: NDGrid.titleBar)
        .background(ND.bar)
        .overlay(alignment: .bottom) { ND.hair.frame(height: 0.5) }
    }
}

/// The native segmented control with its labels drawn by us (t1 Medium selected, t2 otherwise), and a page with faults
/// carries a 16 pt red-tint count pill after its label. Selection, keyboard and VoiceOver stay native.
struct NDSegments: NSViewRepresentable {
    let titles: [String]
    var counts: [Int] = []
    @Binding var selection: Int
    var accessibilityLabel: String
    var fontSize: CGFloat = 13
    var padding: CGFloat = 14
    var controlSize: NSControl.ControlSize = .regular

    final class Coordinator: NSObject {
        var selection: Binding<Int>
        init(selection: Binding<Int>) { self.selection = selection }
        @MainActor @objc func changed(_ sender: NSSegmentedControl) {
            if selection.wrappedValue != sender.selectedSegment { selection.wrappedValue = sender.selectedSegment }
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator(selection: $selection) }

    func makeNSView(context: Context) -> NSSegmentedControl {
        let control = NSSegmentedControl(labels: titles, trackingMode: .selectOne,
                                         target: context.coordinator, action: #selector(Coordinator.changed(_:)))
        control.segmentStyle = .automatic
        control.controlSize = controlSize
        control.setAccessibilityLabel(accessibilityLabel)
        update(control)
        return control
    }

    func updateNSView(_ control: NSSegmentedControl, context: Context) {
        context.coordinator.selection = $selection
        update(control)
    }

    @MainActor private func update(_ control: NSSegmentedControl) {
        for (index, title) in titles.enumerated() {
            let count = index < counts.count ? counts[index] : 0
            let image = Self.labelImage(title, count: count, selected: index == selection, size: fontSize)
            control.setLabel("", forSegment: index)
            control.setImage(image, forSegment: index)
            control.setImageScaling(.scaleNone, forSegment: index)
            control.setWidth(image.size.width + 2 * padding, forSegment: index)
        }
        control.selectedSegment = selection
    }

    /// "Backups ⓵": the label, then a 16 pt pill in red tint with the count in red text, 6 pt after it. Drawn at
    /// display time, so the colours follow the control's appearance.
    @MainActor static func labelImage(_ title: String, count: Int, selected: Bool, size: CGFloat = 13) -> NSImage {
        let font = NSFont.systemFont(ofSize: size, weight: selected ? .medium : .regular)
        let label = NSAttributedString(string: title, attributes: [.font: font])
        let countFont = NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .semibold)
        let numeral = NSAttributedString(string: String(count), attributes: [.font: countFont])
        let pill = max(16, ceil(numeral.size().width) + 8)
        let width = ceil(label.size().width + (count > 0 ? 6 + pill : 0))
        let height = max(16, ceil(label.size().height))
        let image = NSImage(size: NSSize(width: width, height: height), flipped: false) { rect in
            let y = (rect.height - label.size().height) / 2
            NSAttributedString(string: title, attributes: [.font: font, .foregroundColor: selected ? NDColor.t1 : NDColor.t2])
                .draw(at: NSPoint(x: 0, y: y))
            if count > 0 {
                let box = NSRect(x: label.size().width + 6, y: (rect.height - 16) / 2, width: pill, height: 16)
                NDColor.red.setFill()
                NSBezierPath(roundedRect: box, xRadius: 8, yRadius: 8).fill()
                NSAttributedString(string: String(count), attributes: [.font: countFont, .foregroundColor: NSColor.white])
                    .draw(at: NSPoint(x: box.midX - numeral.size().width / 2, y: box.midY - numeral.size().height / 2))
            }
            return true
        }
        image.isTemplate = false
        image.accessibilityDescription = count > 0
            ? "\(title), \(count == 1 ? BrainBarMenuPresenter.Say.countOne.text : BrainBarMenuPresenter.SurfaceSay.thingsNeedYou(count))"
            : title
        return image
    }
}

struct NDSegmentedPages: View {
    @ObservedObject var store: NDStore
    var body: some View {
        NDSegments(titles: NDPage.allCases.map(\.title),
                   counts: NDPage.allCases.map { [.backups: store.r1.backupsBadge, .jobs: store.r1.jobsBadge][$0] ?? 0 },
                   selection: Binding(get: { NDPage.allCases.firstIndex(of: store.page) ?? 0 },
                                      set: { store.page = NDPage.allCases[max(0, min($0, NDPage.allCases.count - 1))] }),
                   accessibilityLabel: "Pages")
    }
}

// MARK: - Overview

struct NDOverviewPage: View {
    @ObservedObject var store: NDStore
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// R1.2 (H6) changes 1 to 3: the headline, then the Activity and Pipeline blocks, each a heading above one filled,
    /// rounded surface. No dividers, no shadows. (The Problems surface, changes 4 to 7, waits for revision 1.3.)
    var body: some View {
        let r1 = store.r1
        VStack(alignment: .leading, spacing: 0) {
            NDHeadline(store: store, r1: r1)
            NDActivitySection(store: store).padding(.top, 20)
            NDBlockHeading(title: Say.pipeline).padding(.top, 20)
            NDCard(padding: EdgeInsets(top: 16, leading: 16, bottom: 14, trailing: 16)) {
                NDPipelineChart(state: r1.pipeline, memories: store.model.memories, items: r1.stationItems,
                                waiting: store.inputs.driveReconnecting, perform: { store.perform($0) },
                                popover: { station in AnyView(NDStationPopover(store: store, station: station)) })
            }
            .animation(reduceMotion ? nil : NDMotion.breakAppears, value: r1.pipeline)
        }
        .animation(reduceMotion ? .easeOut(duration: 0.15) : NDMotion.needsBand, value: r1.items.map(\.id))
    }
}

/// A block's heading above its surface: 13 Semibold `t1`, 2 pt inset, a 22 pt row, 8 pt to the surface (R1.2 change 1).
struct NDBlockHeading<Trailing: View>: View {
    let title: String
    @ViewBuilder var trailing: Trailing
    var body: some View {
        HStack {
            Text(title).font(NDType.bodyStrong).foregroundStyle(ND.t1).accessibilityAddTraits(.isHeader)
            Spacer()
            trailing
        }
        .frame(height: 22)
        .padding(.horizontal, 2)
        .padding(.bottom, 8)
    }
}

extension NDBlockHeading where Trailing == EmptyView {
    init(title: String) {
        self.title = title
        self.trailing = EmptyView()
    }
}

/// The one surface treatment (R1.2 change 2): `card` fill, a 0.5 pt `card-line` inside, radius 12, no shadow.
struct NDCard<Content: View>: View {
    var padding = EdgeInsets(top: 14, leading: 16, bottom: 16, trailing: 16)
    @ViewBuilder var content: Content
    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(ND.card))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(ND.cardLine, lineWidth: 0.5))
            .environment(\.ndPlane, ND.card)
    }
}

private struct NDPlaneKey: EnvironmentKey { static let defaultValue = ND.bg }
extension EnvironmentValues {
    /// The colour under a drawing (the plane, or a card): masks behind words set in lines use it.
    var ndPlane: Color {
        get { self[NDPlaneKey.self] }
        set { self[NDPlaneKey.self] = newValue }
    }
}

/// A page's first line on Backups and Jobs: the title (17 Semibold) and its detail (13, t2).
struct NDPageTop: View {
    let title: String
    var detail: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(NDType.title).tracking(-0.25).foregroundStyle(ND.t1).frame(height: 22)
            if let detail { Text(detail).font(NDType.body).foregroundStyle(ND.t2) }
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }
}

/// The Overview's first lines (§ 2.3): "All good" alone; one fault: symbol and title, the action right-aligned on the
/// title line, the detail under it; two or more: "N things need you" and one 24 pt line per fault. Not confirmed: the
/// question mark in t2 and its detail, no button. The symbol sits inside the text column.
struct NDHeadline: View {
    @ObservedObject var store: NDStore
    let r1: NDR1

    /// L27: one line. The mark and "2 things need you", or the single fault's own sentence; each problem is carried by
    /// its station (its treatment, its popover, its action at rest). Amber and not confirmed keep their line under the
    /// title (R1.2 change 8, L24).
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .center, spacing: 8) {
                if let symbol = r1.headlineSymbol {
                    Image(systemName: symbol).font(.system(size: 15, weight: .regular)).foregroundStyle(tone(r1.headlineTone))
                        .frame(width: 16, height: 22)
                }
                Text(r1.headline).font(NDType.title).tracking(-0.25).foregroundStyle(ND.t1)
                Spacer(minLength: 0)
            }
            .frame(minHeight: 22)
            .accessibilityElement(children: .combine)
            .accessibilityAddTraits(.isHeader)
            if let detail = r1.headlineDetail, !detail.isEmpty {
                Text(keepingUnits(detail)).font(NDType.body).foregroundStyle(ND.t2).padding(.leading, 24).padding(.top, 2)
            }
        }
    }

    private func tone(_ t: NDModel.Tone) -> Color {
        switch t {
        case .fault: ND.red
        case .warn: ND.orange
        case .fine: ND.t2
        }
    }
}

// MARK: - Activity (§ 3.1)

struct NDActivitySection: View {
    @ObservedObject var store: NDStore
    @State private var hover: Int?
    /// The plot's frame in the section, reported as laid out (D8.1's test).
    var reportPlotFrame: ((CGRect) -> Void)? = nil
    /// R1.6 item 9: the readout's frame in the section, reported as laid out.
    var reportReadoutFrame: ((CGRect) -> Void)? = nil

    init(store: NDStore, hover: Int? = nil, reportPlotFrame: ((CGRect) -> Void)? = nil,
         reportReadoutFrame: ((CGRect) -> Void)? = nil) {
        self.store = store
        _hover = State(initialValue: hover)
        self.reportPlotFrame = reportPlotFrame
        self.reportReadoutFrame = reportReadoutFrame
    }
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.ndPreviewHover) private var previewHover
    /// R1.7 item 7: the wave on screen while the pointer rests on the chart, and where the pointer last was.
    @State private var heldWave: NDWave?
    @State private var lastPointer: CGPoint?

    /// R1.7 item 7: while the pointer rests on the chart, new data waits: the bars and the readout keep the minute under
    /// it; the pointer moving or leaving lets the newest data in.
    static func shownWave(live: NDWave?, held: NDWave?, hovering: Bool) -> NDWave? { hovering ? (held ?? live) : live }

    /// The wave to hold after a pointer event: the newest when the pointer moved, the held one when it did not.
    static func held(after point: CGPoint, last: CGPoint?, held: NDWave?, live: NDWave?) -> NDWave? {
        point == last ? (held ?? live) : live
    }
    static let gutter: CGFloat = 52
    /// R1.2: the chart is 820 wide inside its surface: a 52 pt gutter and a 768 pt plot.
    static let plotWidth: CGFloat = 768
    static let plotHeight: CGFloat = 112

    var body: some View {
        let wave = Self.shownWave(live: store.inputs.heroWaves[store.range], held: heldWave, hovering: hover != nil)
        let cut = Self.noDataFraction(store, wave: wave)
        // R1.6 item 6: not confirmed, the totals say "—".
        let known = wave?.readable == true && cut == nil
        VStack(alignment: .leading, spacing: 0) {
            NDBlockHeading(title: Say.activity) { NDRangeControl(store: store) }
            NDCard { chart(wave: wave, known: known, cut: cut) }
        }
    }

    /// R1.6 item 6: where the last confirmed check falls on the range, 0 to 1 from the left; nil while the state is
    /// confirmed.
    static func noDataFraction(_ store: NDStore, wave: NDWave?, minutes: Int? = nil) -> CGFloat? {
        guard store.r1.unconfirmedSince != nil, let at = store.inputs.capture.verdict?.evaluatedAt else { return nil }
        let span = Double((minutes ?? store.range.minutes) * 60)
        let end = wave?.endsAt ?? store.inputs.capture.now
        return CGFloat(min(1, max(0, (at.timeIntervalSince(end) + span) / span)))
    }

    /// R1.6 item 6: a column can be hovered only before the last confirmed check (past it nothing is known).
    static func hoverable(_ index: Int, count: Int, cut: CGFloat?) -> Bool {
        guard let cut else { return true }
        let pitch = plotWidth / CGFloat(max(count, 1))
        return CGFloat(index + 1) * pitch <= NDWaveform.noDataX(cut, width: plotWidth)
    }

    /// The hero canvas the waveform draws on (the plot and the 8 pt under it for the now tick).
    static let canvas = CGSize(width: plotWidth, height: plotHeight + 8)

    @ViewBuilder private func chart(wave: NDWave?, known: Bool, cut: CGFloat?) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 24) {
                legend(color: ND.saved, value: known ? wave.map { NDFormat.integer($0.savesTotal) } : nil, words: Say.savedByAgents)
                legend(color: ND.read, value: known ? wave.map { NDFormat.integer($0.readsTotal) } : nil, words: Say.readFromConversations)
                Spacer()
            }
            .frame(height: 22)
            .accessibilityElement(children: .combine)
            ZStack(alignment: .topLeading) {
                // R1.4 changes 11 and 12: the halves' maxima moved into the hover readout's first line.
                // "saved" just above the baseline, "read" just below it, in the series' text colours (H1).
                Text(Say.saved).font(NDType.captionMedium).foregroundStyle(ND.savedText)
                    .frame(width: Self.gutter, alignment: .leading).position(x: Self.gutter / 2, y: 53)
                Text(Say.readWord).font(NDType.captionMedium).foregroundStyle(ND.readText)
                    .frame(width: Self.gutter, alignment: .leading).position(x: Self.gutter / 2, y: 72)
                // Council r2a finding 3: the hover reads the pointer in the plot's own coordinates, the same origin the
                // rule is drawn from (before, an offset moved the drawing but not the hit area, 52 pt apart).
                NDWaveform(wave: wave, size: .hero, nowTick: true, hover: shown(wave, cut: cut), noDataFrom: cut)
                    .frame(width: Self.plotWidth, height: Self.plotHeight + 8)
                    .contentShape(Rectangle())
                    .onContinuousHover(coordinateSpace: .local) { phase in
                        switch phase {
                        case let .active(p):
                            let live = store.inputs.heroWaves[store.range]
                            heldWave = Self.held(after: p, last: lastPointer, held: heldWave, live: live)
                            lastPointer = p
                            hover = Self.bucket(at: p.x, count: heldWave?.count ?? 0)
                        case .ended:
                            hover = nil
                            heldWave = nil
                            lastPointer = nil
                        }
                    }
                    .background(GeometryReader { g in
                        Color.clear.onAppear { reportPlotFrame?(g.frame(in: .named("activity"))) }
                    })
                    .padding(.leading, Self.gutter)
                    .id(store.range)
                    .transition(reduceMotion ? .identity : .asymmetric(insertion: .opacity.animation(NDMotion.rangeIn),
                                                                        removal: .opacity.animation(NDMotion.rangeOut)))
            }
            .frame(width: Self.gutter + Self.plotWidth, height: Self.plotHeight + 8, alignment: .topLeading)
            // R1.6 item 6: the hatch's words, centred in it.
            .overlay(alignment: .topLeading) {
                if let cut, let at = store.inputs.capture.verdict?.evaluatedAt {
                    let x0 = NDWaveform.noDataX(cut, width: Self.plotWidth)
                    Text(Say.noDataSince(NDR1.hhmm(at, store.inputs.capture)))
                        .font(NDType.caption).foregroundStyle(ND.t2)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(RoundedRectangle(cornerRadius: 4, style: .continuous).fill(ND.card))
                        .fixedSize()
                        .frame(width: max(0, Self.plotWidth - x0), height: Self.plotHeight)
                        .offset(x: Self.gutter + x0)
                        .accessibilityHidden(true)
                }
            }
            // D8.1: the readout is an overlay: it takes no layout space, so the plot never moves when it appears.
            // R1.6 item 9: placed by a layout that measures it (no alignment guides), beside its column.
            .overlay(alignment: .topLeading) {
                if let index = shown(wave, cut: cut), let wave {
                    let c = Self.center(index, count: wave.count)
                    NDReadoutPlacement(column: Self.gutter + c,
                                       barTop: NDWaveform.columnTop(wave, index: index, canvas: Self.canvas, nowTick: true, size: .hero),
                                       flips: Self.readoutFlips(index, count: wave.count)) {
                        NDActivityReadout(wave: wave, index: index, minutes: store.range.minutes, now: store.inputs.capture.now,
                                          zone: store.inputs.capture.timeZone)
                            .fixedSize()
                            .background(GeometryReader { g in
                                Color.clear.onAppear { reportReadoutFrame?(g.frame(in: .named("activity"))) }
                            })
                    }
                    .allowsHitTesting(false)
                }
            }
            .padding(.top, 14)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(NDModel.waveAccessibility(wave, minutes: store.range.minutes))
            let axis = store.range.axis
            ZStack {
                HStack {
                    Text(axis.0)
                    Spacer()
                    Text(axis.2)
                }
                Text(axis.1)
            }
            .padding(.leading, Self.gutter)
            .font(NDType.captionNumber)
            .foregroundStyle(ND.t3)
            .frame(height: 14)
            .padding(.top, 4)
            // R2A item 4: the axis words swap at once, never crossfading two texts.
            .transaction { $0.animation = nil }
            .accessibilityHidden(true)
        }
        .coordinateSpace(name: "activity")
    }

    /// The hovered column, when it is readable and before any unconfirmed stretch (the harness can preset one).
    private func shown(_ wave: NDWave?, cut: CGFloat?) -> Int? {
        guard let index = hover ?? previewHover, let wave, wave.readable, index >= 0, index < wave.count,
              Self.hoverable(index, count: wave.count, cut: cut) else { return nil }
        return index
    }

    /// R2A item 2 and R1.6 item 9: the readout flips to the column's left in the chart's right third.
    static func readoutFlips(_ index: Int, count: Int) -> Bool { center(index, count: count) >= plotWidth * 2 / 3 }

    static func center(_ index: Int, count: Int) -> CGFloat {
        let pitch = plotWidth / CGFloat(max(count, 1))
        return CGFloat(index) * pitch + pitch / 2
    }

    /// The bucket under a pointer given in the chart row's coordinates (the gutter, then the plot): what the hover
    /// selects, through the same origin the rule is drawn from.
    static func hoverIndex(atChartX x: CGFloat, count: Int) -> Int? { bucket(at: x - gutter, count: count) }

    /// The bucket under the pointer: the pair whose pitch holds it; outside the plot, none.
    static func bucket(at x: CGFloat, count: Int) -> Int? {
        guard count > 0, x >= 0, x < plotWidth else { return nil }
        return min(count - 1, Int(x / (plotWidth / CGFloat(count))))
    }

    private func legend(color: Color, value: String?, words: String) -> some View {
        // R1.2 change 19: the hour's totals are the calm page's hero numbers, at the title size.
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 8, height: 8).alignmentGuide(.firstTextBaseline) { d in d.height }
            Text(value ?? NDModel.Number.unknown.value).font(NDType.title.monospacedDigit()).foregroundStyle(ND.t1)
            Text(words).font(NDType.body).foregroundStyle(ND.t2)
        }
    }
}

/// The hover readout (H2): the bucket's time range, then each half's count and words. An overlay, not a tooltip, so it
/// appears with no delay and moves in place.
struct NDActivityReadout: View {
    let wave: NDWave
    let index: Int
    let minutes: Int
    let now: Date
    let zone: TimeZone

    var body: some View {
        let width = Double(minutes * 60) / Double(max(wave.count, 1))
        let end = (wave.endsAt ?? now).addingTimeInterval(-Double(wave.count - 1 - index) * width)
        let start = end.addingTimeInterval(-width)
        // R1.6 item 9: its lines describe the hovered column only.
        VStack(alignment: .leading, spacing: 0) {
            Text(Self.clock(start, zone) + " to " + Self.clock(end, zone)).font(NDType.captionNumber).foregroundStyle(ND.t2)
                .padding(.bottom, 3)
            row(ND.saved, index < wave.saves.count ? wave.saves[index] : 0, Say.savedByAgents)
            row(ND.read, index < wave.reads.count ? wave.reads[index] : 0, Say.readWord)
        }
        .padding(.horizontal, 9).padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(ND.pop)
            .shadow(color: .black.opacity(0.28), radius: 9, y: 6))
        .overlay(RoundedRectangle(cornerRadius: 6, style: .continuous).stroke(ND.hair2, lineWidth: 0.5))
    }

    private func row(_ color: Color, _ value: Int, _ words: String) -> some View {
        HStack(spacing: 6) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 8, height: 8)
            Text(NDFormat.integer(value)).font(NDType.numberStrong).foregroundStyle(ND.t1)
            Text(words).font(NDType.body).foregroundStyle(ND.t2)
        }
        .frame(height: 18)
    }

    static func clock(_ date: Date, _ zone: TimeZone) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = zone
        f.dateFormat = "HH:mm"
        return f.string(from: date)
    }
}

/// R1.6 item 9: places the hover readout by measuring it: beside the hovered column, 12 pt right of it (left of it in
/// the chart's right third), its bottom 8 pt above the column's tallest bar. A tall column would lift it into the totals
/// row, so its top stops at the row's foot; beside the column it still never covers it. Coordinates: the chart row's
/// (the gutter, then the plot; y 0 at the plot's top).
struct NDReadoutPlacement: Layout {
    let column: CGFloat
    let barTop: CGFloat
    let flips: Bool
    static let gap: CGFloat = 12
    static let lift: CGFloat = 8
    /// The totals row's foot, above the plot's top (the row's 14 pt padding).
    static let ceiling: CGFloat = -14

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        proposal.replacingUnspecifiedDimensions()
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        guard let readout = subviews.first else { return }
        let size = readout.sizeThatFits(.unspecified)
        let origin = Self.origin(column: column, barTop: barTop, flips: flips, size: size)
        readout.place(at: CGPoint(x: bounds.minX + origin.x, y: bounds.minY + origin.y), anchor: .topLeading,
                      proposal: ProposedViewSize(size))
    }

    static func origin(column: CGFloat, barTop: CGFloat, flips: Bool, size: CGSize) -> CGPoint {
        CGPoint(x: flips ? column - gap - size.width : column + gap, y: max(ceiling, barTop - lift - size.height))
    }
}

/// The range switch (1 h, 3 h, 24 h): the system's small segmented control.
struct NDRangeControl: View {
    @ObservedObject var store: NDStore
    var body: some View {
        NDSegments(titles: NDRange.allCases.map(\.label),
                   selection: Binding(get: { NDRange.allCases.firstIndex(of: store.range) ?? 0 },
                                      set: { store.range = NDRange.allCases[max(0, min($0, NDRange.allCases.count - 1))] }),
                   accessibilityLabel: "Range", fontSize: 11, padding: 10, controlSize: .small)
            .fixedSize()
    }
}

// MARK: - Buttons (§ 3.3)

/// The filled button (Reconnect only, R1.1 change 23) or a bordered one: 24 pt (22 small), 6 pt corners; hover and press
/// tints, never a scale (change 22).
struct NDButton: View {
    let label: String
    var filled = false
    var small = false
    /// R1.3 change 16: the action under a station (20 pt high, 10 pt sides, 13 pt).
    var compact = false
    /// The spoken name when the visible one leans on what stands beside it ("Reconnect" under Google Drive).
    var accessibility: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) { Text(label) }
            .buttonStyle(NDButtonStyle(filled: filled, small: small, compact: compact))
            .ndKeyboardFocus(action: action)
            .accessibilityLabel(Text(accessibility ?? label))
    }
}

struct NDButtonStyle: ButtonStyle {
    var filled = false
    var small = false
    var compact = false

    func makeBody(configuration: Configuration) -> some View {
        NDButtonBody(configuration: configuration, filled: filled, small: small, compact: compact)
    }

    private struct NDButtonBody: View {
        let configuration: Configuration
        let filled: Bool
        let small: Bool
        var compact = false
        @State private var hovering = false
        /// D8.2: while the button leaves for the waiting line it keeps its pressed look: no rest frame after the press.
        @Environment(\.ndHeldPressed) private var held

        var body: some View {
            let pressed = configuration.isPressed || held
            let fill: Color = filled ? (pressed ? ND.accentPressed : hovering ? ND.accentHover : ND.accent)
                : (pressed ? ND.buttonPressed : hovering ? ND.buttonHover : ND.button)
            configuration.label
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(filled ? Color.white : ND.t1)
                .lineLimit(1)
                .padding(.horizontal, small || compact ? 10 : 12)
                .frame(height: compact ? 20 : small ? 22 : 24)
                .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(fill))
                .overlay {
                    // R1.1 change 22: hover adds a 0.5 pt inner highlight (filled) or a 0.5 pt t3 inner stroke (bordered);
                    // nothing scales, the size never changes.
                    if filled {
                        RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(Color.white.opacity(hovering ? 0.25 : 0), lineWidth: 0.5)
                    } else {
                        RoundedRectangle(cornerRadius: 6, style: .continuous).strokeBorder(hovering ? ND.t3 : ND.hair2, lineWidth: 0.5)
                    }
                }
                .contentShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                .onHover { hovering = $0 }
        }
    }
}

// MARK: - Page height (L17)

/// The window's height for a page: the title bar, the page's laid-out content with its insets, and the footer; never
/// below the spec's page height (Overview 716, Backups 480, Jobs 512), capped at the screen. A two-fault headline is
/// two 24 pt rows taller than the calm one, so the Overview grows with it; past the screen the page scrolls (§ 2.3).
@MainActor enum NDPageHeight {
    /// Title bar 52, page padding 20 top and 20 bottom (R1.2 change 2), footer 28.
    static let chrome: CGFloat = NDGrid.titleBar + NDGrid.pageTop + 20 + NDGrid.footer

    /// The page's content height at the page width (852 pt), as SwiftUI lays it out.
    static func content(_ store: NDStore, page: NDPage) -> CGFloat {
        let width = NDGrid.windowWidth - 2 * NDGrid.pageSide
        let view: AnyView = switch page {
        case .overview: AnyView(NDOverviewPage(store: store))
        case .backups: AnyView(NDBackupsPage(store: store))
        case .jobs: AnyView(NDJobsPage(store: store))
        }
        let hosting = NSHostingController(rootView: view.frame(width: width).environment(\.ndPalette, .window))
        return ceil(hosting.sizeThatFits(in: CGSize(width: width, height: CGFloat.greatestFiniteMagnitude)).height)
    }

    /// R1.2 change 22 and L25: the page's real height in every state (nothing reserved), up to the screen's visible
    /// height minus 40 pt; past that the page scrolls inside the window and the footer stays.
    static func window(_ store: NDStore, page: NDPage, screen: CGFloat?) -> CGFloat {
        let needed = ceil(content(store, page: page) + chrome)
        guard let screen, screen > 0 else { return needed }
        return min(needed, screen - 40)
    }

    /// What decides the page's height: the page and the headline's lines.
    static func key(_ store: NDStore) -> String {
        let r1 = store.r1
        let parts: [String] = [store.page.rawValue, r1.headline, r1.unconfirmedDetail ?? ""] + r1.items.map(\.id)
        return parts.joined(separator: "|")
    }
}

private struct NDPreviewHoverKey: EnvironmentKey { static let defaultValue: Int? = nil }
extension EnvironmentValues {
    /// Render harness only: the Activity chart's hovered column (R1.6 item 9: the readout drawn as the window draws it).
    var ndPreviewHover: Int? {
        get { self[NDPreviewHoverKey.self] }
        set { self[NDPreviewHoverKey.self] = newValue }
    }
}
