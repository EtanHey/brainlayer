import AppKit
import SwiftUI

/// The new window (DESIGN-SPEC § 6.3, 6.4): a fixed 52 pt title bar with the native segmented control, then one page.
/// The page scrolls under the title bar when it must (three or more faults, or a small window). No window animation:
/// F4 is a keyboard action (motion 2); switching segments is instant (motion 3).
struct NDWindowRoot: View {
    @ObservedObject var store: NDStore
    /// Draws stand-in traffic lights for headless renders only; the real window draws its own.
    var drawsTrafficLights = false

    var body: some View {
        VStack(spacing: 0) {
            NDTitleBar(store: store, drawsTrafficLights: drawsTrafficLights)
            ScrollView(.vertical) {
                Group {
                    switch store.page {
                    case .overview: NDOverviewPage(store: store)
                    case .backups: NDBackupsPage(store: store)
                    case .jobs: NDJobsPage(store: store)
                    }
                }
                .padding(.horizontal, NDGrid.pageSide)
                .padding(.top, NDGrid.pageTop)
                .padding(.bottom, 24)
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .scrollIndicators(.automatic)
        }
        .frame(minWidth: NDGrid.windowMinimum.width, minHeight: NDGrid.windowMinimum.height)
        .background(ND.night)
        .foregroundStyle(ND.ink)
        .environment(\.ndPalette, .window)
    }
}

// MARK: - Title bar

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
                    .padding(.leading, 18)
                }
                Spacer()
                if let memories = store.model.memories {
                    (Text(memories).foregroundColor(ND.ink).fontWeight(.medium) + Text(" memories").foregroundColor(ND.ink2))
                        .font(.system(size: 12).monospacedDigit())
                        .padding(.trailing, 18)
                }
            }
            NDSegmentedPages(store: store).fixedSize()
        }
        .frame(height: NDGrid.titleBar)
        .background(ND.deck)
        .overlay(alignment: .bottom) { ND.rule.frame(height: 1) }
    }
}

/// The native segmented control with its labels drawn by us: Ink 2 when not selected, Ink when selected (AppKit would
/// draw both in the label colour), and a page with faults carries a small red count after its label. The control, its
/// selected fill and its keyboard and VoiceOver behaviour stay native.
struct NDSegments: NSViewRepresentable {
    let titles: [String]
    var counts: [Int] = []
    @Binding var selection: Int
    var accessibilityLabel: String
    /// The title bar's segments are 13 pt with 14 pt each side; Settings' Classic / New are 12 pt with 12 pt (comps 04, 06).
    var fontSize: CGFloat = 13
    var padding: CGFloat = 14

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
        control.controlSize = .regular
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


    /// "Backups 1": the label in Ink (selected) or Ink 2, the count in Fault, Bold 11, 5 pt after it. Drawn at display
    /// time, so the colours follow the control's appearance.
    @MainActor static func labelImage(_ title: String, count: Int, selected: Bool, size: CGFloat = 13) -> NSImage {
        let font = NSFont.systemFont(ofSize: size, weight: .medium)
        let label = NSAttributedString(string: title, attributes: [.font: font])
        let countFont = NSFont.monospacedDigitSystemFont(ofSize: 11, weight: .bold)
        let numeral = NSAttributedString(string: String(count), attributes: [.font: countFont])
        let width = ceil(label.size().width + (count > 0 ? 5 + numeral.size().width : 0))
        let image = NSImage(size: NSSize(width: width, height: ceil(label.size().height)), flipped: false) { rect in
            NSAttributedString(string: title, attributes: [.font: font, .foregroundColor: selected ? NDColor.ink : NDColor.ink2])
                .draw(at: .zero)
            if count > 0 {
                let y = (rect.height - numeral.size().height) / 2
                NSAttributedString(string: String(count), attributes: [.font: countFont, .foregroundColor: NDColor.fault])
                    .draw(at: NSPoint(x: label.size().width + 5, y: y))
            }
            return true
        }
        image.isTemplate = false
        image.accessibilityDescription = count > 0 ? "\(title), \(count == 1 ? "1 thing needs you" : "\(count) things need you")" : title
        return image
    }
}

struct NDSegmentedPages: View {
    @ObservedObject var store: NDStore
    var body: some View {
        NDSegments(titles: NDPage.allCases.map(\.title),
                   counts: NDPage.allCases.map { [.backups: store.model.backupsTabCount, .jobs: store.model.jobsTabCount][$0] ?? 0 },
                   selection: Binding(get: { NDPage.allCases.firstIndex(of: store.page) ?? 0 },
                                      set: { store.page = NDPage.allCases[max(0, min($0, NDPage.allCases.count - 1))] }),
                   accessibilityLabel: "Pages")
    }
}

// MARK: - Overview

struct NDOverviewPage: View {
    @ObservedObject var store: NDStore
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let m = store.model
        VStack(alignment: .leading, spacing: 0) {
            NDPageTop(title: m.headline, mark: m.headlineMark, checked: m.checked, large: true)
            if let detail = m.headlineDetail {
                Text(detail).font(NDType.fact).foregroundStyle(ND.ink2).padding(.top, 2)
            }
            if !m.faults.isEmpty {
                NDNeedsBand(store: store)
                    .padding(.top, 12)
                    .transition(reduceMotion ? .opacity : .opacity.combined(with: .move(edge: .top)))
            }
            NDHero(store: store, height: m.faults.isEmpty ? 104 : m.faults.count == 1 ? 80 : 48).padding(.top, 14)
            NDPanel(model: m).padding(.top, 16)
        }
        .animation(reduceMotion ? .easeOut(duration: 0.15) : NDMotion.needsBand, value: m.faults.map(\.title))
    }
}

/// The page's first line: the headline (or page title) with its mark hanging 20 pt into the margin, and when it was checked.
struct NDPageTop: View {
    let title: String
    let mark: NDModel.Mark
    let checked: String?
    var large = false

    var body: some View {
        HStack(alignment: .center) {
            Text(title)
                .font(large ? NDType.headline : NDType.pageTitle)
                .tracking(large ? -0.4 : 0)
                .overlay(alignment: .leading) { NDMarkView(mark: mark).offset(x: -20) }
            Spacer(minLength: 16)
            if let checked { Text(checked).font(NDType.caption).foregroundStyle(ND.ink2) }
        }
        .frame(minHeight: 34)
        .accessibilityElement(children: .combine)
    }
}

struct NDNeedsBand: View {
    @ObservedObject var store: NDStore
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 0) {
            ForEach(Array(store.model.faults.enumerated()), id: \.element.id) { index, fault in
                if index > 0 { ND.rule.frame(height: 1) }
                row(fault)
            }
        }
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(ND.deck))
    }

    private func row(_ fault: NDModel.Fault) -> some View {
        let waiting = store.inputs.driveReconnecting && [.reconnectDrive, .connectDrive].contains(fault.action?.kind)
        return HStack(alignment: .center, spacing: 12) {
            NDMarkView(mark: .fault)
            VStack(alignment: .leading, spacing: 2) {
                Text(keepingUnits(fault.title)).font(NDType.faultTitle)
                if !fault.shownFacts.isEmpty {
                    Text(keepingUnits(fault.shownFacts.joined(separator: " "))).font(NDType.fact).foregroundStyle(ND.ink2)
                        .lineSpacing(2).fixedSize(horizontal: false, vertical: true)
                }
                if fault.action?.kind == .copyReport, store.inputs.copyReportFailed {
                    Text(BrainBarMenuPresenter.SurfaceSay.copyFailed).font(NDType.fact).foregroundStyle(ND.ink2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if let action = fault.action {
                ZStack(alignment: .trailing) {
                    NDActionButton(label: action.label, large: true) { store.perform(action.kind) }.opacity(waiting ? 0 : 1)
                    HStack(spacing: 7) {
                        NDMarkView(mark: .running)
                        Text(BrainBarMenuPresenter.SurfaceSay.waitingForBrowser).font(NDType.fact).foregroundStyle(ND.ink2)
                    }
                    .opacity(waiting ? 1 : 0)
                }
                .animation(reduceMotion ? nil : NDMotion.waitingFade, value: waiting)
            }
        }
        .padding(.vertical, 9)
        .padding(.horizontal, 16)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(([fault.title] + fault.facts).joined(separator: " "))
    }
}

// MARK: Hero waveform

struct NDHero: View {
    @ObservedObject var store: NDStore
    /// 104 pt calm; 80 pt with one fault; 48 pt with two or more (the needs band takes the room).
    let height: CGFloat
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var shift: CGFloat = 0
    @State private var lastBucket: Int?

    var body: some View {
        let wave = store.inputs.heroWaves[store.range]
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .center, spacing: 0) {
                if let wave, wave.readable {
                    total(wave.savesTotal, "saved by agents").accessibilityHidden(true)
                    total(wave.readsTotal, "read from conversations").padding(.leading, 14).accessibilityHidden(true)
                } else {
                    Text("—").font(.system(size: 13, weight: .semibold)).foregroundStyle(ND.ink2)
                }
                Spacer()
                NDRangeControl(store: store)
            }
            .padding(.bottom, 8)
            // N03: the waveform and its totals are one element with one sentence; the range control stays three buttons.
            GeometryReader { geo in
                NDWaveform(wave: wave, size: .hero, nowTick: true)
                    .offset(x: shift)
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()
                    .id(store.range)
                    .transition(.opacity)
            }
            .frame(height: height + 10)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(NDModel.waveAccessibility(wave, minutes: store.range.minutes))
            .animation(reduceMotion ? nil : NDMotion.rangeFade, value: store.range)
            .animation(reduceMotion ? nil : NDMotion.needsBand, value: height)
            let axis = store.range.axis
            HStack {
                Text(axis.0)
                Spacer()
                Text(axis.1)
                Spacer()
                Text(axis.2)
            }
            .font(NDType.axis).foregroundStyle(ND.ink2)
            .padding(.top, 0)
            .accessibilityHidden(true)
        }
        .padding(EdgeInsets(top: 12, leading: 16, bottom: 10, trailing: 16))
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(ND.deck))
        .accessibilityElement(children: .contain)
        .onChange(of: wave?.endsAt) { _, end in shiftIfNewBucket(end) }
    }

    private func total(_ value: Int, _ label: String) -> some View {
        (Text(NDFormat.integer(value)).font(.system(size: 13, weight: .semibold).monospacedDigit()).foregroundColor(ND.ink)
            + Text(" " + label).font(.system(size: 12)).foregroundColor(ND.ink2))
    }

    /// Motion 5: when the newest bucket is a new one, the bars slide one pitch left over 240 ms (a retargetable
    /// transition on the offset). Reduce Motion swaps them in place.
    private func shiftIfNewBucket(_ end: Date?) {
        guard let end else { return }
        let width = Double(store.range.minutes * 60) / 60
        let bucket = Int(end.timeIntervalSinceReferenceDate / width)
        defer { lastBucket = bucket }
        guard let lastBucket, bucket == lastBucket + 1, !reduceMotion else { return }
        shift = 13
        withAnimation(NDMotion.shift) { shift = 0 }
    }
}

struct NDRangeControl: View {
    @ObservedObject var store: NDStore
    var body: some View {
        HStack(spacing: 0) {
            ForEach(NDRange.allCases) { range in
                Button { store.range = range } label: {
                    Text(range.label).font(.system(size: 11, weight: .semibold).monospacedDigit())
                        .foregroundStyle(store.range == range ? ND.ink : ND.ink2)
                        .padding(.horizontal, 9).padding(.vertical, 2)
                        .background(RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .fill(store.range == range ? ND.chip : .clear))
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(store.range == range ? .isSelected : [])
            }
        }
        .padding(2)
        .background(RoundedRectangle(cornerRadius: 6, style: .continuous).fill(ND.rule))
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Range")
    }
}

// MARK: The instrument panel

struct NDPanel: View {
    let model: NDModel
    var body: some View {
        let left = model.areas.filter { ["Remembering", "Finding", "Agents"].contains($0.name) }
        let right = model.areas.filter { ["Backups", "Upkeep"].contains($0.name) }
        HStack(alignment: .top, spacing: NDGrid.columnGutter) {
            column(left)
            column(right)
        }
    }

    private func column(_ areas: [NDModel.Area]) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(areas) { NDAreaView(area: $0) }
        }
        .frame(maxWidth: .infinity, alignment: .topLeading)
    }
}

struct NDAreaView: View {
    let area: NDModel.Area

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(area.name).font(NDType.areaName).tracking(0.2)
                    .foregroundStyle(area.fault ? ND.fault : ND.ink)
                    .overlay(alignment: .leading) { NDMarkView(mark: area.mark).offset(x: -20) }
                Spacer()
                if let aside = area.aside { Text(aside).font(NDType.axis).foregroundStyle(ND.ink2) }
            }
            .frame(height: 20)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(area.name + (area.fault ? ", needs you" : area.mark == .unknown ? ", " + (area.unknownReason ?? "not confirmed") : ""))
            .help(area.unknownReason ?? "")
            .accessibilityAddTraits(.isHeader)
            VStack(spacing: 0) {
                ForEach(area.rows) { NDPanelRow(row: $0) }
            }
            // Under the rows: a state word's Night plate (6 pt leading) masks the line where the word crosses it.
            .background(alignment: .topLeading) {
                if area.limitLine {
                    GeometryReader { geo in
                        NDLimitLine()
                            .frame(width: 7, height: max(0, geo.size.height - 6))
                            .position(x: geo.size.width - NDGrid.rowNumber - NDGrid.rowLane, y: geo.size.height / 2)
                    }
                    .allowsHitTesting(false)
                }
            }
        }
    }
}

/// One area's limit: a 1 pt Track line through every row's limit, capped 3 pt at both ends (N1).
struct NDLimitLine: View {
    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                ctx.fill(Path(CGRect(x: 3, y: 0, width: 1, height: size.height)), with: .color(ND.track))
                ctx.fill(Path(CGRect(x: 0, y: 0, width: 7, height: 1)), with: .color(ND.track))
                ctx.fill(Path(CGRect(x: 0, y: size.height - 1, width: 7, height: 1)), with: .color(ND.track))
            }
            .frame(width: 7, height: geo.size.height)
        }
        .accessibilityHidden(true)
    }
}

struct NDPanelRow: View {
    let row: NDModel.Row

    var body: some View {
        HStack(spacing: 0) {
            // A name never truncates: one a few points over the 146 pt column takes them from the gauge, whose end (the
            // limit line) stays put.
            name.fixedSize().frame(minWidth: NDGrid.rowName, alignment: .leading)
            Spacer().frame(width: NDGrid.rowGap)
            if row.stateWord {
                Spacer(minLength: 0)
                NDNumberText(number: row.number)
                    .padding(.leading, 6)
                    .background(ND.night)  // masks the limit line where the word crosses it
            } else {
                visual.frame(maxWidth: .infinity)
                lane.frame(width: NDGrid.rowLane, alignment: .center)
                NDNumberText(number: row.number).frame(width: NDGrid.rowNumber, alignment: .trailing)
            }
        }
        .frame(height: NDGrid.rowPitch)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(row.accessibility)
    }

    @ViewBuilder private var name: some View {
        if case let .agent(sessions, _, _) = row.visual {
            HStack(spacing: 8) {
                Text(row.name).font(NDType.row)
                NDSessionDots(sessions: sessions)
            }
        } else {
            Text(row.name).font(row.strong ? NDType.rowStrong : NDType.row).lineLimit(1)
        }
    }

    @ViewBuilder private var visual: some View {
        switch row.visual {
        case .gauge, .emptyTrack, .unknown, .streak: NDGaugeView(visual: row.visual)
        case let .fill(share): NDFillBar(share: share)
        case let .agent(_, saves, scale): NDSavesBar(saves: saves, scale: scale)
        case .none: Color.clear
        }
    }

    @ViewBuilder private var lane: some View {
        if case let .streak(n) = row.visual { NDStreakDots(count: n).padding(.leading, 4) } else { Color.clear }
    }
}
