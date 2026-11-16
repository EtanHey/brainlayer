import AppKit
import SwiftUI

private typealias Say = BrainBarMenuPresenter.SurfaceSay

/// The dropdown (DESIGN-SPEC-R1 § 2.2): a real `NSMenu`, 320 pt. The headline; under a fault, a block with its title, the
/// broken pipe cut to its two stages, and its one button (the first filled, others bordered); the four area rows, each
/// with its symbol and one value, each opening its page; then Open BrainBar, Settings… and Quit as system items.
/// Opening and closing are the system's own, never animated.
@MainActor final class NDMenuBuilder: NSObject, NSMenuDelegate {
    let store: NDStore
    let perform: @MainActor (BrainBarMenuAction) -> Void
    let openWindow: @MainActor (NDPage) -> Void
    private var rowStates: [ObjectIdentifier: NDMenuRowState] = [:]
    /// Ends the menu's tracking before a row's click opens the window.
    var menuDismiss: (@MainActor () -> Void)?

    init(store: NDStore, perform: @escaping @MainActor (BrainBarMenuAction) -> Void,
         openWindow: @escaping @MainActor (NDPage) -> Void) {
        self.store = store
        self.perform = perform
        self.openWindow = openWindow
    }

    func build() -> NSMenu {
        let menu = NSMenu()
        menu.autoenablesItems = false
        menu.delegate = self
        populate(menu)
        return menu
    }

    func populate(_ menu: NSMenu) {
        menu.removeAllItems()
        rowStates = [:]
        let r1 = store.r1
        let headline = NDMenuHeadline.text(r1)
        let hadRed = store.inputs.capture.verdict?.checks.contains { $0.state == .red } == true
        menu.addItem(hosted(NDMenuHeadline(r1: r1, lastVerdictHadRed: hadRed), label: headline))
        menu.addItem(.separator())
        // R1.2 change 9 and L24: blocks when something is red or two or more amber notices; a single amber state says
        // itself in the headline and its line.
        let blocks = r1.showsProblemRows ? r1.items : []
        let filled = NDMenuFaultBlock.filledIndex(blocks)
        for (index, item) in blocks.enumerated() {
            if index > 0 { menu.addItem(.separator()) }
            let state = NDMenuRowState()
            let row = hosted(NDMenuFaultBlock(store: store, item: item, filled: index == filled, state: state) { [weak self] kind in
                self?.perform(.row(kind))
            }, label: [item.title, item.detail].compactMap { $0 }.joined(separator: ". "), state: state)
            if let action = item.action {
                row.title = action.label  // type-select and VoiceOver reach the action; the view draws the block
                row.target = self
                row.action = #selector(performItem)
                row.representedObject = BrainBarMenuAction.row(action.kind)
                row.isEnabled = true
            }
            menu.addItem(row)
        }
        if !blocks.isEmpty { menu.addItem(.separator()) }
        // R1.2 change 12: not confirmed: "Last known at 14:02" once, above the rows that show their last-known values.
        if let at = r1.pipeline.lastKnownAt {
            menu.addItem(hosted(NDMenuCaption(text: Say.lastKnownAt(at)), label: Say.lastKnownAt(at)))
        }
        for row in r1.menuRows {
            let state = NDMenuRowState()
            let item = hosted(NDMenuAreaRow(store: store, row: row, state: state) { [weak self] in self?.open(row.kind) },
                              label: [row.label, row.value, row.help].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: ", "),
                              state: state)
            item.title = row.label
            item.toolTip = row.help
            item.target = self
            item.action = #selector(openRow)
            item.representedObject = row.kind.rawValue
            item.isEnabled = true
            menu.addItem(item)
        }
        menu.addItem(.separator())
        let open = NSMenuItem(title: Say.openBrainBar, action: #selector(openOverview), keyEquivalent: "")
        open.target = self
        open.image = Self.symbol("macwindow")
        open.isEnabled = true
        let settings = standard(BrainBarMenuPresenter.Say.settings.text, key: ",", action: .settings, symbol: "gearshape")
        let quit = standard(BrainBarMenuPresenter.Say.quit.text, key: "q", action: .quit, symbol: "power")
        let restart = standard(BrainBarMenuPresenter.Say.restart.text, key: "q", action: .restart, symbol: "arrow.clockwise")
        restart.keyEquivalentModifierMask = [.command, .option]
        restart.isAlternate = true
        [open, settings, quit, restart].forEach(menu.addItem)
    }

    static func symbol(_ name: String) -> NSImage? {
        NSImage(systemSymbolName: name, accessibilityDescription: nil)?
            .withSymbolConfiguration(NSImage.SymbolConfiguration(pointSize: 13, weight: .regular))
    }

    /// The pointer and the arrow keys reach the area rows and the fault blocks; the row draws the system selection.
    func menu(_ menu: NSMenu, willHighlight item: NSMenuItem?) {
        let byKeyboard = NSApp.currentEvent?.type == .keyDown
        for (id, state) in rowStates {
            let on = item.map(ObjectIdentifier.init) == id
            state.highlighted = on
            state.keyboard = on && byKeyboard
        }
    }

    func rowState(of item: NSMenuItem) -> NDMenuRowState? { rowStates[ObjectIdentifier(item)] }

    @objc private func performItem(_ sender: NSMenuItem) {
        if let action = sender.representedObject as? BrainBarMenuAction { perform(action) }
    }

    @objc private func openRow(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String, let kind = NDR1.MenuRow.Kind(rawValue: raw) else { return }
        open(kind)
    }

    @objc private func openOverview() { open(.activity) }

    /// Each row opens the window at its page (Activity and Search at the Overview).
    private func open(_ kind: NDR1.MenuRow.Kind) {
        menuDismiss?()
        switch kind {
        case .activity, .search: openWindow(.overview)
        case .backups: openWindow(.backups)
        case .jobs: openWindow(.jobs)
        }
    }

    private func hosted<V: View>(_ view: V, label: String, state: NDMenuRowState? = nil) -> NSMenuItem {
        let item = NSMenuItem()
        item.view = NDMenuRowHost(view.environment(\.ndPalette, .menu), label: label)
        item.isEnabled = false
        if let state { rowStates[ObjectIdentifier(item)] = state }
        return item
    }

    private func standard(_ title: String, key: String, action: BrainBarMenuAction, symbol: String) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: #selector(performItem), keyEquivalent: key)
        item.keyEquivalentModifierMask = .command
        item.target = self
        item.representedObject = action
        item.image = Self.symbol(symbol)
        item.isEnabled = true
        return item
    }
}

/// The highlight a hosted row draws, set by the menu as the pointer or the arrow keys reach it.
@MainActor final class NDMenuRowState: ObservableObject {
    @Published var highlighted = false
    @Published var keyboard = false
}

/// A hosted row: sized to its view at the menu's width, one accessibility element that reads the row's words.
final class NDMenuRowHost: NSView {
    init<V: View>(_ view: V, label: String) {
        let hosting = NSHostingView(rootView: view.frame(width: NDGrid.menuWidth, alignment: .leading))
        let height = ceil(hosting.fittingSize.height)
        super.init(frame: NSRect(x: 0, y: 0, width: NDGrid.menuWidth, height: height))
        hosting.frame = bounds
        hosting.autoresizingMask = [.width, .height]
        addSubview(hosting)
        setAccessibilityElement(true)
        setAccessibilityRole(.group)
        setAccessibilityLabel(label)
    }

    @available(*, unavailable) required init?(coder: NSCoder) { nil }
}

// MARK: - Rows

/// The headline: "All good" (17 Semibold) alone; with faults one red mark and "N things need you"; stale, the question
/// mark and "Not confirmed". 19 pt from the menu's edge, as the system rows' text.
struct NDMenuHeadline: View {
    let r1: NDR1

    /// D6: "needs you" and its count are for red items only; amber alone says what is late, as the window does.
    static func text(_ r1: NDR1) -> String {
        let red = r1.redCount
        if red == 0 { return r1.headline }
        return red == 1 ? BrainBarMenuPresenter.Say.countOne.text : Say.thingsNeedYou(red)
    }

    /// The one line under a headline: an amber state's own line; not confirmed, "Nothing is known to be broken." only when
    /// the last confirmed verdict had no red check (L26).
    static func detail(_ r1: NDR1, lastVerdictHadRed: Bool) -> String? {
        if r1.pipeline.allUnconfirmed { return lastVerdictHadRed ? nil : Say.nothingKnownBrokenShort }
        return r1.redCount == 0 ? r1.headlineDetail : nil
    }

    var lastVerdictHadRed = false

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            if !r1.items.isEmpty {
                let red = r1.redCount > 0
                Image(systemName: red ? "exclamationmark.circle.fill" : "exclamationmark.triangle.fill")
                    .font(.system(size: 15)).foregroundStyle(red ? ND.red : ND.orange)
            } else if let symbol = r1.headlineSymbol {
                Image(systemName: symbol).font(.system(size: 15)).foregroundStyle(Color(nsColor: .secondaryLabelColor))
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(Self.text(r1)).font(NDType.title).tracking(-0.25).foregroundStyle(Color(nsColor: .labelColor))
                if let line = Self.detail(r1, lastVerdictHadRed: lastVerdictHadRed) {
                    Text(keepingUnits(line)).font(NDType.body).foregroundStyle(Color(nsColor: .secondaryLabelColor))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.top, 7).padding(.bottom, 6)
        .padding(.horizontal, NDGrid.menuInset)
        .frame(minHeight: 36)
    }
}

/// "Last known at 14:02" (11 `t3`), once above the rows.
struct NDMenuCaption: View {
    let text: String
    var body: some View {
        Text(text).font(NDType.caption).foregroundStyle(Color(nsColor: .tertiaryLabelColor))
            .padding(.horizontal, NDGrid.menuInset).padding(.top, 4).padding(.bottom, 2)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// One fault: its title (13 Semibold), its one line, and its one button.
struct NDMenuFaultBlock: View {
    @ObservedObject var store: NDStore
    let item: NDR1.Item
    let filled: Bool
    @ObservedObject var state: NDMenuRowState
    let perform: (BrainBarMenuPresentation.ActionKind) -> Void

    /// One filled button per surface: a reconnect when there is one, otherwise the first action.
    static func filledIndex(_ items: [NDR1.Item]) -> Int {
        items.firstIndex { $0.action.map { [.reconnectDrive, .connectDrive].contains($0.kind) } == true }
            ?? items.firstIndex { $0.action != nil } ?? 0
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(item.menuTitle ?? item.title).font(NDType.bodyStrong).foregroundStyle(Color(nsColor: .labelColor))
                .fixedSize(horizontal: false, vertical: true)
            // R1.1 change 28: one line of words under the title (the strips are gone).
            if let detail = item.menuDetail ?? item.detail, !detail.isEmpty {
                Text(keepingUnits(detail)).font(NDType.body).foregroundStyle(Color(nsColor: .secondaryLabelColor))
                    .fixedSize(horizontal: false, vertical: true).padding(.top, 2)
            }
            if let action = item.action {
                let waiting = store.inputs.driveReconnecting && [.reconnectDrive, .connectDrive].contains(action.kind)
                NDWaitingSwap(waiting: waiting, alignment: .leading) {
                    NDButton(label: action.label, filled: filled, small: true) { perform(action.kind) }
                        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .strokeBorder(Color(nsColor: .keyboardFocusIndicatorColor), lineWidth: 3).padding(-3)
                            .opacity(state.keyboard ? 1 : 0))
                } line: {
                    Text(Say.waitingForBrowser).font(NDType.body).foregroundStyle(Color(nsColor: .secondaryLabelColor))
                }
                .padding(.top, 9)
            }
        }
        .padding(.top, 6).padding(.bottom, 8)
        .padding(.horizontal, NDGrid.menuInset)
    }
}

/// The broken pipe at 320 pt (§ 3.2): the two stages either side of the break, the pipe between them, and the same red or
/// orange stretch and mark at the same side; downstream dotted `quiet`. 250 × 14.
struct NDPipeStrip: View {
    let strip: NDR1.Strip
    static let width: CGFloat = 250

    var body: some View {
        Canvas { ctx, size in Self.draw(ctx, size, strip) }
            .frame(width: Self.width, height: 14)
            .accessibilityHidden(true)
    }

    static func labelWidth(_ s: String) -> CGFloat {
        ceil(NSAttributedString(string: s, attributes: [.font: NSFont.systemFont(ofSize: 11)]).size().width)
    }

    static func draw(_ ctx: GraphicsContext, _ size: CGSize, _ strip: NDR1.Strip) {
        let w = size.width, y: CGFloat = 7
        let x0 = labelWidth(strip.from) + 8, x1 = w - labelWidth(strip.to) - 8
        let tint = strip.severity == .red ? ND.red : ND.orange
        let solid = StrokeStyle(lineWidth: 1.6, lineCap: .round)
        func line(_ a: CGFloat, _ b: CGFloat) -> Path {
            var p = Path()
            p.move(to: CGPoint(x: a, y: y)); p.addLine(to: CGPoint(x: b, y: y))
            return p
        }
        let mark: CGFloat
        if strip.side == .target {
            ctx.stroke(line(x0, x1 - 30), with: .color(ND.pipe), style: solid)
            ctx.stroke(line(x1 - 30, x1 - 16), with: .color(tint), style: solid)
            mark = x1 - 7
        } else {
            ctx.stroke(line(x0 + 16, x0 + 26), with: .color(tint), style: solid)
            ctx.stroke(line(x0 + 30, x1), with: .color(ND.quiet), style: StrokeStyle(lineWidth: 1.6, dash: [2, 3]))
            mark = x0 + 7
        }
        NDPipelineChart.symbol(ctx, strip.severity == .red ? "exclamationmark.circle.fill" : "exclamationmark.triangle.fill",
                               at: CGPoint(x: mark, y: y), size: 12, color: tint)
        let secondary = Color(nsColor: .secondaryLabelColor)
        ctx.draw(Text(strip.from).font(.system(size: 11)).foregroundColor(secondary), at: CGPoint(x: 0, y: y), anchor: .leading)
        ctx.draw(Text(strip.to).font(.system(size: 11)).foregroundColor(secondary), at: CGPoint(x: w, y: y), anchor: .trailing)
    }
}

/// An area row (24 pt): `16 symbol | 8 | label | flexible | value`; the Activity row carries the 128 × 18 two-sided chart
/// instead of a value. The row under the pointer takes the system selection with white text.
struct NDMenuAreaRow: View {
    @ObservedObject var store: NDStore
    let row: NDR1.MenuRow
    @ObservedObject var state: NDMenuRowState
    let open: () -> Void

    static let symbols: [NDR1.MenuRow.Kind: String] = [.activity: "chart.bar", .search: "magnifyingglass", .backups: "externaldrive",
                                                       .jobs: "clock"]

    var body: some View {
        let selected = state.highlighted
        let text = Color(nsColor: selected ? .alternateSelectedControlTextColor : .labelColor)
        let secondary = Color(nsColor: selected ? .alternateSelectedControlTextColor : .secondaryLabelColor)
        HStack(spacing: 8) {
            Image(systemName: Self.symbols[row.kind] ?? "circle").font(.system(size: 13)).foregroundStyle(secondary)
                .frame(width: 16)
            Text(row.label).font(NDType.body).foregroundStyle(text)
            Spacer(minLength: 8)
            if row.kind == .activity {
                NDWaveform(wave: store.inputs.menuWave, size: .menu)
                    .frame(width: 128, height: 18)
                    .environment(\.ndPalette, selected ? .menuSelected : .menu)
            } else {
                Text(row.value)
                    .font(row.tone == .fine ? NDType.number : NDType.numberStrong)
                    .foregroundStyle(selected ? secondary : row.tone == .fault ? ND.redText : row.tone == .warn ? ND.orangeText : secondary)
            }
        }
        .padding(.horizontal, 14)
        .frame(height: NDGrid.menuAreaRow)
        .background { if selected { NDMenuSelection() } }
        .padding(.horizontal, 5)
        .contentShape(Rectangle())
        .onTapGesture(perform: open)
    }
}

/// Review polish 4: the button leaves, then the waiting line enters, in the same place and row height; Reduce Motion
/// swaps them at once.
struct NDWaitingSwap<Button: View, Line: View>: View {
    let waiting: Bool
    var alignment: Alignment = .trailing
    @ViewBuilder let button: Button
    @ViewBuilder let line: Line
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack(alignment: alignment) {
            button
                .environment(\.ndHeldPressed, waiting)
                .opacity(waiting ? 0 : 1)
                .scaleEffect(waiting ? 0.97 : 1)
                .animation(reduceMotion ? nil : NDMotion.buttonLeaves, value: waiting)
                .allowsHitTesting(!waiting)
                .accessibilityHidden(waiting)
            line
                .opacity(waiting ? 1 : 0)
                .animation(reduceMotion ? nil : waiting ? NDMotion.waitingEnters : NDMotion.buttonLeaves, value: waiting)
                .accessibilityHidden(!waiting)
        }
    }
}

private struct NDHeldPressedKey: EnvironmentKey { static let defaultValue = false }
extension EnvironmentValues {
    /// A button that is leaving because its action started keeps its pressed look (D8.2).
    var ndHeldPressed: Bool {
        get { self[NDHeldPressedKey.self] }
        set { self[NDHeldPressedKey.self] = newValue }
    }
}

/// The system menu's own selection: the `.selection` material, emphasized, 6 pt continuous corners.
struct NDMenuSelection: NSViewRepresentable {
    func makeNSView(context: Context) -> NSVisualEffectView {
        let view = NSVisualEffectView()
        view.material = .selection
        view.isEmphasized = true
        view.state = .active
        view.wantsLayer = true
        view.layer?.cornerRadius = 6
        view.layer?.cornerCurve = .continuous
        view.layer?.masksToBounds = true
        return view
    }

    func updateNSView(_ view: NSVisualEffectView, context: Context) {}
}

/// Fix 11: the report sentence is one unit, so neither "an agent." nor "to an agent." is left alone on a line.
func keepingUnits(_ string: String) -> String {
    let report = BrainBarMenuPresenter.Say.giveReport.text
    return string.replacingOccurrences(of: report, with: report.replacingOccurrences(of: " ", with: "\u{00A0}"))
        .replacing(#/(\d%?) /#) { "\($0.1)\u{00A0}" }.replacing(#/\b([A-Z][a-z]{2}) (\d)/#) { "\($0.1)\u{00A0}\($0.2)" }
}
