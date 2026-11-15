import AppKit
import SwiftUI

/// The new dropdown (DESIGN-SPEC § 6.2): custom views in a real `NSMenu`, on the system menu material. The headline, the
/// fault blocks with one action each, "Everything else is fine.", the four area rows, then the standing items on the
/// system edge (R4). Every row shares one grid, `12 mark | 8 | content`, so the headline's text and the area names start
/// at the same x in every state (review finding 5). Opening and closing are the system's own, never animated.
@MainActor final class NDMenuBuilder: NSObject, NSMenuDelegate {
    let store: NDStore
    let perform: (BrainBarMenuAction) -> Void
    let openWindow: (NDPage) -> Void
    private var rowStates: [ObjectIdentifier: NDMenuRowState] = [:]
    /// Ends the menu's tracking before a row's click opens the window.
    var menuDismiss: (() -> Void)?

    init(store: NDStore, perform: @escaping (BrainBarMenuAction) -> Void, openWindow: @escaping (NDPage) -> Void) {
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
        let model = store.model
        menu.addItem(hosted(NDMenuHeadline(store: store), label: [model.headline, model.headlineDetail].compactMap { $0 }.joined(separator: ". ")))
        menu.addItem(.separator())
        for (index, fault) in model.faults.enumerated() {
            if index > 0 { menu.addItem(.separator()) }
            let state = NDMenuRowState()
            let item = hosted(NDMenuFaultRow(store: store, fault: fault, state: state) { [weak self] kind in self?.perform(.row(kind)) },
                              label: ([fault.title] + fault.facts).joined(separator: " "), state: state)
            if let action = fault.action {
                item.title = action.label  // type-select and VoiceOver reach the action; the view draws the row
                item.target = self
                item.action = #selector(performItem)
                item.representedObject = BrainBarMenuAction.row(action.kind)
                item.isEnabled = true
            }
            menu.addItem(item)
        }
        if model.everythingElse {
            menu.addItem(hosted(NDMenuEverythingElse(), label: BrainBarMenuPresenter.Say.everythingElse.text))
        }
        if !model.faults.isEmpty { menu.addItem(.separator()) }
        for area in model.menuAreas {
            let state = NDMenuRowState()
            let item = hosted(NDMenuAreaRow(store: store, area: area, state: state) { [weak self] in self?.openOverview() },
                              label: "\(area.name), \(area.accessibility)" + (area.unknownReason.map { ", " + $0 } ?? ""), state: state)
            item.title = area.name
            item.toolTip = area.unknownReason
            item.target = self
            item.action = #selector(openArea)
            item.isEnabled = true
            menu.addItem(item)
        }
        menu.addItem(.separator())
        let settings = standard(BrainBarMenuPresenter.Say.settings.text, key: ",", action: .settings)
        let quit = standard(BrainBarMenuPresenter.Say.quit.text, key: "q", action: .quit)
        let restart = standard(BrainBarMenuPresenter.Say.restart.text, key: "q", action: .restart)
        restart.keyEquivalentModifierMask = [.command, .option]
        restart.isAlternate = true
        [settings, quit, restart].forEach(menu.addItem)
    }

    /// Arrow keys and the pointer reach the area rows and the fault rows; the row draws its own highlight.
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

    @objc private func openArea(_ sender: NSMenuItem) { openOverview() }

    private func openOverview() {
        menuDismiss?()
        openWindow(.overview)
    }

    private func hosted<V: View>(_ view: V, label: String, state: NDMenuRowState? = nil) -> NSMenuItem {
        let item = NSMenuItem()
        item.view = NDMenuRowHost(view.environment(\.ndPalette, .menu), label: label)
        item.isEnabled = false
        if let state { rowStates[ObjectIdentifier(item)] = state }
        return item
    }

    private func standard(_ title: String, key: String, action: BrainBarMenuAction) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: #selector(performItem), keyEquivalent: key)
        item.keyEquivalentModifierMask = .command
        item.target = self
        item.representedObject = action
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

/// `12 mark | 8 | content`, inside the menu's 5 pt selection margin and 9 pt padding: content starts at x = 34.
struct NDMenuGrid<Mark: View, Content: View>: View {
    var alignment: VerticalAlignment = .center
    @ViewBuilder var mark: Mark
    @ViewBuilder var content: Content
    var body: some View {
        HStack(alignment: alignment, spacing: NDGrid.menuMarkGap) {
            mark.frame(width: NDGrid.menuMark)
            content.frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 9)
        .padding(.horizontal, 5)
    }
}

struct NDMenuHeadline: View {
    @ObservedObject var store: NDStore
    var body: some View {
        let m = store.model
        VStack(alignment: .leading, spacing: 0) {
            NDMenuGrid {
                NDMarkView(mark: m.headlineMark)
            } content: {
                Text(m.headline).font(NDType.menuHeadline).tracking(-0.1).foregroundStyle(Color(nsColor: .labelColor))
            }
            .frame(minHeight: 34)
            if let detail = m.headlineDetail {
                NDMenuGrid { Color.clear } content: {
                    Text(detail).font(NDType.menuFact).foregroundStyle(Color(nsColor: .secondaryLabelColor))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(.bottom, 6)
            }
        }
    }
}

struct NDMenuFaultRow: View {
    @ObservedObject var store: NDStore
    let fault: NDModel.Fault
    @ObservedObject var state: NDMenuRowState
    let perform: (BrainBarMenuPresentation.ActionKind) -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var waiting: Bool {
        store.inputs.driveReconnecting && [.reconnectDrive, .connectDrive].contains(fault.action?.kind)
    }

    var body: some View {
        NDMenuGrid(alignment: .top) {
            NDMarkView(mark: .fault).padding(.top, 2.5)
        } content: {
            VStack(alignment: .leading, spacing: 0) {
                Text(keepingUnits(fault.title)).font(NDType.menuFaultTitle).foregroundStyle(Color(nsColor: .labelColor))
                    .lineSpacing(2).fixedSize(horizontal: false, vertical: true)
                if !fault.shownFacts.isEmpty {
                    Text(keepingUnits(fault.shownFacts.joined(separator: " "))).font(NDType.menuFact)
                        .foregroundStyle(Color(nsColor: .secondaryLabelColor)).lineSpacing(2)
                        .fixedSize(horizontal: false, vertical: true).padding(.top, 2)
                }
                if let action = fault.action {
                    ZStack(alignment: .leading) {
                        // The waiting line takes the button's place at the button's height (R3): nothing below moves.
                        NDActionButton(label: action.label, keyboardFocus: state.keyboard) { perform(action.kind) }
                            .opacity(waiting ? 0 : 1)
                        HStack(spacing: 7) {
                            NDMarkView(mark: .running)
                            Text("Waiting for you to approve in your browser…").font(NDType.menuFact)
                                .foregroundStyle(Color(nsColor: .secondaryLabelColor))
                        }
                        .opacity(waiting ? 1 : 0)
                    }
                    .animation(reduceMotion ? nil : NDMotion.waitingFade, value: waiting)
                    .padding(.top, 8)
                    if action.kind == .copyReport, store.inputs.copyReportFailed {
                        Text("Couldn’t copy the report. Try again.").font(NDType.menuFact)
                            .foregroundStyle(Color(nsColor: .secondaryLabelColor)).padding(.top, 4)
                    }
                }
            }
        }
        .padding(.top, 7)
        .padding(.bottom, 9)
    }
}

/// The one action: a filled button in Lumen. It answers a press (scale 0.97, motion 9) and shows focus when the arrow
/// keys reach its row.
struct NDActionButton: View {
    let label: String
    var keyboardFocus = false
    var large = false
    let action: () -> Void

    var body: some View {
        Button(action: action) { Text(label) }
            .buttonStyle(NDFilledButtonStyle(large: large))
            .overlay(RoundedRectangle(cornerRadius: large ? 9 : 8, style: .continuous)
                .strokeBorder(Color(nsColor: .keyboardFocusIndicatorColor), lineWidth: 3).padding(-3)
                .opacity(keyboardFocus ? 1 : 0))
            .accessibilityLabel(Text(label))
    }
}

/// Motion 9: a press scales to 0.97 over 100 ms and releases over 160 ms. The spec keeps it under Reduce Motion (§ 8, row
/// 9: it is feedback at the pointer, not travel), so the style does not read the setting.
struct NDFilledButtonStyle: ButtonStyle {
    var large = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: large ? 13 : 12, weight: .semibold))
            .foregroundStyle(ND.onLumen)
            .padding(.horizontal, large ? 14 : 11)
            .frame(height: large ? 28 : 24)
            .background(RoundedRectangle(cornerRadius: large ? 7 : 6, style: .continuous).fill(ND.lumen))
            .contentShape(RoundedRectangle(cornerRadius: large ? 7 : 6, style: .continuous))
            .scaleEffect(configuration.isPressed ? 0.97 : 1)
            .animation(.easeOut(duration: configuration.isPressed ? 0.1 : 0.16), value: configuration.isPressed)
    }
}

struct NDMenuEverythingElse: View {
    var body: some View {
        NDMenuGrid {
            NDMarkView(mark: .check)
        } content: {
            Text(BrainBarMenuPresenter.Say.everythingElse.text).font(NDType.menuFact)
                .foregroundStyle(Color(nsColor: .secondaryLabelColor))
        }
        .padding(.vertical, 3)
        .padding(.bottom, 4)
    }
}

struct NDMenuAreaRow: View {
    @ObservedObject var store: NDStore
    let area: NDModel.MenuArea
    @ObservedObject var state: NDMenuRowState
    let open: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            NDMarkView(mark: area.mark).frame(width: NDGrid.menuMark)
            Spacer().frame(width: NDGrid.menuMarkGap)
            Text(area.name)
                .font(area.mark == .fault ? NDType.menuAreaNameStrong : NDType.menuAreaName)
                .tracking(0.15)
                .foregroundStyle(Color(nsColor: .labelColor))
                .frame(width: NDGrid.menuName, alignment: .leading)
            visual.frame(width: NDGrid.menuVisual, height: 20)
            NDNumberText(number: area.number, size: 12).frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding(.horizontal, 9)
        .frame(height: NDGrid.menuAreaRow)
        .background(RoundedRectangle(cornerRadius: 5, style: .continuous).fill(ND.menuHighlight).opacity(state.highlighted ? 1 : 0))
        .padding(.horizontal, 5)
        .contentShape(Rectangle())
        .onTapGesture(perform: open)
    }

    @ViewBuilder private var visual: some View {
        switch area.visual {
        case .wave: NDWaveform(wave: store.inputs.menuWave, size: .menu)
        case let .fill(share): NDFillBar(share: share)
        case let .gauge(g): NDGaugeView(visual: .gauge(g), endTick: true)
        case let .streak(n):
            HStack(spacing: 6) {
                NDGaugeView(visual: .streak(n))
                NDStreakDots(count: n)
            }
        case .unknown: NDDashed().frame(height: 12)
        }
    }
}

/// R28a, for line breaking only: a number stays with the word after it.
func keepingUnits(_ string: String) -> String {
    string.replacingOccurrences(of: "to an agent.", with: "to\u{00A0}an\u{00A0}agent.").replacing(#/(\d%?) /#) { "\($0.1)\u{00A0}" }.replacing(#/\b([A-Z][a-z]{2}) (\d)/#) { "\($0.1)\u{00A0}\($0.2)" }
}
