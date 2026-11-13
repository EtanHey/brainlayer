import AppKit
import SwiftUI

/// What a menu item asks for. The row actions are the presentation's own kinds; the standing items are Settings…,
/// Quit and its Option alternate.
enum BrainBarMenuAction: Equatable, Sendable {
    case row(BrainBarMenuPresentation.ActionKind)
    case settings, quit, restart
}

@MainActor protocol BrainBarMenuActions: AnyObject {
    func perform(_ action: BrainBarMenuAction)
}

/// Menu PR 1: nothing is wired. The default only records what was asked for.
@MainActor final class BrainBarMenuRecordedActions: BrainBarMenuActions {
    private(set) var calls: [BrainBarMenuAction] = []
    func perform(_ action: BrainBarMenuAction) { calls.append(action) }
}

/// Turns one `BrainBarMenuPresentation` into the items of a real `NSMenu`: standard items for the standing block (real
/// key equivalents, a real Option alternate), a real separator for every separator row and every divider, and a hosted
/// row for each row a standard item cannot draw. It renders the value and nothing else. The menu holds it weakly (as
/// its delegate and its items' target): whoever shows the menu keeps the builder.
@MainActor final class BrainBarMenuBuilder: NSObject, NSMenuDelegate {
    typealias P = BrainBarMenuPresentation
    let actions: BrainBarMenuActions
    let accessibility: BrainBarMenuAccessibility
    private var focus: [ObjectIdentifier: BrainBarMenuRowFocus] = [:]

    init(actions: BrainBarMenuActions = BrainBarMenuRecordedActions(), accessibility: BrainBarMenuAccessibility = .system) {
        self.actions = actions
        self.accessibility = accessibility
    }

    func build(_ presentation: P) -> NSMenu {
        let menu = NSMenu()
        menu.autoenablesItems = false
        menu.delegate = self
        populate(menu, with: presentation)
        return menu
    }

    func populate(_ menu: NSMenu, with presentation: P) {
        menu.removeAllItems()
        focus = [:]
        let rows = presentation.rows
        for (index, row) in rows.enumerated() {
            if row.dividerBefore { menu.addItem(.separator()) }
            switch row.kind {
            case .separator: menu.addItem(.separator())
            case .standardItem: standardItems(row).forEach(menu.addItem)
            default:
                let previous = index > 0 ? (row.dividerBefore ? .separator : rows[index - 1].kind) : nil
                let next = index + 1 < rows.count ? (rows[index + 1].dividerBefore ? .separator : rows[index + 1].kind) : nil
                menu.addItem(hostedItem(row, previous: previous, next: next))
            }
        }
    }

    /// The menu highlights a Needs-you row as the arrow keys reach it; only then does its button show focus.
    func menu(_ menu: NSMenu, willHighlight item: NSMenuItem?) {
        highlight(item, byKeyboard: NSApp.currentEvent?.type == .keyDown)
    }

    func highlight(_ item: NSMenuItem?, byKeyboard: Bool) {
        for (id, state) in focus { state.keyboardFocused = byKeyboard && item.map(ObjectIdentifier.init) == id }
    }

    func focusState(of item: NSMenuItem) -> BrainBarMenuRowFocus? { focus[ObjectIdentifier(item)] }

    @objc private func performItem(_ sender: NSMenuItem) {
        if let action = sender.representedObject as? BrainBarMenuAction { actions.perform(action) }
    }

    private func hostedItem(_ row: P.Row, previous: P.RowKind?, next: P.RowKind?) -> NSMenuItem {
        let item = NSMenuItem()
        let rowFocus = BrainBarMenuRowFocus()
        let view = BrainBarMenuRowView(row: row, previous: previous, next: next, accessibility: accessibility, focus: rowFocus) {
            [weak self] kind in self?.actions.perform(.row(kind))
        }
        item.view = BrainBarMenuRowHost(view, words: row.strings + [row.action?.label].compactMap { $0 }, isAction: row.action != nil)
        // Rows that are not actions are disabled: the keyboard skips them and they never highlight.
        if let action = row.action {
            item.title = action.label  // not drawn (the view draws the row); lets type-select reach the action
            item.target = self
            item.action = #selector(performItem)
            item.representedObject = BrainBarMenuAction.row(action.kind)
            focus[ObjectIdentifier(item)] = rowFocus
        } else {
            item.isEnabled = false
        }
        return item
    }

    /// The standing items. Their actions follow the macOS convention for their key equivalents: ⌘, is Settings…,
    /// ⌘Q is Quit, and its Option alternate is Restart.
    private func standardItems(_ row: P.Row) -> [NSMenuItem] {
        let (key, modifiers) = Self.keyEquivalent(row.keyEquivalent ?? "")
        let primary: BrainBarMenuAction? = modifiers == .command ? [",": .settings, "q": .quit][key] : nil
        let item = standardItem(row.strings.first ?? "", key: key, modifiers: modifiers, action: primary)
        guard let alternate = row.optionAlternate else { return [item] }
        let alt = standardItem(alternate, key: key, modifiers: modifiers.union(.option), action: primary == .quit ? .restart : nil)
        alt.isAlternate = true
        return [item, alt]
    }

    private func standardItem(_ title: String, key: String, modifiers: NSEvent.ModifierFlags, action: BrainBarMenuAction?) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action == nil ? nil : #selector(performItem), keyEquivalent: key)
        item.keyEquivalentModifierMask = modifiers
        item.target = self
        item.representedObject = action
        item.isEnabled = action != nil
        return item
    }

    /// "⌘Q" → ("q", ⌘): modifier glyphs, then the key, lowercased as AppKit wants it without Shift.
    static func keyEquivalent(_ glyphs: String) -> (String, NSEvent.ModifierFlags) {
        let flags: [Character: NSEvent.ModifierFlags] = ["⌘": .command, "⌥": .option, "⇧": .shift, "⌃": .control]
        let modifiers = glyphs.compactMap { flags[$0] }.reduce(into: NSEvent.ModifierFlags()) { $0.formUnion($1) }
        return (String(glyphs.filter { flags[$0] == nil }).lowercased(), modifiers)
    }
}

/// The container a hosted row sits in: sized to its row, and one accessibility element that reads the row's words in
/// order (a Needs-you row stays a group, so its button remains a separate control).
final class BrainBarMenuRowHost: NSView {
    let hosting: NSView

    init(_ row: BrainBarMenuRowView, words: [String], isAction: Bool) {
        hosting = NSHostingView(rootView: row)
        let height = ceil(hosting.fittingSize.height)
        super.init(frame: NSRect(x: 0, y: 0, width: BrainBarMenuMetrics.width, height: height))
        hosting.frame = bounds
        hosting.autoresizingMask = [.width, .height]
        addSubview(hosting)
        setAccessibilityElement(!isAction)
        setAccessibilityRole(isAction ? .group : .staticText)
        setAccessibilityLabel(words.joined(separator: " "))
    }

    @available(*, unavailable) required init?(coder: NSCoder) { nil }
}
