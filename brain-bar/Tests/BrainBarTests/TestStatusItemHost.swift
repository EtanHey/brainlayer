import AppKit
@testable import BrainBar

/// No native status bar/item or menu-bar window is allocated by test fixtures.
@MainActor
final class TestStatusItemHost: BrainBarStatusItemHosting {
    final class Item: BrainBarStatusItem {
        let length: CGFloat
        var menu: NSMenu?
        let button: NSButton? = NSButton(frame: .zero)

        init(length: CGFloat) { self.length = length }
    }

    private(set) var createdItem: Item?
    private(set) var removedItems: [any BrainBarStatusItem] = []

    func statusItem(withLength length: CGFloat) -> any BrainBarStatusItem {
        let item = Item(length: length)
        createdItem = item
        return item
    }

    func removeStatusItem(_ item: any BrainBarStatusItem) {
        precondition(item === createdItem, "Status item belongs to another host")
        removedItems.append(item)
    }
}
