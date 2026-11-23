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

/// Legacy-menu fixtures own a stored opt-out, removed when the controller releases it.
private final class BrainBarLegacyMenuDefaults: UserDefaults, @unchecked Sendable {
    private let domainName: String

    init(domainName: String) {
        self.domainName = domainName
        super.init(suiteName: domainName)!
        set(false, forKey: BrainBarNewMenuPreference.key)
        set(BrainBarDesign.classic.rawValue, forKey: BrainBarDesign.key)
    }

    deinit { removePersistentDomain(forName: domainName) }
}

func brainBarLegacyMenuDefaults() -> UserDefaults {
    BrainBarLegacyMenuDefaults(domainName: "BrainBarLegacyMenuTests." + UUID().uuidString)
}
