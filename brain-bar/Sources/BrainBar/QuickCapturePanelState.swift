// QuickCapturePanelState.swift — Observable state for the quick capture panel.
//
// Tracks panel visibility and current mode (capture vs search).

import Foundation

final class QuickCapturePanelState {

    enum Mode: Equatable {
        case capture
        case search
    }

    private(set) var mode: Mode = .capture
    private(set) var isVisible: Bool = false

    func show() {
        isVisible = true
    }

    func dismiss() {
        isVisible = false
        mode = .capture  // Reset to default mode on dismiss
    }

    func toggle() {
        if isVisible {
            dismiss()
        } else {
            show()
        }
    }

    func switchMode(_ newMode: Mode) {
        mode = newMode
    }
}
