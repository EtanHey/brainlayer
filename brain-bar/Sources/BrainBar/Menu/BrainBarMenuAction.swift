import AppKit
import SwiftUI

/// What a menu item asks for. The row actions are the presentation's own kinds; the standing items are Settings…,
/// Quit and its Option alternate.
enum BrainBarMenuAction: Equatable, Sendable {
    case row(BrainBarMenuPresentation.ActionKind)
    case settings, quit, restart
}
