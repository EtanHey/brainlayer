import SwiftUI

/// A status dot that sits on its label's line, centred on the text (Etan row E4, 2026-10-01).
///
/// A bare `Circle` has no text baseline, so in a `.firstTextBaseline` row its bottom edge lands
/// on the label's baseline and the dot rides low; in a `.top` row it rides high. Here the dot is
/// centred on an invisible zero-width line of text in the label's font: the row keeps baseline
/// alignment (a wrapped label stays anchored to its first line) and the dot is centred on that
/// line's height. Give it the label's font with `.font(_:)`.
struct BrainBarStatusDot: View {
    let color: Color
    var size: CGFloat = 7

    var body: some View {
        Text(verbatim: "\u{200B}")
            .frame(width: size)
            .overlay(Circle().fill(color).frame(width: size, height: size))
            .accessibilityHidden(true)
    }
}
