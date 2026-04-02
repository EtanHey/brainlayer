import AppKit
import SwiftUI

/// Transparent NSView overlay that captures scroll wheel events for zoom.
struct ScrollWheelZoomView: NSViewRepresentable {
    @Binding var scale: CGFloat

    func makeNSView(context: Context) -> ScrollWheelNSView {
        let view = ScrollWheelNSView()
        view.onScroll = { delta in
            let factor = 1.0 + delta * 0.03
            scale = max(0.2, min(5.0, scale * factor))
        }
        return view
    }

    func updateNSView(_ nsView: ScrollWheelNSView, context: Context) {
        nsView.onScroll = { delta in
            let factor = 1.0 + delta * 0.03
            scale = max(0.2, min(5.0, scale * factor))
        }
    }
}

final class ScrollWheelNSView: NSView {
    var onScroll: ((CGFloat) -> Void)?

    override func scrollWheel(with event: NSEvent) {
        let delta = event.scrollingDeltaY
        guard abs(delta) > 0.1 else { return }
        onScroll?(delta)
    }

    override var acceptsFirstResponder: Bool { true }
}
