import AppKit
import Foundation

enum SparklineRenderer {
    static func render(state: PipelineState, values: [Int], size: NSSize = NSSize(width: 44, height: 18)) -> NSImage {
        let image = NSImage(size: size)
        image.lockFocus()

        let rect = NSRect(origin: .zero, size: size)
        NSColor.clear.setFill()
        rect.fill()

        let indicatorRect = NSRect(x: 1, y: size.height - 7, width: 5, height: 5)
        state.color.setFill()
        NSBezierPath(ovalIn: indicatorRect).fill()

        guard values.count > 1 else {
            image.unlockFocus()
            image.isTemplate = false
            return image
        }

        let maxValue = max(values.max() ?? 0, 1)
        let chartRect = NSRect(x: 8, y: 2, width: size.width - 10, height: size.height - 4)
        let step = chartRect.width / CGFloat(max(values.count - 1, 1))
        let path = NSBezierPath()
        path.lineWidth = 1.6

        for (index, value) in values.enumerated() {
            let x = chartRect.minX + CGFloat(index) * step
            let normalized = CGFloat(value) / CGFloat(maxValue)
            let y = chartRect.minY + normalized * chartRect.height
            let point = NSPoint(x: x, y: y)
            if index == 0 {
                path.move(to: point)
            } else {
                path.line(to: point)
            }
        }

        state.color.setStroke()
        path.stroke()

        image.unlockFocus()
        image.isTemplate = false
        return image
    }
}
