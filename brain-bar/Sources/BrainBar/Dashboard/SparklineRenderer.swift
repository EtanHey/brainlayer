import AppKit
import CoreGraphics
import Foundation

enum SparklineRenderer {
    static func endpoint(
        values: [Int],
        size: NSSize = NSSize(width: 44, height: 18)
    ) -> CGPoint? {
        let width = max(Int(size.width.rounded(.up)), 1)
        let height = max(Int(size.height.rounded(.up)), 1)
        let isCompact = height <= 20 || width <= 52

        guard values.count > 1 else { return nil }

        let maxValue = max(values.max() ?? 0, 1)
        let horizontalInset: CGFloat = isCompact ? 2 : 10
        let verticalInset: CGFloat = isCompact ? 2 : 10
        let chartRect = CGRect(
            x: horizontalInset,
            y: verticalInset,
            width: max(CGFloat(width) - (horizontalInset * 2), 1),
            height: max(CGFloat(height) - (verticalInset * 2), 1)
        )
        let step = chartRect.width / CGFloat(max(values.count - 1, 1))

        guard let lastValue = values.last else { return nil }
        let normalized = CGFloat(lastValue) / CGFloat(maxValue)
        return CGPoint(
            x: chartRect.minX + CGFloat(values.count - 1) * step,
            y: chartRect.minY + normalized * chartRect.height
        )
    }

    static func render(
        state: PipelineState,
        values: [Int],
        size: NSSize = NSSize(width: 44, height: 18),
        accentColor: NSColor? = nil
    ) -> NSImage {
        let width = max(Int(size.width.rounded(.up)), 1)
        let height = max(Int(size.height.rounded(.up)), 1)
        let isCompact = height <= 20 || width <= 52
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let lineColor = accentColor ?? state.color

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return NSImage(size: size)
        }

        let rect = CGRect(origin: .zero, size: CGSize(width: width, height: height))
        context.setFillColor(NSColor.clear.cgColor)
        context.fill(rect)

        context.setAllowsAntialiasing(true)
        context.setShouldAntialias(true)

        guard values.count > 1 else {
            return image(from: context, size: size)
        }

        let maxValue = max(values.max() ?? 0, 1)
        let horizontalInset: CGFloat = isCompact ? 2 : 10
        let verticalInset: CGFloat = isCompact ? 2 : 10
        let chartRect = CGRect(
            x: horizontalInset,
            y: verticalInset,
            width: max(CGFloat(width) - (horizontalInset * 2), 1),
            height: max(CGFloat(height) - (verticalInset * 2), 1)
        )
        let step = chartRect.width / CGFloat(max(values.count - 1, 1))
        let path = CGMutablePath()
        var points: [CGPoint] = []

        if !isCompact {
            context.setStrokeColor(NSColor.separatorColor.withAlphaComponent(0.55).cgColor)
            context.setLineWidth(1)
            for fraction in [0.25, 0.5, 0.75] {
                let y = chartRect.minY + chartRect.height * CGFloat(fraction)
                context.move(to: CGPoint(x: chartRect.minX, y: y))
                context.addLine(to: CGPoint(x: chartRect.maxX, y: y))
            }
            context.strokePath()
        }

        for (index, value) in values.enumerated() {
            let x = chartRect.minX + CGFloat(index) * step
            let normalized = CGFloat(value) / CGFloat(maxValue)
            let y = chartRect.minY + normalized * chartRect.height
            let point = CGPoint(x: x, y: y)
            points.append(point)
            if index == 0 {
                path.move(to: point)
            } else {
                path.addLine(to: point)
            }
        }

        if !isCompact, let first = points.first, let last = points.last {
            let fill = CGMutablePath()
            fill.move(to: CGPoint(x: first.x, y: chartRect.minY))
            for point in points {
                fill.addLine(to: point)
            }
            fill.addLine(to: CGPoint(x: last.x, y: chartRect.minY))
            fill.closeSubpath()
            context.addPath(fill)
            context.setFillColor(lineColor.withAlphaComponent(0.10).cgColor)
            context.fillPath()
        }

        context.addPath(path)
        context.setStrokeColor(lineColor.withAlphaComponent(0.85).cgColor)
        context.setLineWidth(isCompact ? 1.6 : 2)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        context.strokePath()

        return image(from: context, size: size)
    }

    private static func image(from context: CGContext, size: NSSize) -> NSImage {
        guard let cgImage = context.makeImage() else {
            return NSImage(size: size)
        }
        let image = NSImage(cgImage: cgImage, size: size)
        image.isTemplate = false
        return image
    }
}
