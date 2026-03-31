import AppKit
import CoreGraphics
import Foundation

enum SparklineRenderer {
    static func render(state: PipelineState, values: [Int], size: NSSize = NSSize(width: 44, height: 18)) -> NSImage {
        let width = max(Int(size.width.rounded(.up)), 1)
        let height = max(Int(size.height.rounded(.up)), 1)
        let colorSpace = CGColorSpaceCreateDeviceRGB()

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

        let indicatorRect = CGRect(x: 1, y: CGFloat(height) - 7, width: 5, height: 5)
        context.setFillColor(state.color.cgColor)
        context.fillEllipse(in: indicatorRect)

        guard values.count > 1 else {
            return image(from: context, size: size)
        }

        let maxValue = max(values.max() ?? 0, 1)
        let chartRect = CGRect(x: 8, y: 2, width: CGFloat(width) - 10, height: CGFloat(height) - 4)
        let step = chartRect.width / CGFloat(max(values.count - 1, 1))
        let path = CGMutablePath()

        for (index, value) in values.enumerated() {
            let x = chartRect.minX + CGFloat(index) * step
            let normalized = CGFloat(value) / CGFloat(maxValue)
            let y = chartRect.minY + normalized * chartRect.height
            let point = CGPoint(x: x, y: y)
            if index == 0 {
                path.move(to: point)
            } else {
                path.addLine(to: point)
            }
        }

        context.addPath(path)
        context.setStrokeColor(state.color.cgColor)
        context.setLineWidth(1.6)
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
