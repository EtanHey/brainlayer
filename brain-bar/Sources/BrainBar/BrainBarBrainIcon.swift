import AppKit

/// A1's own artwork on a 22-point grid, inset to a 20-point glyph. Drawing handlers
/// keep the image vector-backed; template tinting is left to the status bar.
@MainActor enum BrainBarBrainIcon {
    static func image(for presentation: BrainBarMenuPresentation) -> NSImage {
        let marked: Bool = presentation.icon == .marked
        let image: NSImage = NSImage(size: NSSize(width: 22, height: 22), flipped: true) { _ in
            guard let context = NSGraphicsContext.current?.cgContext else { return false }
            context.saveGState()
            defer { context.restoreGState() }
            // A1 reference captures align the 20-point SVG at the slot’s top, with a 1-point horizontal inset.
            context.translateBy(x: 1, y: 0)
            context.scaleBy(x: 20.0 / 22.0, y: 20.0 / 22.0)
            context.clip(to: CGRect(x: 0, y: 0, width: 22, height: 22))
            context.beginTransparencyLayer(auxiliaryInfo: nil)
            let color: NSColor = marked ? NSColor(srgbRed: 1, green: 59.0 / 255.0, blue: 48.0 / 255.0, alpha: 1) : .black
            context.setFillColor(color.cgColor)
            // SVG lobes, expanded by half the 2.7-point stroke. Opaque fills form their union.
            let lobes: [(CGFloat, CGFloat, CGFloat)] = [
                (6.1, 7.2, 3.3), (15.9, 7.2, 3.3), (4.7, 11.4, 3.1), (17.3, 11.4, 3.1),
                (6.1, 15.4, 3.2), (15.9, 15.4, 3.2), (8.9, 5.1, 2.8), (13.1, 5.1, 2.8),
                (8.9, 17.4, 2.7), (13.1, 17.4, 2.7),
            ]
            for (x, y, r) in lobes {
                context.fillEllipse(in: CGRect(x: x - r - 1.35, y: y - r - 1.35, width: (r + 1.35) * 2, height: (r + 1.35) * 2))
            }
            if marked {
                context.fillEllipse(in: CGRect(x: 5.6, y: 5.1, width: 10.8, height: 12.4))
                context.setFillColor(NSColor.white.cgColor)
                context.addPath(CGPath(roundedRect: CGRect(x: 9.8, y: 5.6, width: 2.4, height: 8),
                                       cornerWidth: 1.2, cornerHeight: 1.2, transform: nil))
                context.fillPath()
                context.fillEllipse(in: CGRect(x: 9.55, y: 15.15, width: 2.9, height: 2.9))
            } else {
                // m-plain subtracts the entire core, including the original lobe fills.
                context.setBlendMode(.clear)
                for (x, y, r) in lobes {
                    context.fillEllipse(in: CGRect(x: x - r, y: y - r, width: r * 2, height: r * 2))
                }
                context.fillEllipse(in: CGRect(x: 5.6, y: 5.1, width: 10.8, height: 12.4))
                context.setBlendMode(.normal)
                context.setStrokeColor(NSColor.black.cgColor)
                context.setLineCap(.round)
                context.setLineWidth(1.35)
                context.move(to: CGPoint(x: 11, y: 4.4)); context.addLine(to: CGPoint(x: 11, y: 19))
                context.strokePath()
                context.setLineWidth(1.25)
                let folds: [(CGFloat, CGFloat, CGFloat)] = [(4.9, 10.1, 1), (5.9, 14.6, 1), (17.1, 10.1, -1), (16.1, 14.6, -1)]
                for (x, y, direction) in folds {
                    let upper = y == 10.1
                    context.move(to: CGPoint(x: x, y: y))
                    context.addCurve(to: CGPoint(x: x + direction * (upper ? 3.1 : 2.8), y: y + (upper ? 2.1 : 1.1)),
                        control1: CGPoint(x: x + direction * (upper ? 1.5 : 1.1), y: y - (upper ? 0.2 : 0.4)),
                        control2: CGPoint(x: x + direction * (upper ? 2.7 : 2.2), y: y + (upper ? 0.6 : 0)))
                    context.strokePath()
                }
            }
            context.endTransparencyLayer()
            return true
        }
        image.isTemplate = !marked
        image.accessibilityDescription = "BrainBar — " + presentation.headline
        return image
    }
}
