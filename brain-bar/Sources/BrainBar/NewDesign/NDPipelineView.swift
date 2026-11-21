import AppKit
import SwiftUI

// MARK: - Polylines with rounded corners (the comp's `rpath`, `split`, `at`)

struct NDPolyline: Equatable, Sendable {
    var points: [CGPoint]

    static func seg(_ a: CGPoint, _ b: CGPoint) -> CGFloat { hypot(b.x - a.x, b.y - a.y) }

    var length: CGFloat {
        guard points.count > 1 else { return 0 }
        var total: CGFloat = 0
        for i in 1..<points.count { total += Self.seg(points[i - 1], points[i]) }
        return total
    }

    /// The polyline cut at distance `d` from its start: the part before and the part after.
    func split(_ d: CGFloat) -> (NDPolyline, NDPolyline) {
        guard points.count > 1 else { return (self, self) }
        var walked: CGFloat = 0
        for i in 1..<points.count {
            let a = points[i - 1], b = points[i]
            let l = Self.seg(a, b)
            if walked + l >= d, l > 0 {
                let t = max(0, min(1, (d - walked) / l))
                let p = CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t)
                return (NDPolyline(points: Array(points[..<i]) + [p]), NDPolyline(points: [p] + Array(points[i...])))
            }
            walked += l
        }
        return (self, NDPolyline(points: [points[points.count - 1], points[points.count - 1]]))
    }

    func point(at d: CGFloat) -> CGPoint { split(d).1.points.first ?? .zero }

    /// The path with corners rounded by quadratic curves of radius `r` (never more than half a segment).
    func path(radius r: CGFloat) -> Path {
        var p = Path()
        guard let first = points.first, points.count > 1 else { return p }
        p.move(to: first)
        if points.count > 2 {
            for i in 1..<(points.count - 1) {
                let a = points[i - 1], c = points[i], b = points[i + 1]
                let la = Self.seg(a, c), lb = Self.seg(c, b)
                guard la > 0, lb > 0 else { continue }
                let rr = min(r, la / 2, lb / 2)
                let p1 = CGPoint(x: c.x + (a.x - c.x) / la * rr, y: c.y + (a.y - c.y) / la * rr)
                let p2 = CGPoint(x: c.x + (b.x - c.x) / lb * rr, y: c.y + (b.y - c.y) / lb * rr)
                p.addLine(to: p1)
                p.addQuadCurve(to: p2, control: c)
            }
        }
        p.addLine(to: points[points.count - 1])
        return p
    }

    /// Trimmed by `start` at the start and `end` at the end (a station's radius).
    func trimmed(start: CGFloat = 0, end: CGFloat = 0) -> NDPolyline {
        var line = self
        if start > 0 { line = line.split(start).1 }
        if end > 0 { line = line.split(line.length - end).0 }
        return line
    }
}
