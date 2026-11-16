import SwiftUI

/// The visual vocabulary (DESIGN-SPEC § 3): position means time, fill means amount, dots mean count, dashed means
/// unknown, empty means a measured zero. Each form draws only what its value says; none decides a state.

/// The colours a visual draws with, so the same form serves the window (its own planes) and the dropdown (the system
/// menu material).
struct NDPalette: Equatable {
    /// D7.5: which palette this is. Equality compares it, so a view handed another palette redraws (the selected
    /// dropdown row's bars used to keep their muted colours on the blue selection, because every palette compared equal).
    var id: String = "window"
    var ink: Color, ink2: Color, track: Color, saves: Color, reads: Color, fault: Color, warn: Color, plane: Color
    static let window = NDPalette(id: "window", ink: ND.ink, ink2: ND.ink2, track: ND.track, saves: ND.lumenGraphic, reads: ND.tide,
                                  fault: ND.fault, warn: ND.warn, plane: ND.night)
    static let menu = NDPalette(id: "menu", ink: Color(nsColor: .labelColor), ink2: Color(nsColor: .secondaryLabelColor), track: ND.menuTrack,
                                saves: ND.menuSaves, reads: ND.menuReads, fault: ND.fault, warn: ND.warn, plane: .clear)
    /// Review polish 2: on a highlighted dropdown row the waveform, the gauge's neutral parts and the number take the
    /// system's selected text colour (saves 100 %, reads 55 %); fault and warning stay their colours.
    static let menuSelected: NDPalette = {
        let selected = Color(nsColor: .alternateSelectedControlTextColor)
        return NDPalette(id: "menu.selected", ink: selected, ink2: selected, track: selected.opacity(0.4), saves: selected,
                         reads: selected.opacity(0.6),
                         fault: ND.fault, warn: ND.warn, plane: .clear)
    }()
    static func == (a: Self, b: Self) -> Bool { a.id == b.id }
}

private struct NDPaletteKey: EnvironmentKey { static let defaultValue = NDPalette.window }
extension EnvironmentValues {
    var ndPalette: NDPalette {
        get { self[NDPaletteKey.self] }
        set { self[NDPaletteKey.self] = newValue }
    }
}

/// Unknown is always this: a dashed line, 3 pt dash, 3 pt gap, 1.5 pt, Ink 2.
struct NDDashed: View {
    @Environment(\.ndPalette) private var palette
    var color: Color? = nil
    var lineWidth: CGFloat = 1.5
    var body: some View {
        Canvas { ctx, size in
            var path = Path()
            path.move(to: CGPoint(x: 0, y: size.height / 2))
            path.addLine(to: CGPoint(x: size.width, y: size.height / 2))
            ctx.stroke(path, with: .color(color ?? palette.ink2), style: StrokeStyle(lineWidth: lineWidth, dash: [3, 3]))
        }
        .accessibilityHidden(true)
    }
}

// MARK: V1 waveform

/// Saves by agents above a 1 pt line (62 % of the height), conversations read below (38 %), newest on the right; each
/// half scaled to its own maximum (floors: 4 saves, 40 reads). A zero bucket draws nothing.
///
/// Review B1: two layers. The baseline (or the dashed unknown) and the now tick are static, across the whole plot. The
/// bars are a moving strip clipped to the plot: on a one-bucket shift the data arrives with its outgoing bucket, and in
/// that same transaction the strip stands one bucket width right (where every bar was) and slides to rest by transform
/// over 240 ms with the strong ease-out; the outgoing bucket leaves under the clip. Reduce Motion: in place.
struct NDWaveform: View {
    enum Size { case hero, menu }
    @Environment(\.ndPalette) private var palette
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let wave: NDWave?
    let size: Size
    var nowTick = false
    /// The bucket under the pointer (H2): that pair at 100 %, every other bar at 32 %, a 1 pt line marks it.
    var hover: Int? = nil

    var body: some View {
        let target = wave?.sequence ?? 0
        ZStack {
            Canvas { ctx, canvas in
                Self.drawStatic(ctx, canvas, wave: wave, size: size, palette: palette, nowTick: nowTick, hover: hover)
            }
            NDWaveStrip(phase: Double(target), target: target, wave: wave, size: size, palette: palette, nowTick: nowTick,
                        hover: hover)
                .clipShape(Rectangle())
        }
        .animation(wave?.outgoing != nil && !reduceMotion ? NDMotion.shift : nil, value: target)
        .accessibilityHidden(true)
    }

    /// R1 geometry (§ 3.1): the window's 6 pt bars, the dropdown's 3 pt bars, each centred in its pitch.
    static func layout(width: CGFloat, count: Int, size: Size) -> (bar: CGFloat, gap: CGFloat) {
        let pitch = width / CGFloat(max(count, 1))
        let bar = min(size == .hero ? 6 : 3, pitch)
        return (bar, pitch - bar)
    }

    /// 2 pt corners on the window's bars, 1 pt in the dropdown, never more than half a bar.
    static func barRadius(height: CGFloat, width: CGFloat, pill: Bool, corner: CGFloat) -> CGFloat {
        min(corner, height / 2, width / 2)
    }

    /// The halves: saves above the baseline (62 pt, dropdown 10), reads below (44 pt, dropdown 6).
    static func halves(_ size: Size) -> (up: CGFloat, down: CGFloat) { size == .hero ? (62, 44) : (10, 6) }

    /// The plot's height and baseline centre for a canvas (the 8 pt under the window's plot hold the now tick).
    static func plot(_ canvas: CGSize, nowTick: Bool) -> (h: CGFloat, cy: CGFloat) {
        let h = canvas.height - (nowTick ? 8 : 0)
        let up = h >= 100 ? 62 : (h * 10 / 18).rounded()
        return (h, up + (h >= 100 ? 1 : 0.5))
    }

    /// The static layer's baseline: the same rect before, during and after a shift.
    static func baseline(_ canvas: CGSize, nowTick: Bool) -> CGRect {
        CGRect(x: 0, y: plot(canvas, nowTick: nowTick).cy - 0.5, width: canvas.width, height: 1)
    }

    static func drawStatic(_ ctx: GraphicsContext, _ canvas: CGSize, wave: NDWave?, size: Size, palette: NDPalette, nowTick: Bool,
                           hover: Int? = nil) {
        let (h, cy) = plot(canvas, nowTick: nowTick)
        guard let wave, wave.readable else {
            // Unknown: the baseline dashed, no bars.
            var path = Path()
            path.move(to: CGPoint(x: 0, y: cy)); path.addLine(to: CGPoint(x: canvas.width, y: cy))
            ctx.stroke(path, with: .color(palette.ink2), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            return
        }
        ctx.fill(Path(baseline(canvas, nowTick: nowTick)), with: .color(palette.track))
        let n = wave.count
        guard n > 0 else { return }
        let pitch = canvas.width / CGFloat(n)
        if nowTick {
            let x = CGFloat(n - 1) * pitch + pitch / 2 - 0.5
            ctx.fill(Path(CGRect(x: x, y: h + 3, width: 1, height: 5)), with: .color(ND.t3))
        }
        if let hover, hover < n {
            let x = CGFloat(hover) * pitch + pitch / 2 - 0.5
            ctx.fill(Path(CGRect(x: x, y: 0, width: 1, height: h)), with: .color(ND.t2.opacity(0.45)))
        }
    }

    /// The moving strip's offset: one bucket width (the pitch) at the start of a shift, 0 at rest, never outside that.
    static func offset(target: Int, phase: Double, pitch: CGFloat) -> CGFloat {
        min(max(CGFloat(Double(target) - phase) * pitch, 0), pitch)
    }

    /// The strip's bars and their left x at an offset: n at rest, n + 1 while the outgoing bucket is still sliding out.
    static func strip(_ wave: NDWave, width: CGFloat, size: Size, offset: CGFloat) -> [(x: CGFloat, bucket: NDWave.Bucket)] {
        let n = wave.count
        let (bw, gap) = layout(width: width, count: n, size: size)
        let pitch = bw + gap, inset = gap / 2
        var out: [(x: CGFloat, bucket: NDWave.Bucket)] = []
        if offset > 0, let outgoing = wave.outgoing { out.append((offset - pitch + inset, outgoing)) }
        for i in 0..<n {
            let bucket = NDWave.Bucket(saves: i < wave.saves.count ? wave.saves[i] : 0, reads: i < wave.reads.count ? wave.reads[i] : 0)
            out.append((CGFloat(i) * pitch + inset + offset, bucket))
        }
        return out
    }
}

/// The bars, animatable over the shift's phase (B1). `target` is the wave's grid index; `phase` runs from the last one
/// to it, so the offset starts at exactly one pitch in the transaction that brings the new bucket.
struct NDWaveStrip: View, Animatable {
    var phase: Double
    let target: Int
    let wave: NDWave?
    let size: NDWaveform.Size
    let palette: NDPalette
    let nowTick: Bool
    var hover: Int? = nil

    nonisolated var animatableData: Double {
        get { phase }
        set { phase = newValue }
    }

    var body: some View {
        Canvas { ctx, canvas in draw(ctx, canvas) }
    }

    private func draw(_ ctx: GraphicsContext, _ canvas: CGSize) {
        guard let wave, wave.readable, wave.count > 0 else { return }
        let (_, cy) = NDWaveform.plot(canvas, nowTick: nowTick)
        let (up, down) = NDWaveform.halves(size)
        let (bw, gap) = NDWaveform.layout(width: canvas.width, count: wave.count, size: size)
        let offset = NDWaveform.offset(target: target, phase: phase, pitch: bw + gap)
        // Each half to its own maximum; the scale includes the outgoing bucket, so no bar changes height as it leaves.
        let maxSaves = CGFloat(max(wave.saves.max() ?? 0, wave.outgoing?.saves ?? 0, 1))
        let maxReads = CGFloat(max(wave.reads.max() ?? 0, wave.outgoing?.reads ?? 0, 1))
        let minBar: CGFloat = size == .hero ? 2 : 1.5
        let corner: CGFloat = size == .hero ? 2 : 1
        let first = offset > 0 && wave.outgoing != nil ? -1 : 0
        for (k, item) in NDWaveform.strip(wave, width: canvas.width, size: size, offset: offset).enumerated() {
            let (x, b) = item
            let index = k + first
            let alpha: Double = hover.map { $0 == index ? 1 : 0.32 } ?? 1
            if b.saves > 0 {
                let bh = max(minBar, CGFloat(b.saves) / maxSaves * up)
                let radius = NDWaveform.barRadius(height: bh, width: bw, pill: false, corner: corner)
                ctx.fill(Path(roundedRect: CGRect(x: x, y: cy - 0.5 - bh, width: bw, height: bh), cornerRadius: radius),
                         with: .color(palette.saves.opacity(alpha)))
            }
            if b.reads > 0 {
                let bh = max(minBar, CGFloat(b.reads) / maxReads * down)
                let radius = NDWaveform.barRadius(height: bh, width: bw, pill: false, corner: corner)
                ctx.fill(Path(roundedRect: CGRect(x: x, y: cy + 0.5, width: bw, height: bh), cornerRadius: radius),
                         with: .color(palette.reads.opacity(alpha)))
            }
        }
    }
}

// MARK: V2 age against its limit

/// A notch on a track that ends at the limit. The window draws the limit as the area's joined line; the dropdown, where
/// there is no panel, draws a 1 × 7 pt end tick.
struct NDGaugeView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let visual: NDModel.Visual
    var endTick = false

    /// Motion 6: when the value changes, the notch glides there (240 ms, strong ease-out); Reduce Motion moves it at once.
    var body: some View {
        NDGaugeShape(visual: visual, endTick: endTick)
            .animation(reduceMotion ? nil : NDMotion.shift, value: visual)
            .frame(height: 12)
            .accessibilityHidden(true)
    }

    static func tone(_ t: NDModel.Tone, _ p: NDPalette) -> Color { NDGaugeShape.tone(t, p) }
}

/// The gauge as drawn, animatable over its fraction.
struct NDGaugeShape: View, Animatable {
    @Environment(\.ndPalette) private var palette
    var visual: NDModel.Visual
    var endTick = false

    nonisolated var animatableData: Double {
        get { if case let .gauge(g) = visual { return g.fraction }; return 0 }
        set { if case var .gauge(g) = visual { g.fraction = newValue; visual = .gauge(g) } }
    }

    var body: some View {
        Canvas { ctx, size in Self.draw(ctx, size, visual: visual, palette: palette, endTick: endTick) }
    }

    static func tone(_ t: NDModel.Tone, _ p: NDPalette) -> Color {
        switch t {
        case .fine: p.ink2
        case .warn: p.warn
        case .fault: p.fault
        }
    }

    static func draw(_ ctx: GraphicsContext, _ size: CGSize, visual: NDModel.Visual, palette: NDPalette, endTick: Bool) {
        let w = size.width, mid: CGFloat = 6
        func track() { ctx.fill(Path(CGRect(x: 0, y: mid - 0.5, width: w, height: 1)), with: .color(palette.track)) }
        func tick() { if endTick { ctx.fill(Path(CGRect(x: w - 1, y: mid - 3.5, width: 1, height: 7)), with: .color(palette.track)) } }
        switch visual {
        case .unknown:
            var path = Path()
            path.move(to: CGPoint(x: 0, y: mid)); path.addLine(to: CGPoint(x: w, y: mid))
            ctx.stroke(path, with: .color(palette.ink2), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
        case .emptyTrack:
            track(); tick()
        case let .gauge(g):
            let x = max(1, min(g.fraction, 1) * (w - 1))
            let fault = g.tone == .fault
            if g.concerned && !g.pinned {
                var path = Path()
                path.move(to: CGPoint(x: x + 3, y: mid)); path.addLine(to: CGPoint(x: w, y: mid))
                ctx.stroke(path, with: .color(palette.fault), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            } else {
                track()
            }
            tick()
            ctx.fill(Path(roundedRect: CGRect(x: 0, y: mid - 1, width: x, height: 2), cornerRadius: 1), with: .color(tone(g.tone, palette)))
            ctx.fill(Path(roundedRect: CGRect(x: x - 1, y: mid - 5, width: 2, height: 10), cornerRadius: 1),
                     with: .color(fault ? palette.fault : palette.ink))
        case .streak:
            // V4 with V2 pinned: the notch on the limit, in Fault, all the way.
            ctx.fill(Path(roundedRect: CGRect(x: 0, y: mid - 1, width: w, height: 2), cornerRadius: 1), with: .color(palette.fault))
            ctx.fill(Path(roundedRect: CGRect(x: w - 2, y: mid - 5, width: 2, height: 10), cornerRadius: 1), with: .color(palette.fault))
        default:
            break
        }
    }
}

// MARK: V3 fill

struct NDFillBar: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let share: Double?
    var body: some View {
        NDFillShape(share: share).animation(reduceMotion ? nil : NDMotion.shift, value: share)
    }
}

struct NDFillShape: View, Animatable {
    @Environment(\.ndPalette) private var palette
    var share: Double?
    nonisolated var animatableData: Double {
        get { share ?? 0 }
        set { if share != nil { share = newValue } }
    }
    var body: some View {
        Canvas { ctx, size in
            guard let share else {
                var path = Path()
                path.move(to: CGPoint(x: 0, y: 6)); path.addLine(to: CGPoint(x: size.width, y: 6))
                ctx.stroke(path, with: .color(palette.ink2), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
                return
            }
            ctx.fill(Path(roundedRect: CGRect(x: 0, y: 4, width: size.width, height: 4), cornerRadius: 2), with: .color(palette.track))
            let p = max(0, min(1, share))
            if p > 0 {
                ctx.fill(Path(roundedRect: CGRect(x: 0, y: 4, width: max(2, size.width * p), height: 4), cornerRadius: 2),
                         with: .color(palette.ink2))
            }
        }
        .frame(height: 12)
        .accessibilityHidden(true)
    }
}

// MARK: V4 streak dots

struct NDStreakDots: View {
    @Environment(\.ndPalette) private var palette
    let count: Int
    var body: some View {
        let n = min(3, max(0, count))
        HStack(spacing: 3) {
            ForEach(0..<n, id: \.self) { _ in Circle().fill(palette.fault).frame(width: 5, height: 5) }
        }
        .accessibilityHidden(true)
    }
}

// MARK: V5 marks

/// Fault square, unknown dashed ring, running arc, the "Everything else is fine." check. Fine draws nothing.
struct NDMarkView: View {
    @Environment(\.ndPalette) private var palette
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let mark: NDModel.Mark
    var size: CGFloat = 12

    /// A fixed phase for headless renders and captures, so two runs draw the same still.
    @MainActor static var captureClock: Date?

    var body: some View {
        Group {
            switch mark {
            case .none: Color.clear
            case .fault:
                RoundedRectangle(cornerRadius: 1.5, style: .continuous).fill(palette.fault).frame(width: 8, height: 8)
            case .unknown:
                Circle().strokeBorder(palette.ink2, style: StrokeStyle(lineWidth: 1.5, dash: [3.14, 3.14])).frame(width: 9.5, height: 9.5)
            case .running:
                if reduceMotion || Self.captureClock != nil {
                    arc(angle: Self.captureClock.map { Self.angle(at: $0) } ?? 0)
                } else {
                    TimelineView(.animation) { context in arc(angle: Self.angle(at: context.date)) }
                }
            case .check:
                Path { p in
                    p.move(to: CGPoint(x: 2.6, y: 6.4)); p.addLine(to: CGPoint(x: 5, y: 8.7)); p.addLine(to: CGPoint(x: 9.6, y: 3.6))
                }
                .stroke(palette.ink2, style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
            }
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }

    /// Motion 8: one turn per 1.6 s, linear; still under Reduce Motion.
    static func angle(at date: Date) -> Double {
        date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: NDMotion.runningPeriod) / NDMotion.runningPeriod * 360
    }

    private func arc(angle: Double) -> some View {
        ZStack {
            Circle().stroke(palette.track, lineWidth: 1.5)
            Circle().trim(from: 0, to: 0.25).stroke(ND.lumen, style: StrokeStyle(lineWidth: 1.6, lineCap: .round))
                .rotationEffect(.degrees(angle - 90))
        }
        .frame(width: 8, height: 8)
    }
}

// MARK: V6 agents

struct NDSessionDots: View {
    @Environment(\.ndPalette) private var palette
    let sessions: Int?
    /// More than this many live sessions draw this many dots; the accessibility label says the count.
    static let cap = 8
    var body: some View {
        if let sessions {
            HStack(spacing: 2) {
                ForEach(0..<min(sessions, Self.cap), id: \.self) { _ in Circle().fill(palette.ink).frame(width: 6, height: 6) }
            }
            .accessibilityHidden(true)
        } else {
            NDMarkView(mark: .unknown)
        }
    }
}

struct NDSavesBar: View {
    @Environment(\.ndPalette) private var palette
    let saves: Int?
    let scale: Int
    var body: some View {
        Canvas { ctx, size in
            guard let saves else {
                var path = Path()
                path.move(to: CGPoint(x: 0, y: 6)); path.addLine(to: CGPoint(x: size.width, y: 6))
                ctx.stroke(path, with: .color(palette.ink2), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
                return
            }
            guard saves > 0 else { return }
            let p = CGFloat(saves) / CGFloat(max(scale, 1))
            ctx.fill(Path(roundedRect: CGRect(x: 0, y: 4, width: max(2, size.width * p), height: 4), cornerRadius: 2), with: .color(palette.saves))
        }
        .frame(height: 12)
        .accessibilityHidden(true)
    }
}

// MARK: V7 rail

/// When things ran and run next. Filled dot: a finished, verified run; Fault dot: a failed attempt; hollow ring: the
/// next run; a band to "now" for an always-on job; chevrons past either end.
struct NDRail: View {
    struct Event: Equatable {
        /// ok: a finished, verified run. fail: a failed attempt. next: the next run. unknown: a run whose outcome is not
        /// recorded (or still running): a dashed ring, never a success dot (F02). unconfirmed: an attempt from the app's own
        /// log on a row whose newest copy is unknown this moment (review N1): an open 1 pt ring in Ink 3 (Ink 2 at 55 %),
        /// smaller and fainter than the next-run ring (Ink 2, 1.5 pt, filled with the plane), and always left of now.
        enum Kind: Equatable { case ok, fail, next, unknown, unconfirmed, retried }
        var hours: Double
        var kind: Kind
    }
    @Environment(\.ndPalette) private var palette
    var past: Double
    var ahead: Double
    var events: [Event] = []
    /// An always-on job: an Ink 2 band up to now.
    var band = false
    /// A job that runs every few minutes: a dotted band up to now (R1 legend "every few minutes").
    var dotted = false
    /// The log could not be read: the dashed line to now.
    var unknown = false
    /// Drive disconnected: a Fault dashed line from the last dot (hours, ≤ 0) to now; no next ring.
    var stalledFrom: Double? = nil
    /// Review N1: the row's newest copy is unknown, so the line from its first dot (hours, <= 0) to now is the unknown
    /// stroke (Ink 2, 1.5 pt, 3 / 3 dashes), as the Overview lane draws it.
    var unconfirmedFrom: Double? = nil
    /// A run beyond the left end: "‹ 4 d".
    var olderLabel: String? = nil
    var nowLine = true

    var body: some View {
        Canvas { ctx, size in draw(ctx, size) }
            .frame(height: 14)
            .accessibilityHidden(true)
    }

    private func x(_ h: Double, _ w: CGFloat) -> CGFloat { CGFloat((h + past) / (past + ahead)) * w }

    /// Fix 15: when the last run and the next run both sit within 6 pt of now, the rail draws only the next-run ring
    /// (the two marks would overprint); the lane's sentence still says both.
    static func shown(_ events: [Event], width: CGFloat, past: Double, ahead: Double) -> [Event] {
        func nearNow(_ e: Event) -> Bool { abs(CGFloat(e.hours / (past + ahead)) * width) <= 6 }
        guard events.contains(where: { $0.kind == .next && nearNow($0) }) else { return events }
        return events.filter { $0.kind == .next || !nearNow($0) }
    }

    private func draw(_ ctx: GraphicsContext, _ size: CGSize) {
        let w = size.width, mid: CGFloat = 7, nowX = x(0, w)
        ctx.fill(Path(CGRect(x: 0, y: mid - 0.5, width: w, height: 1)), with: .color(ND.hair2))
        if unknown {
            var p = Path(); p.move(to: CGPoint(x: 0, y: mid)); p.addLine(to: CGPoint(x: nowX, y: mid))
            ctx.stroke(p, with: .color(palette.ink2), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
        }
        if let from = unconfirmedFrom, stalledFrom == nil {
            var p = Path(); p.move(to: CGPoint(x: max(0, x(from, w)), y: mid)); p.addLine(to: CGPoint(x: nowX, y: mid))
            ctx.stroke(p, with: .color(palette.ink2), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
        }
        if let from = stalledFrom {
            var p = Path(); p.move(to: CGPoint(x: x(from, w) + 4, y: mid)); p.addLine(to: CGPoint(x: nowX, y: mid))
            ctx.stroke(p, with: .color(palette.fault), style: StrokeStyle(lineWidth: 1.5, dash: [3, 3]))
        }
        if band { ctx.fill(Path(roundedRect: CGRect(x: 0, y: mid - 1, width: nowX, height: 2), cornerRadius: 1), with: .color(palette.ink2)) }
        if dotted {
            var x: CGFloat = 1
            while x < nowX - 2 {
                ctx.fill(Path(ellipseIn: CGRect(x: x - 1, y: mid - 1, width: 2, height: 2)), with: .color(palette.ink2))
                x += 4
            }
        }
        if nowLine { ctx.fill(Path(CGRect(x: nowX - 0.5, y: 0, width: 1, height: 14)), with: .color(palette.ink2.opacity(0.6))) }
        func chevron(left: Bool, color: Color) {
            var p = Path()
            if left {
                p.move(to: CGPoint(x: 6, y: mid - 3.5)); p.addLine(to: CGPoint(x: 2.5, y: mid)); p.addLine(to: CGPoint(x: 6, y: mid + 3.5))
            } else {
                p.move(to: CGPoint(x: w - 6, y: mid - 3.5)); p.addLine(to: CGPoint(x: w - 2.5, y: mid)); p.addLine(to: CGPoint(x: w - 6, y: mid + 3.5))
            }
            ctx.stroke(p, with: .color(color), style: StrokeStyle(lineWidth: 1.4, lineCap: .round, lineJoin: .round))
        }
        for e in Self.shown(events, width: w, past: past, ahead: ahead) {
            if e.hours < -past {
                chevron(left: true, color: e.kind == .fail ? palette.fault : palette.ink2)
                if let olderLabel {
                    let failed = e.kind == .fail
                    ctx.draw(Text(olderLabel).font(.system(size: 11, weight: failed ? .semibold : .regular))
                        .foregroundColor(failed ? ND.redText : palette.ink2),
                             at: CGPoint(x: 10, y: mid), anchor: .leading)
                }
                continue
            }
            if e.hours > ahead { chevron(left: false, color: palette.ink2); continue }
            let cx = x(e.hours, w)
            switch e.kind {
            case .next:
                let r = CGRect(x: cx - 3.25, y: mid - 3.25, width: 6.5, height: 6.5)
                ctx.fill(Path(ellipseIn: r), with: .color(palette.plane))
                ctx.stroke(Path(ellipseIn: r), with: .color(palette.ink2), lineWidth: 1.5)
            case .ok:
                ctx.fill(Path(ellipseIn: CGRect(x: cx - 3.5, y: mid - 3.5, width: 7, height: 7)), with: .color(palette.ink))
            case .fail:
                ctx.fill(Path(ellipseIn: CGRect(x: cx - 3.5, y: mid - 3.5, width: 7, height: 7)), with: .color(palette.fault))
            case .retried:
                // R1.2 change 13: failed, then copied on retry the same night: a hollow red ring.
                ctx.fill(Path(ellipseIn: CGRect(x: cx - 3.2, y: mid - 3.2, width: 6.4, height: 6.4)), with: .color(ND.card))
                ctx.stroke(Path(ellipseIn: CGRect(x: cx - 3.2, y: mid - 3.2, width: 6.4, height: 6.4)), with: .color(palette.fault),
                           lineWidth: 1.4)
            case .unconfirmed:
                let r = CGRect(x: cx - 3, y: mid - 3, width: 6, height: 6)
                ctx.fill(Path(ellipseIn: r), with: .color(palette.plane))
                ctx.stroke(Path(ellipseIn: r), with: .color(palette.ink2.opacity(0.55)), lineWidth: 1)
            case .unknown:
                let r = CGRect(x: cx - 3.5, y: mid - 3.5, width: 7, height: 7)
                ctx.fill(Path(ellipseIn: r), with: .color(palette.plane))
                ctx.stroke(Path(ellipseIn: r), with: .color(palette.ink2), style: StrokeStyle(lineWidth: 1.5, dash: [2.2, 2.2]))
            }
        }
    }
}

// MARK: Numbers

/// A tabular number with its unit in the secondary ink.
struct NDNumberText: View {
    @Environment(\.ndPalette) private var palette
    let number: NDModel.Number
    var size: CGFloat = 13
    /// On a highlighted menu row the numeral takes the selected text colour (fix 7).
    var selected = false

    var body: some View {
        if number.value == NDModel.Number.faultMark.value {
            RoundedRectangle(cornerRadius: 1.5, style: .continuous).fill(palette.fault).frame(width: 8, height: 8)
                .frame(height: 12)
        } else {
            let color: Color = switch number.tone {
            case .fault: palette.fault
            case .warn: palette.warn
            case .fine: number.secondary ? palette.ink2 : palette.ink
            }
            let unitColor = number.tone == .fault ? palette.fault : palette.ink2
            let selectedText = Color(nsColor: .alternateSelectedControlTextColor)
            (Text(number.value).foregroundColor(selected ? selectedText : color)
                + Text(number.unit.isEmpty ? "" : "\u{00A0}" + number.unit).foregroundColor(selected ? selectedText : unitColor))
                .font(.system(size: size, weight: number.strong ? .semibold : .regular).monospacedDigit())
                .lineLimit(1)
                .fixedSize()
        }
    }
}

extension NDModel.Number {
    /// A known failure with no state word of its own: the cell draws the fault square.
    static let faultMark = NDModel.Number(value: "■", tone: .fault)
}
