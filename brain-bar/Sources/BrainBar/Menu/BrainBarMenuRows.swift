import AppKit
import SwiftUI

/// The hosted rows of the M menu: only the rows a standard `NSMenuItem` cannot draw (the headline with its disc, a
/// line with its mark, spinner or bar, a Needs-you row with its button). Every word comes from the presentation value
/// (`BrainBarMenuPresenter`); nothing here decides a state. `BrainBarMenuViewWordsTests` holds that: the only string
/// literals in this file and the builder are SF Symbol names.
enum BrainBarMenuMetrics {
    /// One width for both forms, so nothing shifts when the form changes. The mocks: A1 340, A2 360.
    static let width: CGFloat = 340
    /// Measured from `NSMenu.size` on macOS 27: a title sits 16 pt from either edge, as the mocks have it.
    static let inset: CGFloat = 16
    /// The disc column (A2's 26 pt), then a 10 pt gap: hosted text starts at 52 pt.
    static let gutter: CGFloat = 26
    static let gap: CGFloat = 10
}

/// Reduce Motion and Increase Contrast, read once per build (tests and the harness pass them explicitly).
struct BrainBarMenuAccessibility: Equatable, Sendable {
    var reduceMotion = false
    var increaseContrast = false

    @MainActor static var system: Self {
        .init(reduceMotion: NSWorkspace.shared.accessibilityDisplayShouldReduceMotion,
              increaseContrast: NSWorkspace.shared.accessibilityDisplayShouldIncreaseContrast)
    }
}

/// Set by the builder when the menu highlights a Needs-you row from the keyboard: its button shows focus.
@MainActor final class BrainBarMenuRowFocus: ObservableObject {
    @Published var keyboardFocused = false
}

struct BrainBarMenuRowView: View {
    typealias P = BrainBarMenuPresentation
    let row: P.Row
    let previous: P.RowKind?
    let next: P.RowKind?
    let accessibility: BrainBarMenuAccessibility
    @ObservedObject var focus: BrainBarMenuRowFocus
    let perform: (P.ActionKind) -> Void

    private var secondary: Color { accessibility.increaseContrast ? .primary : .secondary }
    private var lastInBlock: Bool { next == nil || next == .separator }

    /// One path for every kind: whatever field the value carries is drawn, whatever the row's kind (strings, mark,
    /// spinner, progress, action). The kind only chooses the type style and the spacing.
    var body: some View {
        HStack(alignment: .top, spacing: BrainBarMenuMetrics.gap) {
            Color.clear.frame(width: BrainBarMenuMetrics.gutter, height: gutterHeight).overlay { gutter }.accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 0) {
                words
                if let percent = row.progressPercent {
                    ProgressView(value: Double(min(max(percent, 0), 100)), total: 100).progressViewStyle(.linear)
                        .padding(.top, 6).animation(nil, value: percent)
                }
                if let action = row.action { button(action).padding(.top, 9) }
            }
            .fixedSize(horizontal: false, vertical: true)  // every line wraps in full; nothing is cut with "…"
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.top, row.kind == .headline ? 0 : (gutterHeight - 16) / 2)
            if row.mark != nil, row.spinner { spinner.frame(height: 16).accessibilityHidden(true) }  // A2: a sign at the line's end
        }
        .padding(.horizontal, BrainBarMenuMetrics.inset)
        .padding(.top, spacing.top)
        .padding(.bottom, spacing.bottom)
        .frame(width: BrainBarMenuMetrics.width, alignment: .leading)
        .fixedSize(horizontal: false, vertical: true)
    }

    /// Vertical rhythm from the mocks' CSS (A2 `.head`, `.sub`, `.line`; A1 `.problem`, `.others`), by neighbour only.
    private var spacing: (top: CGFloat, bottom: CGFloat) {
        switch row.kind {
        case .headline: (7, lastInBlock ? 9 : 0)
        case .headlineDetail: (2, lastInBlock ? 9 : 0)
        case .needsYou: (4, next == .factLine ? 6 : 12)
        case .factLine: (0, 12)
        default:  // a line after a Needs-you row stands apart from it (space, not a divider: the value decides dividers)
            ([.headline, .headlineDetail].contains(previous) ? 8 : previous == .separator ? 6
                : [.needsYou, .factLine].contains(previous) ? 10 : 3, lastInBlock ? 9 : 3)
        }
    }

    private var gutterHeight: CGFloat {
        switch row.mark {
        case _ where row.kind == .headline: 26
        case .ok?, .busy?, .wait?: 20
        default: 16
        }
    }

    /// R28b: the value's mark; else its spinner; else a mark from the line's state (behind turns, can't tell waits). A
    /// headline's detail line is the exception: the headline's disc above it already carries that state.
    @ViewBuilder private var gutter: some View {
        if let mark = row.mark {
            BrainBarMenuMark(mark: mark, size: gutterHeight, accessibility: accessibility)
        } else if row.spinner || (row.state == .behind && row.kind != .headlineDetail) {
            spinner
        } else if row.state == .cantTell, row.kind != .headlineDetail {
            Image(systemName: "ellipsis").font(.system(size: 12, weight: .bold)).foregroundStyle(secondary)
        }
    }

    @ViewBuilder private var spinner: some View {
        if accessibility.reduceMotion {  // the spinner becomes a static mark
            Image(systemName: "arrow.triangle.2.circlepath").font(.system(size: 11, weight: .semibold)).foregroundStyle(secondary)
        } else {
            ProgressView().controlSize(.mini)
        }
    }

    /// The strings in order: the first in the kind's style, the rest as facts, then the presenter's two hand-off lines
    /// (it appends them to a row whose action is Copy report) in the label colour.
    @ViewBuilder private var words: some View {
        let handoff = row.action?.kind == .copyReport && row.strings.count >= 3 ? 2 : 0
        let lead = Array(row.strings.dropLast(handoff))
        ForEach(Array(lead.enumerated()), id: \.offset) { index, string in
            let text = Text(Self.keepingNumbersWithUnits(string)).monospacedDigit()
            switch (row.kind, index) {
            case (.headline, 0): text.font(.system(size: 15, weight: .semibold)).frame(minHeight: 26)
            case (.needsYou, 0): text.fontWeight(.semibold)
            default: text.foregroundStyle(secondary).padding(.top, index == 0 ? 0 : 2)
            }
        }
        if handoff > 0 {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(row.strings.suffix(handoff), id: \.self) { Text(Self.keepingNumbersWithUnits($0)).monospacedDigit() }
            }.padding(.top, 6)
        }
    }

    private func button(_ action: P.Action) -> some View {
        HStack(spacing: 0) {
            Button { perform(action.kind) } label: { Text(Self.keepingNumbersWithUnits(action.label)) }
                .buttonStyle(BrainBarMenuActionButtonStyle(accessibility: accessibility))
        }
        // The system focus ring, concentric with the button (radius 13 + 1 pt gap = the ring's inner 14). Drawn with
        // circular corners just under half its height: `Capsule` and continuous corners flatten a stroked pill.
        .padding(5)
        .background(RoundedRectangle(cornerRadius: 17, style: .circular).inset(by: 1)
            .strokeBorder(Color(nsColor: .keyboardFocusIndicatorColor), lineWidth: 3).opacity(focus.keyboardFocused ? 1 : 0))
        .padding(-5)
    }

    /// R28a, for line breaking only: a number stays with the word after it ("21 hours", "40% done") and a month with its
    /// day. The value's text, the accessibility text and the contract bytes are unchanged.
    static func keepingNumbersWithUnits(_ string: String) -> String {
        string.replacing(#/(\d%?) /#) { "\($0.1)\u{00A0}" }.replacing(#/\b([A-Z][a-z]{2}) (\d)/#) { "\($0.1)\u{00A0}\($0.2)" }
    }
}

/// A2's signs (ok, busy, wait), A1's headline disc (marked), and A1's row marks. Shape and glyph carry the state as
/// well as the colour; Increase Contrast adds a ring. The busy arrows turn while shown (R28c), still under Reduce Motion.
struct BrainBarMenuMark: View {
    let mark: BrainBarMenuPresentation.Mark
    let size: CGFloat
    let accessibility: BrainBarMenuAccessibility

    var body: some View {
        switch mark {
        case .ok: disc(.systemGreen, Image(systemName: "checkmark"))
        case .busy: disc(.systemBlue, Image(systemName: "arrow.triangle.2.circlepath"), turning: !accessibility.reduceMotion)
        case .wait: disc(.systemGray, Image(systemName: "ellipsis"))
        case .marked: disc(.systemRed, Image(systemName: "exclamationmark"))
        case .alertCircle:
            Image(systemName: "exclamationmark.circle.fill").font(.system(size: 15))
                .foregroundStyle(.white, Color(nsColor: .systemRed))
        case .checkmark:
            Image(systemName: "checkmark").font(.system(size: 12, weight: .bold)).foregroundStyle(Color(nsColor: .systemGreen))
        }
    }

    private func disc(_ color: NSColor, _ glyph: Image, turning: Bool = false) -> some View {
        Circle().fill(Color(nsColor: color))
            .overlay(Circle().strokeBorder(Color.primary.opacity(0.55), lineWidth: accessibility.increaseContrast ? 1 : 0))
            .overlay {
                // One even turn every 2 s, linear: constant motion, slow enough to read as work going on, not alarm.
                TimelineView(.animation(paused: !turning)) { context in
                    glyph.font(.system(size: size * 0.48, weight: .bold)).foregroundStyle(.white)
                        .rotationEffect(.degrees(turning ? context.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 2) * 180 : 0))
                }
            }
            .frame(width: size, height: size)
    }
}

/// The one action: a filled rounded button in the accent colour, regular weight as the system draws one. Its label
/// wraps and the button grows rather than cut a word. Pressed only darkens it, as a native button does.
struct BrainBarMenuActionButtonStyle: ButtonStyle {
    let accessibility: BrainBarMenuAccessibility

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13)).foregroundStyle(.white).multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 12).padding(.vertical, 5).frame(minHeight: 26)
            .background(RoundedRectangle(cornerRadius: 13, style: .circular).fill(Color.accentColor).brightness(configuration.isPressed ? -0.09 : 0))
            .overlay(RoundedRectangle(cornerRadius: 13, style: .circular).strokeBorder(Color.primary.opacity(0.7), lineWidth: accessibility.increaseContrast ? 1 : 0))
            .contentShape(RoundedRectangle(cornerRadius: 13, style: .circular))
            .transaction { $0.animation = nil }
    }
}
