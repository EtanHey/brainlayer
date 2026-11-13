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

    var body: some View {
        HStack(alignment: .top, spacing: BrainBarMenuMetrics.gap) {
            Color.clear.frame(width: BrainBarMenuMetrics.gutter, height: gutterHeight).overlay { gutter }
            content.frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, row.kind == .headline ? 0 : (gutterHeight - 16) / 2)
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
        default: ([.headline, .headlineDetail].contains(previous) ? 8 : previous == .separator ? 6 : 3, lastInBlock ? 9 : 3)
        }
    }

    private var gutterHeight: CGFloat {
        switch row.mark {
        case _ where row.kind == .headline: 26
        case .ok?, .busy?, .wait?: 20
        default: 16
        }
    }

    @ViewBuilder private var gutter: some View {
        if let mark = row.mark {
            BrainBarMenuMark(mark: mark, size: gutterHeight, accessibility: accessibility)
        } else if row.spinner {
            if accessibility.reduceMotion {  // the spinner becomes a static mark
                Image(systemName: "arrow.triangle.2.circlepath").font(.system(size: 11, weight: .semibold)).foregroundStyle(secondary)
            } else {
                ProgressView().controlSize(.mini)
            }
        }
    }

    @ViewBuilder private var content: some View {
        switch row.kind {
        case .headline:
            Text(row.strings.joined(separator: " ")).font(.system(size: 15, weight: .semibold)).frame(minHeight: 26)
        case .needsYou: needsYou
        default:
            VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(row.strings.enumerated()), id: \.offset) { Text($0.element).foregroundStyle(secondary) }
                if let percent = row.progressPercent {
                    ProgressView(value: Double(min(max(percent, 0), 100)), total: 100).progressViewStyle(.linear)
                        .padding(.top, 6).animation(nil, value: percent)
                }
            }
        }
    }

    /// A Needs-you row: title, its facts, then the presenter's two hand-off lines (it appends them to a row whose
    /// action is Copy report), then the one blue button.
    @ViewBuilder private var needsYou: some View {
        let handoff = row.action?.kind == .copyReport && row.strings.count >= 3 ? 2 : 0
        let facts = row.strings.dropFirst().dropLast(handoff)
        VStack(alignment: .leading, spacing: 0) {
            Text(row.strings.first ?? "").fontWeight(.semibold)
            ForEach(Array(facts.enumerated()), id: \.offset) { Text($0.element).foregroundStyle(secondary).padding(.top, 2) }
            if handoff > 0 {
                VStack(alignment: .leading, spacing: 0) { ForEach(row.strings.suffix(handoff), id: \.self) { Text($0) } }.padding(.top, 6)
            }
            if let action = row.action {
                HStack(spacing: 0) {
                    Button(action.label) { perform(action.kind) }.buttonStyle(BrainBarMenuActionButtonStyle(accessibility: accessibility))
                }
                // The system focus ring, concentric with the button (radius 13 + 1 pt gap = the ring's inner 14). Drawn
                // with circular corners just under half its height: `Capsule` and continuous corners flatten a stroked pill.
                .padding(5)
                .background(RoundedRectangle(cornerRadius: 17, style: .circular).inset(by: 1).strokeBorder(Color(nsColor: .keyboardFocusIndicatorColor), lineWidth: 3)
                    .opacity(focus.keyboardFocused ? 1 : 0))
                .padding(-5)
                    .padding(.top, 9)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}

/// A2's signs (ok, busy, wait), A1's headline disc (marked), and A1's row marks. Shape and glyph carry the state as
/// well as the colour; Increase Contrast adds a ring.
struct BrainBarMenuMark: View {
    let mark: BrainBarMenuPresentation.Mark
    let size: CGFloat
    let accessibility: BrainBarMenuAccessibility

    var body: some View {
        switch mark {
        case .ok: disc(.systemGreen, "checkmark")
        case .busy: disc(.systemBlue, "arrow.triangle.2.circlepath")
        case .wait: disc(.systemGray, "ellipsis")
        case .marked: disc(.systemRed, "exclamationmark")
        case .alertCircle:
            Image(systemName: "exclamationmark.circle.fill").font(.system(size: 15))
                .foregroundStyle(.white, Color(nsColor: .systemRed))
        case .checkmark:
            Image(systemName: "checkmark").font(.system(size: 12, weight: .bold)).foregroundStyle(Color(nsColor: .systemGreen))
        }
    }

    private func disc(_ color: NSColor, _ glyph: String) -> some View {
        Circle().fill(Color(nsColor: color))
            .overlay(Circle().strokeBorder(Color.primary.opacity(0.55), lineWidth: accessibility.increaseContrast ? 1 : 0))
            .overlay(Image(systemName: glyph).font(.system(size: size * 0.48, weight: .bold)).foregroundStyle(.white))
            .frame(width: size, height: size)
    }
}

/// The one action: a filled capsule in the accent colour. Pressed darkens it and, without Reduce Motion, settles it to
/// 0.97 at once (nothing animates in this menu but a bar and a spinner).
struct BrainBarMenuActionButtonStyle: ButtonStyle {
    let accessibility: BrainBarMenuAccessibility

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 13, weight: .medium)).foregroundStyle(.white).lineLimit(1)
            .padding(.horizontal, 12).frame(height: 26)
            .background(RoundedRectangle(cornerRadius: 13, style: .circular).fill(Color.accentColor).brightness(configuration.isPressed ? -0.09 : 0))
            .overlay(RoundedRectangle(cornerRadius: 13, style: .circular).strokeBorder(Color.primary.opacity(0.7), lineWidth: accessibility.increaseContrast ? 1 : 0))
            .scaleEffect(configuration.isPressed && !accessibility.reduceMotion ? 0.97 : 1)
            .contentShape(RoundedRectangle(cornerRadius: 13, style: .circular))
            .transaction { $0.animation = nil }
    }
}
