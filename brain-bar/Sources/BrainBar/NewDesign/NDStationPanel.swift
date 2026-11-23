import AppKit
import SwiftUI

/// The panel card's contents (R1.5 change 2): the title row with its mark, the lines, the rows, and the action under a
/// full-width hairline. Opened from the keyboard, the action takes focus (change 4).
struct NDStationPanelBody: View {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    let model: NDStationPopover.Model
    var waiting = false
    var compactRows = false
    var compactAction = false
    var allowsWideLayout = false
    var perform: (BrainBarMenuPresentation.ActionKind) -> Void = { _ in }
    @Environment(\.ndPanelFocusAction) private var focusAction
    @FocusState private var actionFocused: Bool

    var body: some View {
        ViewThatFits(in: .horizontal) {
            if allowsWideLayout { wideBody }
            stackedBody
        }
        .onAppear { if focusAction, model.action != nil { actionFocused = true } }
    }

    private var heading: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            mark
            Text(model.title).font(NDType.bodyStrong).foregroundStyle(ND.t1).fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Widening also reflows the readouts; a vertical stack has a minimum height even at the widest width.
    private var wideBody: some View {
        HStack(alignment: .top, spacing: 12) {
            heading.frame(width: 220, alignment: .topLeading)
            if !model.lines.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(Self.paragraphs(model.lines).enumerated()), id: \.offset) { _, line in
                        Text(keepingUnits(line)).font(NDType.panelBody).foregroundStyle(ND.t2)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
            } else {
                Spacer(minLength: 0)
            }
            if !model.rows.isEmpty { readoutRows.frame(width: model.action == nil ? 180 : 140) }
            if let action = model.action {
                ND.hair2.frame(width: 0.5, height: 36)
                actionButton(action).frame(width: max(220, NDPipelineChart.actionWidth(action)))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .frame(minWidth: 760, idealWidth: 760, maxWidth: .infinity)
    }

    private var stackedBody: some View {
        VStack(alignment: .leading, spacing: 0) {
            heading.padding(.bottom, 6)
            ForEach(Array(Self.paragraphs(model.lines).enumerated()), id: \.offset) { index, line in
                // R1.7 item 10: body 12 pt in the secondary colour, at most two lines a paragraph.
                // R1.12 (§ B): 2 pt between paragraphs, so the top-lane panels fit the room above their stations.
                Text(keepingUnits(line)).font(NDType.panelBody).foregroundStyle(ND.t2)
                    .fixedSize(horizontal: false, vertical: true).padding(.top, index == 0 ? 0 : 2)
            }
            if !model.rows.isEmpty {
                readoutRows
                .padding(.top, compactRows ? 2 : 8)
            }
            if let action = model.action {
                // R1.12 (§ B): 10 and 8 pt around the hairline, so a fault panel fits the room above its station.
                ND.hair2.frame(height: 0.5).padding(.horizontal, -14).padding(.top, compactAction ? 4 : 10)
                actionButton(action)
                    .padding(.top, 8)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)  // R1.12 (§ B)
    }

    private var readoutRows: some View {
        VStack(spacing: 0) {
            ForEach(model.rows) { line in
                HStack {
                    Text(line.label).font(NDType.panelBody).foregroundStyle(ND.t2).fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: line.value.isEmpty ? 0 : 12)
                    if !line.value.isEmpty {
                        Text(line.value).font(NDType.panelValue)
                            .foregroundStyle(line.tone == .fault ? ND.redText : line.tone == .warn ? ND.orangeText : ND.t1)
                    }
                }
                .frame(minHeight: Self.rowHeight)
            }
        }
    }

    /// R1.7 item 10: a row at 12 pt is 18 pt high.
    static let rowHeight: CGFloat = 18

    /// R1.7 item 10: one paragraph per sentence, so a paragraph runs to at most two lines (the words are the deck's).
    static func paragraphs(_ lines: [String]) -> [String] {
        lines.flatMap { line in
            line.split(separator: ". ", omittingEmptySubsequences: true).enumerated().map { index, part in
                let parts = line.components(separatedBy: ". ").count
                return index < parts - 1 ? String(part) + "." : String(part)
            }
        }
    }

    /// R1.8 item 6: the line that replaces Reconnect while the browser asks: a 12 pt spinner, then the words.
    static var waitingLine: some View {
        HStack(alignment: .center, spacing: 6) {
            ProgressView().progressViewStyle(.circular).controlSize(.mini).frame(width: 12, height: 12)
            Text(Say.waitingForBrowser).font(NDType.body).foregroundStyle(ND.t2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private var mark: some View {
        switch model.mark {
        case .fault: Image(systemName: "exclamationmark.circle.fill").font(.system(size: 13)).foregroundStyle(ND.red)
        case .notice: Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 13)).foregroundStyle(ND.orange)
        // R1.10: the two job stations' glyphs are drawn (no SF Symbol is a broom), the same drawing as on the chart.
        case let .symbol(name) where name == "moon" || name == "broom":
            Canvas { ctx, size in
                NDPipelineChart.jobGlyph(ctx, name == "moon" ? .jobs : .weekly, at: CGPoint(x: size.width / 2, y: size.height / 2),
                                         size: 13, color: ND.t2)
            }
            .frame(width: 14, height: 14)
            .alignmentGuide(.firstTextBaseline) { d in d.height * 0.8 }
        case let .symbol(name): Image(systemName: name).font(.system(size: 13)).foregroundStyle(ND.t2)
        }
    }

    @ViewBuilder private func actionButton(_ action: BrainBarMenuPresentation.Action) -> some View {
        if action.kind == .copyReport {
            HStack {
                Spacer()
                NDCopyReportButton(label: NDPipelineChart.shortLabel(action), spoken: action.label) { perform(action.kind) }
                    .focused($actionFocused)
            }
        } else {
            // R1.8 item 6: the waiting line takes the whole footer, left-aligned, 13 pt secondary, a small spinner first;
            // one line where it fits (it wrapped into two right-aligned lines in the button's corner).
            NDWaitingSwap(waiting: waiting, alignment: .leading) {
                HStack {
                    Spacer()
                    NDButton(label: NDPipelineChart.shortLabel(action), filled: NDButtonPolicy.filled(action), small: true,
                             accessibility: action.label) { perform(action.kind) }
                        .focused($actionFocused)
                }
            } line: {
                Self.waitingLine
            }
        }
    }
}

private struct NDPreviewCopiedKey: EnvironmentKey { static let defaultValue = false }
extension EnvironmentValues {
    /// Render harness only: draw Copy report in its "Copied" state.
    var ndPreviewCopied: Bool {
        get { self[NDPreviewCopiedKey.self] }
        set { self[NDPreviewCopiedKey.self] = newValue }
    }
}

/// R1.5 change 3: Copy report, which shows a tick and "Copied" for 1.5 s after a click, at the same size, with a 120 ms
/// change of its label and no other motion.
/// R1.13.1: the change is two halves, never a cross-fade: the old face fades out (60 ms), the face swaps with no
/// animation, the new face fades in (60 ms), so no frame shows both labels (the M1 capture caught one that did). Under
/// Reduce Motion the face swaps at once.
struct NDCopyReportButton: View {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    let label: String
    let spoken: String
    let action: () -> Void
    @State private var copied = false
    @State private var faceOpacity: Double = 1
    @Environment(\.ndPreviewCopied) private var previewCopied
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    static let copiedFor: Duration = .milliseconds(1500)
    static let half = Animation.easeInOut(duration: 0.06)

    /// The steps of one change of face: fade out, swap, fade in; or the swap alone under Reduce Motion.
    enum Step: Equatable { case fadeOut, swap, fadeIn }
    static func steps(reduceMotion: Bool) -> [Step] { reduceMotion ? [.swap] : [.fadeOut, .swap, .fadeIn] }

    private func show(copied value: Bool) {
        guard !reduceMotion else { copied = value; return }
        withAnimation(Self.half) { faceOpacity = 0 } completion: {
            var swap = Transaction()
            swap.disablesAnimations = true
            withTransaction(swap) { copied = value }
            withAnimation(Self.half) { faceOpacity = 1 }
        }
    }

    /// What the button reads: its label at rest, "Copied" with a tick for 1.5 s after a click.
    static func face(copied: Bool, label: String) -> (text: String, tick: Bool) {
        copied ? (Say.copiedLabel, true) : (label, false)
    }

    var body: some View {
        let face = Self.face(copied: copied || previewCopied, label: label)
        Button {
            action()
            show(copied: true)
            Task { @MainActor in
                try? await Task.sleep(for: Self.copiedFor)
                show(copied: false)
            }
        } label: {
            ZStack {
                // The wider of the two faces reserves the width, so the button keeps its size.
                HStack(spacing: 4) { Image(systemName: "checkmark").font(.system(size: 12, weight: .semibold)); Text(Say.copiedLabel) }
                    .hidden()
                Text(label).hidden()
                HStack(spacing: 4) {
                    if face.tick { Image(systemName: "checkmark").font(.system(size: 12, weight: .semibold)) }
                    Text(face.text)
                }
                .opacity(faceOpacity)
            }
        }
        .buttonStyle(NDButtonStyle(small: true))
        .focusable(false)  // R1.8 item 4: one stop, the wrapper's
        .ndKeyboardFocus { action() }
        .accessibilityLabel(Text(copied ? Say.copiedLabel : spoken))
    }
}
