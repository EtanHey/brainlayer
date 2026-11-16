import AppKit
import SwiftUI

/// The panel card's contents (R1.5 change 2): the title row with its mark, the lines, the rows, and the action under a
/// full-width hairline. Opened from the keyboard, the action takes focus (change 4).
struct NDStationPanelBody: View {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    let model: NDStationPopover.Model
    var waiting = false
    var perform: (BrainBarMenuPresentation.ActionKind) -> Void = { _ in }
    @Environment(\.ndPanelFocusAction) private var focusAction
    @FocusState private var actionFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                mark
                Text(model.title).font(NDType.bodyStrong).foregroundStyle(ND.t1).fixedSize(horizontal: false, vertical: true)
            }
            .padding(.bottom, 6)
            ForEach(Array(Self.paragraphs(model.lines).enumerated()), id: \.offset) { index, line in
                // R1.7 item 10: body 12 pt in the secondary colour, at most two lines a paragraph.
                Text(keepingUnits(line)).font(NDType.panelBody).foregroundStyle(ND.t2)
                    .fixedSize(horizontal: false, vertical: true).padding(.top, index == 0 ? 0 : 4)
            }
            if !model.rows.isEmpty {
                VStack(spacing: 0) {
                    ForEach(model.rows) { line in
                        HStack {
                            Text(line.label).font(NDType.panelBody).foregroundStyle(ND.t2).fixedSize(horizontal: false, vertical: true)
                            Spacer(minLength: line.value.isEmpty ? 0 : 12)
                            if !line.value.isEmpty {
                                // R1.7 item 10: the value in primary semibold (a fault or a notice keeps its colour).
                                Text(line.value).font(NDType.panelValue)
                                    .foregroundStyle(line.tone == .fault ? ND.redText : line.tone == .warn ? ND.orangeText : ND.t1)
                            }
                        }
                        .frame(minHeight: Self.rowHeight)
                    }
                }
                .padding(.top, 8)
            }
            if let action = model.action {
                ND.hair2.frame(height: 0.5).padding(.horizontal, -14).padding(.top, 12)
                HStack {
                    Spacer()
                    actionButton(action)
                }
                .padding(.top, 10)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .onAppear { if focusAction, model.action != nil { actionFocused = true } }
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

    @ViewBuilder private var mark: some View {
        switch model.mark {
        case .fault: Image(systemName: "exclamationmark.circle.fill").font(.system(size: 13)).foregroundStyle(ND.red)
        case .notice: Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 13)).foregroundStyle(ND.orange)
        case let .symbol(name): Image(systemName: name).font(.system(size: 13)).foregroundStyle(ND.t2)
        }
    }

    @ViewBuilder private func actionButton(_ action: BrainBarMenuPresentation.Action) -> some View {
        if action.kind == .copyReport {
            NDCopyReportButton(label: NDPipelineChart.shortLabel(action), spoken: action.label) { perform(action.kind) }
                .focused($actionFocused)
        } else {
            NDWaitingSwap(waiting: waiting && [.reconnectDrive, .connectDrive].contains(action.kind)) {
                NDButton(label: NDPipelineChart.shortLabel(action), filled: NDButtonPolicy.filled(action), small: true,
                         accessibility: action.label) { perform(action.kind) }
                    .focused($actionFocused)
            } line: {
                Text(Say.waitingForBrowser).font(NDType.body).foregroundStyle(ND.t2)
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
/// cross-fade of its label and no other motion.
struct NDCopyReportButton: View {
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    let label: String
    let spoken: String
    let action: () -> Void
    @State private var copied = false
    @Environment(\.ndPreviewCopied) private var previewCopied
    static let copiedFor: Duration = .milliseconds(1500)
    static let fade = Animation.easeInOut(duration: 0.12)

    /// What the button reads: its label at rest, "Copied" with a tick for 1.5 s after a click.
    static func face(copied: Bool, label: String) -> (text: String, tick: Bool) {
        copied ? (Say.copiedLabel, true) : (label, false)
    }

    var body: some View {
        let face = Self.face(copied: copied || previewCopied, label: label)
        Button {
            action()
            withAnimation(Self.fade) { copied = true }
            Task { @MainActor in
                try? await Task.sleep(for: Self.copiedFor)
                withAnimation(Self.fade) { copied = false }
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
            }
        }
        .buttonStyle(NDButtonStyle(small: true))
        .ndKeyboardFocus { action() }
        .accessibilityLabel(Text(copied ? Say.copiedLabel : spoken))
    }
}
