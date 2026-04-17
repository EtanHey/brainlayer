import AppKit
import SwiftUI

// MARK: - Entry point

struct BrainBarCommandBar: View {
    let viewModel: QuickCaptureViewModel?

    var body: some View {
        if let viewModel {
            BrainBarCommandBarReady(viewModel: viewModel)
        } else {
            BrainBarCommandBarPlaceholder()
        }
    }
}

// MARK: - Ready state (viewModel present)

private struct BrainBarCommandBarReady: View {
    @ObservedObject var viewModel: QuickCaptureViewModel

    var body: some View {
        HStack(spacing: 10) {
            ModePillPair(viewModel: viewModel)

            Image(systemName: viewModel.mode == .capture ? "square.and.pencil" : "magnifyingglass")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)

            CommandBarInput(viewModel: viewModel)
                .frame(maxWidth: .infinity)

            CommandBarTrailingHint(viewModel: viewModel)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .frame(height: 40)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor).opacity(0.85))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(borderColor, lineWidth: 1)
        )
        .animation(.easeInOut(duration: 0.14), value: viewModel.mode)
        .animation(.easeInOut(duration: 0.18), value: viewModel.feedback)
    }

    private var borderColor: Color {
        switch viewModel.feedback {
        case .idle:
            return Color.white.opacity(0.08)
        case .success:
            return Color(nsColor: .systemGreen).opacity(0.55)
        case .error:
            return Color(nsColor: .systemRed).opacity(0.55)
        }
    }
}

// MARK: - Placeholder state (db not ready)

private struct BrainBarCommandBarPlaceholder: View {
    var body: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small)
            Text("Warming memory…")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.tertiary)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(height: 40)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor).opacity(0.45))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(Color.white.opacity(0.06), lineWidth: 1)
        )
    }
}

// MARK: - Mode pill pair

private struct ModePillPair: View {
    @ObservedObject var viewModel: QuickCaptureViewModel

    var body: some View {
        HStack(spacing: 2) {
            ModePill(title: "Capture", isActive: viewModel.mode == .capture) {
                viewModel.setMode(.capture)
            }
            ModePill(title: "Search", isActive: viewModel.mode == .search) {
                viewModel.setMode(.search)
            }
        }
        .padding(3)
        .background(
            Capsule().fill(Color(nsColor: .windowBackgroundColor).opacity(0.6))
        )
        .overlay(
            Capsule().strokeBorder(Color.white.opacity(0.05), lineWidth: 0.5)
        )
    }
}

private struct ModePill: View {
    let title: String
    let isActive: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 11, weight: .semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(
                    Capsule().fill(isActive ? Color.accentColor : Color.clear)
                )
                .foregroundStyle(isActive ? Color.white : Color.secondary)
                .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .focusable(false)
    }
}

// MARK: - Input (NSTextField bridge)

private struct CommandBarInput: NSViewRepresentable {
    @ObservedObject var viewModel: QuickCaptureViewModel

    final class Coordinator: NSObject, NSTextFieldDelegate {
        var parent: CommandBarInput
        var lastFocusRequestCount: Int = 0

        init(parent: CommandBarInput) {
            self.parent = parent
        }

        func controlTextDidChange(_ notification: Notification) {
            guard let field = notification.object as? NSTextField else { return }
            parent.viewModel.handleInputChange(field.stringValue)
        }

        func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            switch commandSelector {
            case #selector(NSResponder.insertTab(_:)):
                parent.viewModel.handleInputTab()
                return true
            case #selector(NSResponder.moveUp(_:)):
                parent.viewModel.handleInputMove(.up)
                return true
            case #selector(NSResponder.moveDown(_:)):
                parent.viewModel.handleInputMove(.down)
                return true
            case #selector(NSResponder.insertNewline(_:)),
                 #selector(NSResponder.insertNewlineIgnoringFieldEditor(_:)):
                let modifiers = NSApp.currentEvent?.modifierFlags.intersection(.deviceIndependentFlagsMask) ?? []
                parent.viewModel.handleInputReturn(modifiers: modifiers)
                return true
            default:
                return false
            }
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeNSView(context: Context) -> NSTextField {
        let field = NSTextField(frame: .zero)
        field.isBordered = false
        field.isBezeled = false
        field.drawsBackground = false
        field.focusRingType = .none
        field.font = .systemFont(ofSize: 13, weight: .regular)
        field.cell?.wraps = false
        field.cell?.isScrollable = true
        field.cell?.usesSingleLineMode = true
        field.cell?.lineBreakMode = .byClipping
        field.maximumNumberOfLines = 1
        field.placeholderString = viewModel.placeholderText
        field.stringValue = viewModel.inputText
        field.delegate = context.coordinator
        return field
    }

    func updateNSView(_ nsView: NSTextField, context: Context) {
        context.coordinator.parent = self
        if nsView.stringValue != viewModel.inputText {
            nsView.stringValue = viewModel.inputText
        }
        if nsView.placeholderString != viewModel.placeholderText {
            nsView.placeholderString = viewModel.placeholderText
        }
        if context.coordinator.lastFocusRequestCount != viewModel.focusRequestCount {
            context.coordinator.lastFocusRequestCount = viewModel.focusRequestCount
            DispatchQueue.main.async {
                nsView.window?.makeFirstResponder(nsView)
            }
        }
    }
}

// MARK: - Trailing hint / feedback

private struct CommandBarTrailingHint: View {
    @ObservedObject var viewModel: QuickCaptureViewModel

    var body: some View {
        Group {
            if !viewModel.feedback.isIdle {
                HStack(spacing: 4) {
                    Image(systemName: feedbackSymbol)
                        .font(.system(size: 11, weight: .semibold))
                    Text(viewModel.feedback.message)
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(1)
                }
                .foregroundStyle(viewModel.feedback.tintColor)
            } else {
                Text(keyboardHint)
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
        }
        .fixedSize(horizontal: true, vertical: false)
    }

    private var feedbackSymbol: String {
        switch viewModel.feedback {
        case .idle: return ""
        case .success: return "checkmark.circle.fill"
        case .error: return "exclamationmark.triangle.fill"
        }
    }

    private var keyboardHint: String {
        switch viewModel.mode {
        case .capture: return "⏎ store · ⇥ switch"
        case .search: return "⏎ open · ⌘⏎ capture"
        }
    }
}

// MARK: - Results overlay (floats above tab content)

struct BrainBarCommandBarResultsOverlay: View {
    let viewModel: QuickCaptureViewModel?

    var body: some View {
        if let viewModel {
            BrainBarCommandBarResultsOverlayGate(viewModel: viewModel)
        }
    }
}

private struct BrainBarCommandBarResultsOverlayGate: View {
    @ObservedObject var viewModel: QuickCaptureViewModel

    var body: some View {
        if shouldShow {
            BrainBarCommandBarResultsOverlayReady(viewModel: viewModel)
                .transition(.opacity.combined(with: .move(edge: .top)))
        }
    }

    private var shouldShow: Bool {
        guard viewModel.mode == .search else { return false }
        return !viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}

private struct BrainBarCommandBarResultsOverlayReady: View {
    @ObservedObject var viewModel: QuickCaptureViewModel

    var body: some View {
        Group {
            if viewModel.results.isEmpty {
                emptyState
            } else {
                resultsList
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(.regularMaterial)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.28), radius: 18, y: 8)
    }

    @ViewBuilder
    private var emptyState: some View {
        HStack(spacing: 10) {
            Image(systemName: viewModel.feedback.isIdle ? "magnifyingglass" : "exclamationmark.triangle.fill")
                .foregroundStyle(viewModel.feedback.isIdle ? Color.secondary : viewModel.feedback.tintColor)
            if viewModel.feedback.isIdle {
                Text("No matches yet for \"\(viewModel.inputText)\"")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            } else {
                Text(viewModel.feedback.message)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(viewModel.feedback.tintColor)
                    .lineLimit(1)
            }
            Spacer()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var resultsList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 4) {
                ForEach(viewModel.results) { row in
                    BrainBarCommandBarResultRow(
                        row: row,
                        isSelected: row.id == viewModel.selectedResultID,
                        isCopied: row.id == viewModel.copiedResultID,
                        onSelect: { viewModel.selectResult(id: row.id) },
                        onActivate: { viewModel.copyResultToClipboard(id: row.id) }
                    )
                }
            }
            .padding(10)
        }
        .frame(maxHeight: 260)
    }
}

private struct BrainBarCommandBarResultRow: View {
    let row: QuickCaptureSearchRow
    let isSelected: Bool
    let isCopied: Bool
    let onSelect: () -> Void
    let onActivate: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(row.title)
                .font(.system(size: 13, weight: .medium))
                .lineLimit(2)
                .foregroundStyle(.primary)
            Text(row.metadata)
                .font(.system(size: 10, weight: .regular, design: .monospaced))
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(rowBackground)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(
                    isSelected ? Color.accentColor.opacity(0.45) : Color.clear,
                    lineWidth: 1
                )
        )
        .overlay(alignment: .topTrailing) {
            if isCopied {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Color(nsColor: .systemGreen))
                    .padding(6)
                    .transition(.scale.combined(with: .opacity))
            }
        }
        .contentShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .onTapGesture(perform: onSelect)
        .onTapGesture(count: 2, perform: onActivate)
        .animation(.easeInOut(duration: 0.14), value: isCopied)
    }

    private var rowBackground: Color {
        if isCopied {
            return Color(nsColor: .systemGreen).opacity(0.16)
        }
        if isSelected {
            return Color.accentColor.opacity(0.14)
        }
        return Color.clear
    }
}
