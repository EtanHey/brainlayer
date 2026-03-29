import AppKit
import SwiftUI

enum QuickCaptureFeedback: Equatable {
    case idle
    case success(String)
    case error(String)

    var isIdle: Bool {
        if case .idle = self {
            return true
        }
        return false
    }

    var message: String {
        switch self {
        case .idle:
            return ""
        case .success(let message), .error(let message):
            return message
        }
    }

    var tintColor: Color {
        switch self {
        case .idle:
            return .clear
        case .success:
            return Color(nsColor: .systemGreen)
        case .error:
            return Color(nsColor: .systemRed)
        }
    }
}

struct QuickCaptureSearchRow: Identifiable, Equatable {
    let id: String
    let title: String
    let metadata: String
}

@MainActor
final class QuickCaptureViewModel: ObservableObject {
    @Published var inputText = ""
    @Published var mode: QuickCapturePanelState.Mode
    @Published private(set) var feedback: QuickCaptureFeedback = .idle
    @Published private(set) var results: [QuickCaptureSearchRow] = []
    @Published private(set) var confirmationFlashCount = 0
    @Published private(set) var focusRequestCount = 0

    private let db: BrainDatabase
    private let panelState: QuickCapturePanelState

    init(db: BrainDatabase, panelState: QuickCapturePanelState) {
        self.db = db
        self.panelState = panelState
        mode = panelState.mode
    }

    var placeholderText: String {
        switch mode {
        case .capture:
            return "Capture an idea. Press Return to store."
        case .search:
            return "Search memory. Press Return to run."
        }
    }

    var statusText: String {
        switch mode {
        case .capture:
            return feedback.message
        case .search:
            if inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "Type to search BrainLayer"
            }
            if results.isEmpty {
                return "No matches yet"
            }
            return results.count == 1 ? "1 result" : "\(results.count) results"
        }
    }

    func setMode(_ newMode: QuickCapturePanelState.Mode) {
        guard mode != newMode else { return }
        mode = newMode
        panelState.switchMode(newMode)
        feedback = .idle
        if newMode == .capture {
            results = []
        }
    }

    func submit() {
        switch mode {
        case .capture:
            submitCapture()
        case .search:
            submitSearch()
        }
    }

    func dismiss() {
        panelState.dismiss()
        mode = panelState.mode
        inputText = ""
        results = []
        feedback = .idle
    }

    func panelDidAppear() {
        focusRequestCount += 1
    }

    private func submitCapture() {
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            feedback = .error("Content cannot be empty")
            return
        }
        
        do {
            _ = try QuickCaptureController.capture(
                db: db,
                content: inputText,
                tags: []
            )
            inputText = ""
            results = []
            feedback = .success("Stored in BrainLayer")
            confirmationFlashCount += 1
        } catch {
            feedback = .error(error.localizedDescription)
        }
    }

    private func submitSearch() {
        do {
            let searchResult = try QuickCaptureController.search(
                db: db,
                query: inputText,
                limit: 8
            )
            results = searchResult.results.map { result in
                let rawID = (result["chunk_id"] as? String) ?? UUID().uuidString
                let title = (result["content"] as? String) ?? "Untitled result"
                let createdAt = (result["created_at"] as? String) ?? "unknown time"
                let importance = (result["importance"] as? Double).map { String(format: "imp %.0f", $0) } ?? "imp ?"
                return QuickCaptureSearchRow(
                    id: rawID,
                    title: title,
                    metadata: "\(importance) • \(createdAt)"
                )
            }
            feedback = .idle
        } catch {
            results = []
            feedback = .error(error.localizedDescription)
        }
    }
}

final class QuickCapturePanel: NSPanel {
    var onEscape: (() -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }

    override func cancelOperation(_ sender: Any?) {
        onEscape?()
    }
}

@MainActor
final class QuickCapturePanelController {
    private let panelState = QuickCapturePanelState()
    private let database: BrainDatabase
    private let panel: QuickCapturePanel
    private let viewModel: QuickCaptureViewModel

    init(dbPath: String) {
        database = BrainDatabase(path: dbPath)
        viewModel = QuickCaptureViewModel(db: database, panelState: panelState)
        panel = QuickCapturePanel(
            contentRect: NSRect(x: 0, y: 0, width: 540, height: 360),
            styleMask: [.titled, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        configurePanel()
    }

    deinit {
        database.close()
    }

    func toggle() {
        if panel.isVisible {
            dismiss()
        } else {
            show()
        }
    }

    func show(mode: QuickCapturePanelState.Mode? = nil) {
        if let mode {
            viewModel.setMode(mode)
        }

        centerPanel()
        panel.alphaValue = 0
        panel.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        viewModel.panelDidAppear()
        panelState.show()

        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.16
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            panel.animator().alphaValue = 1
        }
    }

    func dismiss() {
        guard panel.isVisible else {
            viewModel.dismiss()
            return
        }

        viewModel.dismiss()
        NSAnimationContext.runAnimationGroup { context in
            context.duration = 0.14
            context.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            panel.animator().alphaValue = 0
        } completionHandler: { [weak self] in
            guard let self else { return }
            Task { @MainActor in
                self.panel.orderOut(nil)
                self.panel.alphaValue = 1
            }
        }
    }

    private func configurePanel() {
        panel.titleVisibility = .hidden
        panel.titlebarAppearsTransparent = true
        panel.isMovableByWindowBackground = true
        panel.isFloatingPanel = true
        panel.level = .statusBar
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        panel.standardWindowButton(.closeButton)?.isHidden = true
        panel.standardWindowButton(.miniaturizeButton)?.isHidden = true
        panel.standardWindowButton(.zoomButton)?.isHidden = true
        panel.isReleasedWhenClosed = false
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true
        panel.onEscape = { [weak self] in
            self?.dismiss()
        }
        panel.contentViewController = NSHostingController(
            rootView: QuickCapturePanelView(viewModel: viewModel)
        )
    }

    private func centerPanel() {
        let screenFrame = NSScreen.main?.visibleFrame ?? NSScreen.screens.first?.visibleFrame
        guard let screenFrame else { return }
        let panelFrame = panel.frame
        let origin = CGPoint(
            x: screenFrame.midX - panelFrame.width / 2,
            y: screenFrame.midY - panelFrame.height / 2
        )
        panel.setFrameOrigin(origin)
    }
}

struct QuickCapturePanelView: View {
    @ObservedObject var viewModel: QuickCaptureViewModel
    @FocusState private var isInputFocused: Bool
    @State private var flashOpacity = 0.0

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18)
                .fill(Color(nsColor: .windowBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 18)
                        .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
                )
                .shadow(color: .black.opacity(0.18), radius: 30, y: 10)

            RoundedRectangle(cornerRadius: 18)
                .fill(Color(nsColor: .systemGreen).opacity(flashOpacity))
                .allowsHitTesting(false)

            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .center, spacing: 10) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Quick Capture")
                            .font(.system(size: 18, weight: .semibold, design: .rounded))
                        Text(viewModel.mode == .capture ? "Store a thought fast" : "Search BrainLayer instantly")
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    QuickCaptureModeButton(
                        title: "Capture",
                        shortcut: "1",
                        isSelected: viewModel.mode == .capture
                    ) {
                        viewModel.setMode(.capture)
                    }

                    QuickCaptureModeButton(
                        title: "Search",
                        shortcut: "2",
                        isSelected: viewModel.mode == .search
                    ) {
                        viewModel.setMode(.search)
                    }
                }

                TextField("", text: $viewModel.inputText, prompt: Text(viewModel.placeholderText))
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(Color(nsColor: .controlBackgroundColor))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .strokeBorder(borderColor, lineWidth: 1)
                    )
                    .font(.system(size: 14, weight: .medium))
                    .focused($isInputFocused)
                    .onSubmit {
                        viewModel.submit()
                    }

                HStack(spacing: 8) {
                    Image(systemName: statusSymbol)
                        .foregroundStyle(viewModel.feedback.isIdle ? Color.secondary : viewModel.feedback.tintColor)
                    Text(viewModel.statusText)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(viewModel.feedback.isIdle ? .secondary : viewModel.feedback.tintColor)
                    Spacer()
                    Text("Esc dismiss")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                }

                if viewModel.mode == .search {
                    SearchResultsList(results: viewModel.results)
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                } else {
                    captureHintCard
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }

                Spacer(minLength: 0)
            }
            .padding(18)
        }
        .frame(width: 540, height: 360)
        .padding(10)
        .onAppear {
            isInputFocused = true
        }
        .onChange(of: viewModel.focusRequestCount) { _, _ in
            isInputFocused = true
        }
        .onChange(of: viewModel.confirmationFlashCount) { _, _ in
            flashOpacity = 0.18
            withAnimation(.easeOut(duration: 0.45)) {
                flashOpacity = 0
            }
        }
        .animation(.easeInOut(duration: 0.18), value: viewModel.mode)
    }

    private var borderColor: Color {
        switch viewModel.feedback {
        case .idle:
            return Color.white.opacity(0.08)
        case .success, .error:
            return viewModel.feedback.tintColor.opacity(0.85)
        }
    }

    private var statusSymbol: String {
        switch viewModel.feedback {
        case .idle:
            return viewModel.mode == .capture ? "square.and.arrow.down" : "magnifyingglass"
        case .success:
            return "checkmark.circle.fill"
        case .error:
            return "exclamationmark.triangle.fill"
        }
    }

    private var captureHintCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Capture tips")
                .font(.system(size: 12, weight: .semibold))
            Text("Use short, concrete notes. BrainBar stores them immediately in the main BrainLayer database.")
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
            Text("Shortcuts: ⌘1 Capture, ⌘2 Search, Return Submit")
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
    }
}

private struct QuickCaptureModeButton: View {
    let title: String
    let shortcut: KeyEquivalent
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 12, weight: .semibold))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    Capsule()
                        .fill(isSelected ? Color.accentColor.opacity(0.18) : Color(nsColor: .controlBackgroundColor))
                )
        }
        .buttonStyle(.plain)
        .foregroundStyle(isSelected ? Color.accentColor : .primary)
        .keyboardShortcut(shortcut, modifiers: [.command])
    }
}

private struct SearchResultsList: View {
    let results: [QuickCaptureSearchRow]

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 10) {
                if results.isEmpty {
                    Text("No matches yet. Try a tighter keyword or phrase.")
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(14)
                        .background(
                            RoundedRectangle(cornerRadius: 16)
                                .fill(Color(nsColor: .controlBackgroundColor))
                        )
                } else {
                    ForEach(results) { row in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(row.title)
                                .font(.system(size: 13, weight: .semibold))
                                .lineLimit(2)
                            Text(row.metadata)
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(14)
                        .background(
                            RoundedRectangle(cornerRadius: 16)
                                .fill(Color(nsColor: .controlBackgroundColor))
                        )
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
