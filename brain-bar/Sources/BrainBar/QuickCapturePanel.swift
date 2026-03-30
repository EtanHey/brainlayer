import AppKit
import SwiftUI

protocol QuickCaptureClipboard {
    func copy(_ string: String)
}

struct SystemQuickCaptureClipboard: QuickCaptureClipboard {
    func copy(_ string: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(string, forType: .string)
    }
}

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
    let content: String
    let metadata: String
}

enum QuickCaptureMoveDirection {
    case up
    case down
}

@MainActor
final class QuickCaptureViewModel: ObservableObject {
    @Published var inputText = ""
    @Published var mode: QuickCapturePanelState.Mode
    @Published private(set) var feedback: QuickCaptureFeedback = .idle
    @Published private(set) var results: [QuickCaptureSearchRow] = []
    @Published private(set) var selectedResultIndex: Int?
    @Published private(set) var confirmationFlashCount = 0
    @Published private(set) var focusRequestCount = 0

    private let db: BrainDatabase
    private let panelState: QuickCapturePanelState
    private let clipboard: QuickCaptureClipboard

    init(
        db: BrainDatabase,
        panelState: QuickCapturePanelState,
        clipboard: QuickCaptureClipboard = SystemQuickCaptureClipboard()
    ) {
        self.db = db
        self.panelState = panelState
        self.clipboard = clipboard
        mode = panelState.mode
    }

    var placeholderText: String {
        switch mode {
        case .capture:
            return "Capture an idea. Press Return to store."
        case .search:
            return "Search memory. Press Return to run or select."
        }
    }

    var statusText: String {
        if !feedback.isIdle {
            return feedback.message
        }

        switch mode {
        case .capture:
            return "Ready to store in BrainLayer"
        case .search:
            if inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return "Type to search BrainLayer"
            }
            if results.isEmpty {
                return "No matches yet"
            }
            let countLabel = results.count == 1 ? "1 result" : "\(results.count) results"
            guard let selectedResultIndex else { return countLabel }
            return "\(countLabel) • \(selectedResultIndex + 1) selected"
        }
    }

    var selectedResultID: String? {
        guard let selectedResultIndex, results.indices.contains(selectedResultIndex) else {
            return nil
        }
        return results[selectedResultIndex].id
    }

    func setMode(_ newMode: QuickCapturePanelState.Mode) {
        guard mode != newMode else { return }
        mode = newMode
        panelState.switchMode(newMode)
        feedback = .idle
        if newMode == .capture {
            results = []
            selectedResultIndex = nil
        }
        focusRequestCount += 1
    }

    func toggleMode() {
        setMode(mode == .capture ? .search : .capture)
    }

    func submit(forceCapture: Bool = false) {
        if forceCapture {
            submitCapture(preserveMode: true)
            return
        }

        switch mode {
        case .capture:
            submitCapture(preserveMode: false)
        case .search:
            if selectedResultIndex != nil, !results.isEmpty {
                applySelectedSearchResult()
            } else {
                submitSearch()
            }
        }
    }

    func moveSelectionUp() {
        moveSelection(.up)
    }

    func moveSelectionDown() {
        moveSelection(.down)
    }

    func handleInputTab() {
        toggleMode()
    }

    func handleInputChange(_ newValue: String) {
        if inputText != newValue {
            inputText = newValue
        }

        guard mode == .search else { return }
        submitSearch()
    }

    func handleInputReturn(modifiers: NSEvent.ModifierFlags) {
        if modifiers.contains(.command) {
            submit(forceCapture: true)
        } else {
            submit()
        }
    }

    func handleInputMove(_ direction: QuickCaptureMoveDirection) {
        guard mode == .search else { return }
        switch direction {
        case .up:
            moveSelectionUp()
        case .down:
            moveSelectionDown()
        }
    }

    func isSelected(_ row: QuickCaptureSearchRow) -> Bool {
        selectedResultID == row.id
    }

    func selectResult(id: String) {
        guard let index = results.firstIndex(where: { $0.id == id }) else { return }
        selectedResultIndex = index
    }

    func activateResult(id: String) {
        selectResult(id: id)
        applySelectedSearchResult()
    }

    func copyResultToClipboard(id: String) {
        guard let row = results.first(where: { $0.id == id }) else { return }
        clipboard.copy(row.content)
        feedback = .success("Copied result to clipboard")
    }

    func clearResultsSelection() {
        selectedResultIndex = nil
    }

    func dismiss() {
        panelState.dismiss()
        mode = panelState.mode
        inputText = ""
        results = []
        selectedResultIndex = nil
        feedback = .idle
    }

    func panelDidAppear() {
        focusRequestCount += 1
    }

    private func submitCapture(preserveMode: Bool) {
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            feedback = .error("Content cannot be empty")
            return
        }

        do {
            _ = try QuickCaptureController.capture(
                db: db,
                content: trimmed,
                tags: []
            )
            inputText = ""
            if !preserveMode {
                results = []
                selectedResultIndex = nil
            }
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
                    content: title,
                    metadata: "\(importance) • \(createdAt)"
                )
            }
            selectedResultIndex = results.isEmpty ? nil : 0
            feedback = .idle
        } catch {
            results = []
            selectedResultIndex = nil
            feedback = .error(error.localizedDescription)
        }
    }

    private func moveSelection(_ direction: QuickCaptureMoveDirection) {
        guard mode == .search, !results.isEmpty else { return }

        let currentIndex = selectedResultIndex ?? 0
        switch direction {
        case .up:
            selectedResultIndex = max(currentIndex - 1, 0)
        case .down:
            selectedResultIndex = min(currentIndex + 1, results.count - 1)
        }
        feedback = .idle
    }

    private func applySelectedSearchResult() {
        guard let selectedResultIndex, results.indices.contains(selectedResultIndex) else {
            submitSearch()
            return
        }

        let row = results[selectedResultIndex]
        setMode(.capture)
        inputText = row.title
        feedback = .idle
    }
}

final class KeyHandlingTextView: NSTextView {
    var onTab: (() -> Void)?
    var onMoveUp: (() -> Void)?
    var onMoveDown: (() -> Void)?
    var onReturn: ((NSEvent.ModifierFlags) -> Void)?

    var shouldInterceptArrowKeys: Bool = false

    override func doCommand(by selector: Selector) {
        switch selector {
        case #selector(insertTab(_:)):
            onTab?()
            return
        case #selector(moveDown(_:)):
            if shouldInterceptArrowKeys {
                onMoveDown?()
                return
            }
        case #selector(moveUp(_:)):
            if shouldInterceptArrowKeys {
                onMoveUp?()
                return
            }
        case #selector(insertNewline(_:)), #selector(insertNewlineIgnoringFieldEditor(_:)):
            let modifiers = NSApp.currentEvent?.modifierFlags.intersection(.deviceIndependentFlagsMask) ?? []
            onReturn?(modifiers)
            return
        default:
            break
        }
        super.doCommand(by: selector)
    }
}

private struct QuickCaptureInputField: NSViewRepresentable {
    @Binding var text: String
    let placeholder: String
    let focusRequestCount: Int
    let isSearchMode: Bool
    let onTextChange: (String) -> Void
    let onTab: () -> Void
    let onMoveUp: () -> Void
    let onMoveDown: () -> Void
    let onReturn: (NSEvent.ModifierFlags) -> Void

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: QuickCaptureInputField
        var lastFocusRequestCount: Int = 0

        init(parent: QuickCaptureInputField) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            parent.text = textView.string
            parent.onTextChange(textView.string)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let textView = KeyHandlingTextView(frame: .zero)
        textView.delegate = context.coordinator
        textView.drawsBackground = false
        textView.focusRingType = .none
        textView.font = .systemFont(ofSize: 14, weight: .medium)
        textView.isRichText = false
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.textContainerInset = NSSize(width: 0, height: 6)
        textView.textContainer?.widthTracksTextView = true
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.heightTracksTextView = false
        textView.isHorizontallyResizable = false
        textView.isVerticallyResizable = true
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.minSize = NSSize(width: 0, height: 0)
        textView.string = text
        textView.onTab = onTab
        textView.onMoveUp = onMoveUp
        textView.onMoveDown = onMoveDown
        textView.onReturn = onReturn

        let scrollView = NSScrollView(frame: .zero)
        scrollView.borderType = .noBorder
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.documentView = textView
        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let textView = nsView.documentView as? KeyHandlingTextView else { return }
        if textView.string != text {
            textView.string = text
        }
        textView.shouldInterceptArrowKeys = isSearchMode
        textView.onTab = onTab
        textView.onMoveUp = onMoveUp
        textView.onMoveDown = onMoveDown
        textView.onReturn = onReturn

        if context.coordinator.lastFocusRequestCount != focusRequestCount {
            context.coordinator.lastFocusRequestCount = focusRequestCount
            DispatchQueue.main.async {
                nsView.window?.makeFirstResponder(textView)
            }
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
    private let ownsDatabase: Bool

    init(dbPath: String) {
        database = BrainDatabase(path: dbPath)
        ownsDatabase = true
        viewModel = QuickCaptureViewModel(db: database, panelState: panelState)
        panel = QuickCapturePanel(
            contentRect: NSRect(x: 0, y: 0, width: 540, height: 360),
            styleMask: [.borderless, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        configurePanel()
    }

    init(db: BrainDatabase) {
        database = db
        ownsDatabase = false
        viewModel = QuickCaptureViewModel(db: database, panelState: panelState)
        panel = QuickCapturePanel(
            contentRect: NSRect(x: 0, y: 0, width: 540, height: 360),
            styleMask: [.borderless, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        configurePanel()
    }

    deinit {
        if ownsDatabase {
            database.close()
        }
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
                    .focusable(false)

                    QuickCaptureModeButton(
                        title: "Search",
                        shortcut: "2",
                        isSelected: viewModel.mode == .search
                    ) {
                        viewModel.setMode(.search)
                    }
                    .focusable(false)
                }
                .focusable(false)

                ZStack(alignment: .topLeading) {
                    if viewModel.inputText.isEmpty {
                        Text(viewModel.placeholderText)
                            .font(.system(size: 14, weight: .medium))
                            .foregroundStyle(.tertiary)
                            .padding(.top, 10)
                            .padding(.leading, 12)
                            .allowsHitTesting(false)
                    }

                    QuickCaptureInputField(
                        text: $viewModel.inputText,
                        placeholder: viewModel.placeholderText,
                        focusRequestCount: viewModel.focusRequestCount,
                        isSearchMode: viewModel.mode == .search,
                        onTextChange: { newValue in
                            viewModel.handleInputChange(newValue)
                        },
                        onTab: {
                            viewModel.handleInputTab()
                        },
                        onMoveUp: {
                            viewModel.handleInputMove(.up)
                        },
                        onMoveDown: {
                            viewModel.handleInputMove(.down)
                        },
                        onReturn: { modifiers in
                            viewModel.handleInputReturn(modifiers: modifiers)
                        }
                    )
                }
                    .frame(minHeight: viewModel.mode == .search ? 72 : 44, maxHeight: viewModel.mode == .search ? 96 : 64)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(Color(nsColor: .controlBackgroundColor))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 14)
                            .strokeBorder(borderColor, lineWidth: 1)
                    )

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
                .focusable(false)

                if viewModel.mode == .search {
                    SearchResultsList(
                        results: viewModel.results,
                        selectedResultID: viewModel.selectedResultID,
                        onSelect: { id in
                            viewModel.selectResult(id: id)
                        },
                        onActivate: { id in
                            viewModel.copyResultToClipboard(id: id)
                        }
                    )
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                } else {
                    captureHintCard
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }

                Spacer(minLength: 0)
            }
            .padding(16)
        }
        .frame(width: 540, height: 360)
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
            Text("Shortcuts: Tab Toggle, Return Submit, ⌘↩ Force store")
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .focusable(false)
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
        .focusable(false)
        .foregroundStyle(isSelected ? Color.accentColor : .primary)
        .keyboardShortcut(shortcut, modifiers: [.command])
    }
}

private struct SearchResultsList: View {
    let results: [QuickCaptureSearchRow]
    let selectedResultID: String?
    let onSelect: (String) -> Void
    let onActivate: (String) -> Void

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
                                .fill(row.id == selectedResultID ? Color.accentColor.opacity(0.16) : Color(nsColor: .controlBackgroundColor))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .strokeBorder(
                                    row.id == selectedResultID ? Color.accentColor.opacity(0.55) : Color.clear,
                                    lineWidth: 1
                                )
                        )
                        .contentShape(RoundedRectangle(cornerRadius: 16))
                        .onTapGesture {
                            onSelect(row.id)
                        }
                        .onTapGesture(count: 2) {
                            onActivate(row.id)
                        }
                        .focusable(false)
                    }
                }
            }
        }
        .focusable(false)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
