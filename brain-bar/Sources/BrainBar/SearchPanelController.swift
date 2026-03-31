import AppKit
import SwiftUI

final class SearchPanel: NSPanel {
    var onEscape: (() -> Void)?

    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }

    override func cancelOperation(_ sender: Any?) {
        onEscape?()
    }
}

@MainActor
final class SearchPanelController {
    let panelForTesting: SearchPanel

    private let panel: SearchPanel
    private let viewModel: SearchViewModel

    init(viewModel: SearchViewModel) {
        self.viewModel = viewModel
        panel = SearchPanel(
            contentRect: NSRect(x: 0, y: 0, width: 760, height: 560),
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panelForTesting = panel
        configurePanel()
    }

    convenience init(db: BrainDatabase) {
        let actor = SearchQueryActor(
            lexicalSearch: { query, candidateLimit, filters in
                (try? db.searchCandidates(
                    query: query,
                    limit: candidateLimit,
                    importanceMin: filters.importanceMin,
                    unreadOnly: filters.unreadOnly
                )) ?? []
            },
            rerank: { candidates, _ in candidates }
        )
        self.init(viewModel: SearchViewModel(queryActor: actor))
    }

    func show(query: String? = nil) {
        if let query {
            Task { @MainActor [weak self] in
                await self?.viewModel.updateQuery(query)
            }
        }

        centerPanel()
        panel.makeKeyAndOrderFront(nil)
        panel.orderFrontRegardless()
    }

    func dismiss() {
        panel.orderOut(nil)
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
            rootView: SearchPanelView(viewModel: viewModel, onDismiss: { [weak self] in
                self?.dismiss()
            })
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

private struct SearchPanelView: View {
    @ObservedObject var viewModel: SearchViewModel
    let onDismiss: () -> Void

    @FocusState private var isSearchFieldFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Search BrainLayer")
                .font(.system(size: 24, weight: .semibold))

            TextField("Search memory", text: queryBinding)
                .textFieldStyle(.plain)
                .font(.system(size: 18, weight: .medium))
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(Color(nsColor: .controlBackgroundColor))
                )
                .focused($isSearchFieldFocused)
                .onSubmit {
                    _ = viewModel.activateSelectedResult()
                }

            HStack(spacing: 10) {
                ForEach(SearchFilterChip.allCases, id: \.self) { chip in
                    Button(chip.title) {
                        Task { @MainActor in
                            await viewModel.selectFilter(chip)
                        }
                    }
                    .buttonStyle(.plain)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        Capsule()
                            .fill(viewModel.filters.primaryChip == chip ? Color.accentColor.opacity(0.18) : Color(nsColor: .controlBackgroundColor))
                    )
                    .foregroundStyle(viewModel.filters.primaryChip == chip ? Color.accentColor : .primary)
                }
            }

            SearchResultsList(
                results: viewModel.results,
                selectedResultID: viewModel.selectedResultID,
                copiedResultID: nil,
                onSelect: { id in
                    viewModel.selectResult(id: id)
                },
                onActivate: { id in
                    viewModel.selectResult(id: id)
                    _ = viewModel.activateSelectedResult()
                }
            )
            .onMoveCommand { direction in
                switch direction {
                case .up:
                    viewModel.moveSelection(.up)
                case .down:
                    viewModel.moveSelection(.down)
                default:
                    break
                }
            }

            Text(statusText)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 20)
                .fill(Color(nsColor: .windowBackgroundColor))
                .shadow(color: .black.opacity(0.16), radius: 24, y: 8)
        )
        .overlay(alignment: .topTrailing) {
            Button("Close") {
                onDismiss()
            }
            .buttonStyle(.plain)
            .padding(18)
        }
        .padding(18)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.clear)
        .task {
            isSearchFieldFocused = true
        }
    }

    private var queryBinding: Binding<String> {
        Binding(
            get: { viewModel.queryText },
            set: { newValue in
                viewModel.queryText = newValue
                Task { @MainActor in
                    await viewModel.updateQuery(newValue)
                }
            }
        )
    }

    private var statusText: String {
        if viewModel.queryText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Type to search BrainLayer"
        }
        if viewModel.results.isEmpty {
            return "No matches yet"
        }
        return "\(viewModel.results.count) results"
    }
}
