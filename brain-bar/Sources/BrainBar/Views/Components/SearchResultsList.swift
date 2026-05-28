import SwiftUI

struct SearchResultsList: View {
    let results: [SearchResult]
    let selectedResultID: String?
    let copiedResultID: String?
    let onSelect: (String) -> Void
    let onActivate: (String) -> Void
    /// QA #36/#37: single-click drill-in. When provided, a single click opens the
    /// conversation for that result; copy stays available as the secondary
    /// (double-click) action. Optional so consumers without a conversation source
    /// keep selection-only behavior.
    var onOpenConversation: ((String) -> Void)? = nil

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
                                .fill(Color.brainBarGlassSecondary)
                        )
                } else {
                    ForEach(results) { result in
                        let resultID = result.id
                        let openAction: (() -> Void)? = onOpenConversation.map { open in { open(resultID) } }
                        SearchResultCard(
                            result: result,
                            isSelected: result.id == selectedResultID,
                            isCopied: result.id == copiedResultID,
                            onOpenConversation: openAction,
                            onCopy: { onActivate(resultID) }
                        )
                        .onTapGesture {
                            onSelect(result.id)
                            // QA #36: single click drills in instead of being a no-op.
                            onOpenConversation?(result.id)
                        }
                        .onTapGesture(count: 2) {
                            onActivate(result.id)
                        }
                        .focusable(false)
                    }
                }
            }
        }
        .focusable(false)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .animation(.easeInOut(duration: 0.18), value: copiedResultID)
    }
}
