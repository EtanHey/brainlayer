import SwiftUI

struct SearchResultsList: View {
    let results: [SearchResult]
    let selectedResultID: String?
    let copiedResultID: String?
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
                    ForEach(results) { result in
                        SearchResultCard(
                            result: result,
                            isSelected: result.id == selectedResultID,
                            isCopied: result.id == copiedResultID
                        )
                        .onTapGesture {
                            onSelect(result.id)
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
