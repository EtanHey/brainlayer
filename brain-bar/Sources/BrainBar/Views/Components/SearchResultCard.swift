import SwiftUI

struct SearchResultCard: View {
    let result: SearchResult
    let isSelected: Bool
    let isCopied: Bool
    /// QA #37: explicit "Open conversation" action. Optional so consumers without
    /// a conversation source render the card without the footer.
    var onOpenConversation: (() -> Void)?
    var onCopy: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(result.displayText)
                .font(.system(size: 13, weight: .semibold))
                .lineLimit(2)

            Text(result.compactMetadata)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(.secondary)

            if let tags = result.tagSummary {
                Text(tags)
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            if onOpenConversation != nil || onCopy != nil {
                actionFooter
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(backgroundColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(isSelected ? Color.brainBarAccent.opacity(0.55) : Color.brainBarClear, lineWidth: 1)
        )
        .overlay(alignment: .topTrailing) {
            if isCopied {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(BrainBarStateTheme.active.theme.swiftUIColor)
                    .padding(10)
                    .transition(.scale.combined(with: .opacity))
            }
        }
        .contentShape(RoundedRectangle(cornerRadius: 16))
    }

    @ViewBuilder
    private var actionFooter: some View {
        HStack(spacing: 8) {
            if let onOpenConversation {
                // QA #36/#37: the primary action is opening the conversation.
                Button(action: onOpenConversation) {
                    Label("Open conversation", systemImage: "bubble.left.and.bubble.right")
                        .font(.system(size: 11, weight: .semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Capsule().fill(Color.brainBarAccent.opacity(0.16)))
                        .foregroundStyle(Color.brainBarAccent)
                }
                .buttonStyle(.plain)
            }

            if let onCopy {
                // Copy is the secondary affordance (also double-click on the card).
                Button(action: onCopy) {
                    Label("Copy", systemImage: "doc.on.doc")
                        .font(.system(size: 11, weight: .medium))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Capsule().fill(Color.brainBarTextPrimary.opacity(0.08)))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }

            Spacer(minLength: 0)
        }
        .padding(.top, 2)
    }

    private var backgroundColor: Color {
        if isCopied {
            return BrainBarStateTheme.active.theme.swiftUIColor.opacity(0.16)
        }
        if isSelected {
            return Color.brainBarAccent.opacity(0.16)
        }
        return Color.brainBarGlassSecondary
    }
}
