import SwiftUI

struct SearchResultCard: View {
    let result: SearchResult
    let isSelected: Bool
    let isCopied: Bool

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
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(backgroundColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(isSelected ? Color.accentColor.opacity(0.55) : Color.clear, lineWidth: 1)
        )
        .overlay(alignment: .topTrailing) {
            if isCopied {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Color(nsColor: .systemGreen))
                    .padding(10)
                    .transition(.scale.combined(with: .opacity))
            }
        }
        .contentShape(RoundedRectangle(cornerRadius: 16))
    }

    private var backgroundColor: Color {
        if isCopied {
            return Color(nsColor: .systemGreen).opacity(0.16)
        }
        if isSelected {
            return Color.accentColor.opacity(0.16)
        }
        return Color(nsColor: .controlBackgroundColor)
    }
}
