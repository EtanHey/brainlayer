import SwiftUI

struct ChunkConversationSheet: View {
    let conversation: BrainDatabase.ExpandedConversation
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Conversation")
                        .font(.title3.bold())
                    Text(conversation.target.chunkID)
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button("Close") { dismiss() }
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(conversation.entries) { entry in
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(roleLabel(for: entry))
                                    .font(.system(size: 11, weight: .semibold))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(Capsule().fill(entry.isTarget ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12)))
                                Spacer()
                                Text(String(entry.createdAt.prefix(19)))
                                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.secondary)
                            }

                            if !entry.summary.isEmpty {
                                Text(entry.summary)
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundStyle(.secondary)
                            }

                            Text(entry.content)
                                .font(.system(size: 13))
                                .textSelection(.enabled)
                        }
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            RoundedRectangle(cornerRadius: 10)
                                .fill(entry.isTarget ? Color.accentColor.opacity(0.08) : Color(nsColor: .controlBackgroundColor))
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(entry.isTarget ? Color.accentColor.opacity(0.35) : Color(nsColor: .separatorColor).opacity(0.35), lineWidth: 1)
                        )
                    }
                }
            }
        }
        .padding(18)
        .frame(minWidth: 580, minHeight: 460)
    }

    private func roleLabel(for entry: BrainDatabase.ConversationChunk) -> String {
        switch entry.contentType {
        case "user_message":
            return "User"
        case "assistant_text":
            return "Assistant"
        default:
            return entry.contentType.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
}
