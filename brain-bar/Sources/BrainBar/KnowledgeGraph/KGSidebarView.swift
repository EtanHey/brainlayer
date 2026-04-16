import SwiftUI

struct KGSidebarView: View {
    let entity: EntityCard?
    let chunks: [BrainDatabase.KGChunkRow]
    let onOpenConversation: (String) -> Void
    let onClose: () -> Void

    var body: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 12) {
                if let entity {
                    header(entity)
                    if !entity.description.isEmpty {
                        Text(entity.description)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(3)
                    }
                    Divider()
                    relationsSection(entity.relations)
                    if !entity.metadata.isEmpty {
                        Divider()
                        metadataSection(entity.metadata)
                    }
                    if !chunks.isEmpty {
                        Divider()
                        chunksSection()
                    }
                } else {
                    Text("Select a node")
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding(12)
        }
        .frame(width: 280)
        .background(.ultraThinMaterial)
    }

    @ViewBuilder
    private func header(_ entity: EntityCard) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(entity.name)
                    .font(.headline)
                Text(entity.entityType)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(.quaternary))
            }
            Spacer()
            Button(action: onClose) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
        }
    }

    @ViewBuilder
    private func relationsSection(_ relations: [EntityCard.Relation]) -> some View {
        if relations.isEmpty {
            Text("No relations")
                .font(.caption)
                .foregroundColor(.secondary)
        } else {
            Text("Relations (\(relations.count))")
                .font(.subheadline.bold())
            ForEach(Array(relations.enumerated()), id: \.offset) { _, rel in
                HStack(spacing: 4) {
                    Text(rel.direction == "incoming" ? "←" : "→")
                        .font(.caption)
                        .foregroundColor(.secondary.opacity(0.6))
                    Text(rel.displayText)
                        .font(.caption.bold())
                        .lineLimit(1)
                }
            }
        }
    }

    @ViewBuilder
    private func metadataSection(_ metadata: [String: String]) -> some View {
        Text("Metadata")
            .font(.subheadline.bold())
        ForEach(Array(metadata.sorted(by: { $0.key < $1.key }).prefix(6)), id: \.key) { key, value in
            HStack(spacing: 4) {
                Text(key)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(value)
                    .font(.caption)
                    .lineLimit(1)
            }
        }
    }

    private func chunksSection() -> some View {
        let previewIndices = Array(0..<min(chunks.count, 10))
        return VStack(alignment: .leading, spacing: 8) {
            Text("Linked Chunks (\(chunks.count))")
                .font(.subheadline.bold())
            ForEach(previewIndices, id: \.self) { (index: Int) in
                let chunk = chunks[index]
                Button {
                    onOpenConversation(chunk.chunkID)
                } label: {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(chunk.snippet)
                            .font(.caption)
                            .lineLimit(4)
                            .multilineTextAlignment(.leading)
                        HStack(spacing: 8) {
                            Text(importanceText(for: chunk))
                            Text(relevanceText(for: chunk))
                            Spacer()
                            Text("Open")
                                .font(.caption.bold())
                                .foregroundColor(.accentColor)
                        }
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    }
                }
                .buttonStyle(.plain)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary))
            }
        }
    }

    private func importanceText(for chunk: BrainDatabase.KGChunkRow) -> String {
        "★ \(chunk.importance)"
    }

    private func relevanceText(for chunk: BrainDatabase.KGChunkRow) -> String {
        "↳ \(Int((chunk.relevance * 100).rounded()))%"
    }
}
