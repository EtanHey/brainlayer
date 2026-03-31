import SwiftUI

struct KGSidebarView: View {
    let entity: EntityCard?
    let chunks: [BrainDatabase.KGChunkRow]
    let onClose: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let entity {
                header(entity)
                Divider()
                relationsSection(entity.relations)
                if !chunks.isEmpty {
                    Divider()
                    chunksSection
                }
            } else {
                Text("Select a node")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding(12)
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
            Text("Relations")
                .font(.subheadline.bold())
            ForEach(Array(relations.enumerated()), id: \.offset) { _, rel in
                HStack(spacing: 4) {
                    Text(rel.relationType)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(rel.targetName)
                        .font(.caption.bold())
                }
            }
        }
    }

    @ViewBuilder
    private var chunksSection: some View {
        Text("Linked Chunks (\(chunks.count))")
            .font(.subheadline.bold())
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(Array(chunks.enumerated()), id: \.offset) { _, chunk in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(chunk.snippet)
                            .font(.caption)
                            .lineLimit(3)
                        HStack {
                            Text("imp: \(chunk.importance)")
                            Text("rel: \(String(format: "%.0f%%", chunk.relevance * 100))")
                        }
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    }
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary))
                }
            }
        }
    }
}
