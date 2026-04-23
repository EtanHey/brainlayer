import SwiftUI

struct KGSidebarView: View {
    let entity: EntityCard?
    let chunks: [BrainDatabase.KGChunkRow]
    let onOpenConversation: (String) -> Void
    let onClose: () -> Void

    var body: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(alignment: .leading, spacing: 16) {
                if let entity {
                    header(entity)
                    synopsis(entity)
                    relationsSection(entity.relations)
                    if !entity.metadata.isEmpty {
                        metadataSection(entity.metadata)
                    }
                    chunksSection()
                } else {
                    emptyState
                }
            }
            .padding(18)
        }
        .frame(width: 320)
        .background(
            LinearGradient(
                colors: [
                    Color(nsColor: .controlBackgroundColor),
                    Color.accentColor.opacity(0.03),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(Color.primary.opacity(0.08))
                .frame(width: 1)
        }
    }

    @ViewBuilder
    private func header(_ entity: EntityCard) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 8) {
                    Text(entity.name)
                        .font(.system(size: 24, weight: .bold, design: .rounded))
                        .fixedSize(horizontal: false, vertical: true)

                    HStack(spacing: 8) {
                        labelChip(entity.entityType.isEmpty ? "entity" : entity.entityType.capitalized, tint: .blue)
                        labelChip("\(entity.relations.count) links", tint: .primary)
                        labelChip("\(chunks.count) chunks", tint: .green)
                    }
                }

                Spacer(minLength: 12)

                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(16)
        .background(sectionBackground)
    }

    @ViewBuilder
    private func synopsis(_ entity: EntityCard) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Synopsis")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)

            if entity.description.isEmpty {
                Text("No entity description is stored yet. Use relations and linked chunks as the primary drilldown.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            } else {
                Text(entity.description)
                    .font(.system(size: 13, weight: .medium))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16)
        .background(sectionBackground)
    }

    @ViewBuilder
    private func relationsSection(_ relations: [EntityCard.Relation]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Relations")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)

            if relations.isEmpty {
                Text("No graph relations yet.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            } else {
                ForEach(Array(relations.enumerated()), id: \.offset) { _, relation in
                    HStack(alignment: .top, spacing: 10) {
                        Text(relation.direction == "incoming" ? "←" : "→")
                            .font(.system(size: 13, weight: .bold, design: .rounded))
                            .foregroundStyle(.secondary)
                            .frame(width: 14)

                        VStack(alignment: .leading, spacing: 4) {
                            Text(relation.targetName)
                                .font(.system(size: 13, weight: .semibold))
                            Text(relation.relationType.replacingOccurrences(of: "_", with: " "))
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }

                        Spacer()
                    }
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Color.primary.opacity(0.045))
                    )
                }
            }
        }
        .padding(16)
        .background(sectionBackground)
    }

    @ViewBuilder
    private func metadataSection(_ metadata: [String: String]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Metadata")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)

            ForEach(Array(metadata.sorted(by: { $0.key < $1.key }).prefix(6)), id: \.key) { key, value in
                VStack(alignment: .leading, spacing: 4) {
                    Text(key)
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Text(value)
                        .font(.system(size: 13, weight: .medium))
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(Color.primary.opacity(0.045))
                )
            }
        }
        .padding(16)
        .background(sectionBackground)
    }

    private func chunksSection() -> some View {
        let previewIndices = Array(0..<min(chunks.count, 8))
        return VStack(alignment: .leading, spacing: 12) {
            Text("Linked chunks")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)

            ForEach(Array(previewIndices), id: \.self) { index in
                let chunk = chunks[index]
                Button {
                    onOpenConversation(chunk.chunkID)
                } label: {
                    chunkCard(chunk)
                }
                .buttonStyle(.plain)
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.primary.opacity(0.045))
                )
            }
        }
        .padding(16)
        .background(sectionBackground)
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Knowledge atlas")
                .font(.system(size: 22, weight: .bold, design: .rounded))
            Text("Select a node to inspect its relations, metadata, and linked memory chunks.")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(sectionBackground)
    }

    private var sectionBackground: some View {
        RoundedRectangle(cornerRadius: 20, style: .continuous)
            .fill(Color(nsColor: .windowBackgroundColor))
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
    }

    private func labelChip(_ text: String, tint: ChipTint) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .semibold))
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(
                Capsule()
                    .fill(tint.color.opacity(tint == .primary ? 0.08 : 0.14))
            )
    }

    private func importanceText(for chunk: BrainDatabase.KGChunkRow) -> String {
        "★ \(chunk.importance)"
    }

    private func relevanceText(for chunk: BrainDatabase.KGChunkRow) -> String {
        "↳ \(Int((chunk.relevance * 100).rounded()))%"
    }

    private func chunkCard(_ chunk: BrainDatabase.KGChunkRow) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(chunk.snippet)
                .font(.system(size: 13, weight: .medium))
                .lineLimit(4)
                .multilineTextAlignment(.leading)
                .foregroundStyle(.primary)

            HStack {
                labelChip(importanceText(for: chunk), tint: .amber)
                labelChip(relevanceText(for: chunk), tint: .green)
                Spacer()
                Text("Open thread")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(Color.accentColor)
            }
        }
    }
}

private extension KGSidebarView {
    enum ChipTint {
        case primary
        case blue
        case green
        case amber

        var color: Color {
            switch self {
            case .primary: .primary
            case .blue: .blue
            case .green: .green
            case .amber: .orange
            }
        }
    }
}
