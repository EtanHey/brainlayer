import SwiftUI

struct InjectionFeedView: View {
    @ObservedObject var store: InjectionStore
    @Binding var filterText: String
    @State private var expandedEventIDs: Set<Int64> = []
    @State private var selectedConversation: BrainDatabase.ExpandedConversation?

    private var filteredEvents: [InjectionEvent] {
        let trimmed = filterText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return store.events }
        let needle = trimmed.lowercased()
        return store.events.filter { event in
            event.sessionID.lowercased().contains(needle) ||
                event.query.lowercased().contains(needle) ||
                event.chunkIDs.joined(separator: " ").lowercased().contains(needle)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Filter injections", text: $filterText)
                .textFieldStyle(.roundedBorder)
                .font(.system(size: 12))

            List(filteredEvents) { event in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(event.sessionID)
                            .font(.system(size: 11, weight: .semibold))
                        Spacer()
                        Text(String(event.timestamp.prefix(19)))
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }

                    Text(event.query)
                        .font(.system(size: 12, weight: .medium))
                        .lineLimit(2)

                    Text(event.summaryLine)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)

                    if !event.chunkIDs.isEmpty {
                        Button(expandedEventIDs.contains(event.id) ? "Hide conversation chunks" : "Show conversation chunks") {
                            toggle(event.id)
                        }
                        .buttonStyle(.plain)
                        .font(.system(size: 11, weight: .semibold))

                        if expandedEventIDs.contains(event.id) {
                            VStack(alignment: .leading, spacing: 6) {
                                ForEach(event.chunkIDs, id: \.self) { chunkID in
                                    Button {
                                        selectedConversation = try? store.expandedConversation(chunkID: chunkID)
                                    } label: {
                                        HStack {
                                            Text(chunkID)
                                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                                .lineLimit(1)
                                            Spacer()
                                            Text("Open thread")
                                                .font(.system(size: 10, weight: .semibold))
                                        }
                                    }
                                    .buttonStyle(.borderless)
                                }
                            }
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(RoundedRectangle(cornerRadius: 8).fill(.quaternary))
                        }
                    }
                }
                .padding(.vertical, 2)
            }
            .listStyle(.plain)
        }
        .sheet(item: $selectedConversation) { conversation in
            ChunkConversationSheet(conversation: conversation)
        }
    }

    private func toggle(_ eventID: Int64) {
        if expandedEventIDs.contains(eventID) {
            expandedEventIDs.remove(eventID)
        } else {
            expandedEventIDs.insert(eventID)
        }
    }
}
