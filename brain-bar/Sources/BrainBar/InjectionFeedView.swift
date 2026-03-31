import SwiftUI

struct InjectionFeedView: View {
    let events: [InjectionEvent]
    @Binding var filterText: String

    private var filteredEvents: [InjectionEvent] {
        let trimmed = filterText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return events }
        let needle = trimmed.lowercased()
        return events.filter { event in
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
                        Text(event.chunkIDs.joined(separator: ", "))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }
                .padding(.vertical, 2)
            }
            .listStyle(.plain)
        }
    }
}
