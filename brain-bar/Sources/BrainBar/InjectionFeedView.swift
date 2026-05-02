import SwiftUI

struct InjectionFeedView: View {
    @ObservedObject var store: InjectionStore
    @Binding var filterText: String
    @State private var expandedEventIDs: Set<Int64> = []
    @State private var selectedConversation: BrainDatabase.ExpandedConversation?

    private let accentPalette: [Color] = [
        Color(red: 0.42, green: 0.62, blue: 0.98),
        Color(red: 0.48, green: 0.82, blue: 0.70),
        Color(red: 0.95, green: 0.74, blue: 0.43),
        Color(red: 0.72, green: 0.58, blue: 0.96),
        Color(red: 0.42, green: 0.84, blue: 0.88),
    ]

    var body: some View {
        let snapshot = makePresentation()
        GeometryReader { proxy in
            let wideLayout = proxy.size.width >= 960

            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header(snapshot: snapshot)
                    overviewStrip(snapshot: snapshot)

                    if snapshot.filteredEvents.isEmpty {
                        emptyState
                    } else if wideLayout {
                        HStack(alignment: .top, spacing: 16) {
                            feedColumn(snapshot: snapshot)
                            sideRail(snapshot: snapshot)
                                .frame(width: 260)
                        }
                    } else {
                        VStack(alignment: .leading, spacing: 16) {
                            feedColumn(snapshot: snapshot)
                            sideRail(snapshot: snapshot)
                        }
                    }
                }
                .padding(20)
            }
            .background(pageBackground)
        }
        .overlay {
            if let conversation = selectedConversation {
                ChunkConversationOverlay(
                    conversation: conversation,
                    onClose: { selectedConversation = nil }
                )
            }
        }
    }

    private func header(snapshot: InjectionPresentation.Snapshot) -> some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Injections")
                    .font(.system(size: 26, weight: .semibold, design: .rounded))
                Text("Retrieval activity over the last hour, grouped into session bursts.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 16)

            VStack(alignment: .trailing, spacing: 8) {
                TextField("Filter injections", text: $filterText)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12))
                    .frame(minWidth: 220, maxWidth: 280)

                if !filterText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Text("Showing \(snapshot.filteredEvents.count) matching events")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func overviewStrip(snapshot: InjectionPresentation.Snapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 12) {
                overviewMetric(label: "Queries", value: "\(snapshot.summary.queryCount)")
                overviewMetric(label: "Chunks", value: "\(snapshot.summary.chunkCount)")
                overviewMetric(label: "Tokens", value: "\(snapshot.summary.tokenCount)")
                overviewMetric(label: "Sessions", value: "\(snapshot.summary.activeSessionCount)")
            }

            HStack(alignment: .center, spacing: 12) {
                Text("Last 1h")
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Capsule().fill(Color.primary.opacity(0.06)))

                Text("Activity ribbon")
                    .font(.system(size: 13, weight: .semibold))

                Spacer()

                Text(latestEventText(snapshot: snapshot))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            ribbon(snapshot: snapshot)
        }
        .padding(18)
        .background(cardBackground)
    }

    private func ribbon(snapshot: InjectionPresentation.Snapshot) -> some View {
        let maxBucket = max(snapshot.ribbonBuckets.max() ?? 0, 1)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .bottom, spacing: 4) {
                ForEach(Array(snapshot.ribbonBuckets.enumerated()), id: \.offset) { index, count in
                    Capsule(style: .continuous)
                        .fill(bucketColor(for: index, count: count, snapshot: snapshot))
                        .frame(maxWidth: .infinity)
                        .frame(height: count == 0 ? 8 : 12 + CGFloat(count) / CGFloat(maxBucket) * 50)
                }
            }
            .frame(height: 64, alignment: .bottom)

            HStack {
                Text("-60m")
                Spacer()
                Text("-30m")
                Spacer()
                Text("now")
            }
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(.secondary)
        }
    }

    private func feedColumn(snapshot: InjectionPresentation.Snapshot) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            ForEach(snapshot.bursts) { burst in
                burstCard(burst)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func burstCard(_ burst: InjectionPresentation.Burst) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(InjectionPresentation.burstRangeText(start: burst.startDate, end: burst.endDate))
                        .font(.system(size: 18, weight: .semibold, design: .rounded))
                    HStack(spacing: 8) {
                        chip(text: burst.sessionID, tint: .blue)
                        chip(text: "\(burst.queryCount) queries", tint: .neutral)
                        chip(text: "\(burst.tokenCount) tok", tint: .green)
                    }
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    Text("\(burst.chunkCount)")
                        .font(.system(size: 24, weight: .bold, design: .rounded))
                    Text("retrieved chunks")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 12) {
                ForEach(burst.events) { event in
                    eventRow(event)
                }
            }
        }
        .padding(18)
        .background(cardBackground)
    }

    private func eventRow(_ event: InjectionEvent) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Text(InjectionPresentation.shortTime(event.timestamp))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 48, alignment: .leading)

                VStack(alignment: .leading, spacing: 6) {
                    Text(event.query)
                        .font(.system(size: 14, weight: .semibold))
                        .lineLimit(3)

                    HStack(spacing: 10) {
                        Text("\(event.chunkCount) chunks")
                        Text("\(event.tokenCount) tok")
                        if !event.chunkIDs.isEmpty {
                            Button(expandedEventIDs.contains(event.id) ? "Hide hits" : "Show hits") {
                                toggle(event.id)
                            }
                            .buttonStyle(.plain)
                            .font(.system(size: 11, weight: .semibold))
                        }
                    }
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)

                    chunkRibbon(for: event)

                    if expandedEventIDs.contains(event.id) {
                        chunkList(for: event)
                    }
                }

                Spacer(minLength: 10)

                Button {
                    if let firstChunk = event.chunkIDs.first {
                        selectedConversation = try? store.expandedConversation(chunkID: firstChunk)
                    }
                } label: {
                    Text("Open")
                        .font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(.plain)
                .disabled(event.chunkIDs.isEmpty)
            }

            Rectangle()
                .fill(Color.primary.opacity(0.06))
                .frame(height: 1)
        }
    }

    private func chunkRibbon(for event: InjectionEvent) -> some View {
        HStack(spacing: 4) {
            ForEach(Array(event.chunkIDs.enumerated()), id: \.offset) { _, chunkID in
                RoundedRectangle(cornerRadius: 999, style: .continuous)
                    .fill(color(for: chunkID))
                    .frame(maxWidth: .infinity)
                    .frame(height: 8)
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.primary.opacity(0.04))
        )
    }

    private func chunkList(for event: InjectionEvent) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(event.chunkIDs, id: \.self) { chunkID in
                Button {
                    selectedConversation = try? store.expandedConversation(chunkID: chunkID)
                } label: {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(color(for: chunkID))
                            .frame(width: 8, height: 8)
                        Text(chunkID)
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .lineLimit(1)
                        Spacer()
                        Text("Open thread")
                            .font(.system(size: 10, weight: .semibold))
                    }
                }
                .buttonStyle(.plain)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.primary.opacity(0.05))
        )
    }

    private func sideRail(snapshot: InjectionPresentation.Snapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            sideRailCard(title: "Sessions · last hour") {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(snapshot.sessions.prefix(4)) { session in
                        HStack {
                            Text(session.sessionID)
                                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                            Spacer()
                            Text("\(session.queryCount)q · \(session.tokenCount) tok")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(.secondary)
                        }
                        .padding(10)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(Color.primary.opacity(0.045))
                        )
                    }
                }
            }

            sideRailCard(title: "Token pressure") {
                VStack(alignment: .leading, spacing: 8) {
                    railMetric(label: "Average", value: averageTokenText(snapshot: snapshot))
                    railMetric(label: "Peak event", value: peakTokenText(snapshot: snapshot))
                    railMetric(label: "Burst count", value: "\(snapshot.bursts.count)")
                }
            }

            sideRailCard(title: "Signals") {
                VStack(alignment: .leading, spacing: 8) {
                    Text(filterSignalText)
                        .font(.system(size: 12, weight: .medium))
                    if let highestChunkEvent = snapshot.windowEvents.max(by: { $0.chunkCount < $1.chunkCount }) {
                        Text("Chunk-heavy: \(highestChunkEvent.chunkCount) hits on “\(highestChunkEvent.query)”")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("No injections match the current filter.")
                .font(.system(size: 18, weight: .semibold))
            Text("Clear the filter or wait for fresh retrieval activity to land.")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .padding(22)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(cardBackground)
    }

    private func overviewMetric(label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 22, weight: .bold, design: .rounded))
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.primary.opacity(0.05))
        )
    }

    private func sideRailCard<Content: View>(title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)
            content()
        }
        .padding(16)
        .background(cardBackground)
    }

    private func railMetric(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(size: 12, weight: .semibold))
        }
    }

    private func chip(text: String, tint: BurstChipTint) -> some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(
                Capsule()
                    .fill(tint.color.opacity(tint == .neutral ? 0.08 : 0.14))
            )
    }

    private func color(for chunkID: String) -> Color {
        let index = abs(chunkID.hashValue) % accentPalette.count
        return accentPalette[index]
    }

    private func bucketColor(for index: Int, count: Int, snapshot: InjectionPresentation.Snapshot) -> Color {
        guard count > 0 else { return Color.primary.opacity(0.08) }
        let progress = Double(index) / Double(max(snapshot.ribbonBuckets.count - 1, 1))
        return Color(
            hue: 0.6 - progress * 0.18,
            saturation: 0.55,
            brightness: 0.94
        )
    }

    private func averageTokenText(snapshot: InjectionPresentation.Snapshot) -> String {
        guard !snapshot.windowEvents.isEmpty else { return "0 tok" }
        let total = snapshot.windowEvents.reduce(0) { $0 + $1.tokenCount }
        return "\(Int((Double(total) / Double(snapshot.windowEvents.count)).rounded())) tok"
    }

    private func peakTokenText(snapshot: InjectionPresentation.Snapshot) -> String {
        guard let event = snapshot.windowEvents.max(by: { $0.tokenCount < $1.tokenCount }) else {
            return "0 tok"
        }
        return "\(event.tokenCount) tok"
    }

    private func latestEventText(snapshot: InjectionPresentation.Snapshot) -> String {
        guard let first = snapshot.filteredEvents.first else { return "No recent events" }
        return "Latest · \(InjectionPresentation.shortTime(first.timestamp))"
    }

    private var filterSignalText: String {
        let trimmed = filterText.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "Showing the full retrieval stream."
        }
        return "Filtered by “\(trimmed)”"
    }

    private var pageBackground: some View {
        LinearGradient(
            colors: [
                Color(nsColor: .windowBackgroundColor),
                Color.accentColor.opacity(0.04),
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: 20, style: .continuous)
            .fill(Color(nsColor: .controlBackgroundColor))
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
    }

    private func toggle(_ eventID: Int64) {
        if expandedEventIDs.contains(eventID) {
            expandedEventIDs.remove(eventID)
        } else {
            expandedEventIDs.insert(eventID)
        }
    }

    private func makePresentation(now: Date = Date()) -> InjectionPresentation.Snapshot {
        InjectionPresentation.snapshot(
            events: store.events,
            filterText: filterText,
            now: now,
            bucketCount: 24
        )
    }
}

private extension InjectionFeedView {
    enum BurstChipTint {
        case neutral
        case blue
        case green

        var color: Color {
            switch self {
            case .neutral: .primary
            case .blue: .blue
            case .green: .green
            }
        }
    }
}
