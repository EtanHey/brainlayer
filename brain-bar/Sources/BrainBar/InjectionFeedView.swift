import AppKit
import Combine
import SwiftUI

struct InjectionFeedPresentationState: Equatable {
    let events: [InjectionEvent]
    let degradationState: DegradationState

    static let empty = InjectionFeedPresentationState(
        events: [],
        degradationState: .healthy
    )
}

@MainActor
final class InjectionFeedPresentationModel: ObservableObject {
    @Published private(set) var state: InjectionFeedPresentationState = .empty

    var events: [InjectionEvent] { state.events }
    var degradationState: DegradationState { state.degradationState }

    private let throttleInterval: RunLoop.SchedulerTimeType.Stride
    private var cancellables: Set<AnyCancellable> = []
    private weak var boundStore: InjectionStore?

    init(throttleInterval: RunLoop.SchedulerTimeType.Stride = .milliseconds(500)) {
        self.throttleInterval = throttleInterval
    }

    func bind(to store: InjectionStore) {
        guard boundStore !== store else { return }

        boundStore = store
        cancellables.removeAll()
        let currentState = InjectionFeedPresentationState(
            events: store.events,
            degradationState: store.degradationState
        )
        if state != currentState {
            state = currentState
        }

        Publishers.CombineLatest(store.$events, store.$degradationState)
            .map { events, degradationState in
                InjectionFeedPresentationState(
                    events: events,
                    degradationState: degradationState
                )
            }
            .dropFirst()
            .removeDuplicates()
            .throttle(for: throttleInterval, scheduler: RunLoop.main, latest: true)
            .sink { [weak self] state in
                Task { @MainActor [weak self] in
                    self?.state = state
                }
            }
            .store(in: &cancellables)
    }
}

struct InjectionFeedView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let store: InjectionStore
    @Binding var filterText: String
    @StateObject private var presentationModel = InjectionFeedPresentationModel()
    @State private var expandedBurstIDs: Set<String> = []
    @State private var pendingExpandedBurstScrollID: String?
    @State private var copiedContinuationBurstID: String?
    @State private var conversationSelection = InjectionConversationSelection()
    @State private var loadingConversationChunkID: String?
    @AppStorage("brainbar.injectionFeed.typeFilter") private var typeFilterRaw = InjectionTypeFilter.all.rawValue

    private let accentPalette: [Color] = [
        .brainBarAccent,
        BrainBarStateTheme.active.theme.swiftUIColor,
        BrainBarStateTheme.degraded.theme.swiftUIColor,
        .brainBarAccentViolet,
        .brainBarAccentBright,
    ]

    var body: some View {
        let snapshot = makePresentation()
        GeometryReader { proxy in
            let wideLayout = proxy.size.width >= 960

            ScrollViewReader { scrollProxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        header(snapshot: snapshot)
                        overviewStrip(snapshot: snapshot)

                        if snapshot.bursts.isEmpty {
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
                .onChange(of: pendingExpandedBurstScrollID) { _, scrollID in
                    guard let scrollID else { return }
                    DispatchQueue.main.async {
                        let scroll = {
                            scrollProxy.scrollTo(scrollID, anchor: .top)
                            pendingExpandedBurstScrollID = nil
                        }
                        if reduceMotion {
                            scroll()
                        } else {
                            withAnimation(.easeInOut(duration: 0.18), scroll)
                        }
                    }
                }
            }
            .background(pageBackground)
        }
        .overlay {
            if let conversation = conversationSelection.conversation {
                ChunkConversationOverlay(
                    conversation: conversation,
                    title: conversationSelection.title,
                    onClose: { conversationSelection.close() }
                )
            } else if loadingConversationChunkID != nil {
                ConversationLoadingOverlay(onClose: { loadingConversationChunkID = nil })
            }
        }
        .overlay(alignment: .topTrailing) {
            if presentationModel.degradationState.isDegraded {
                DegradationBadge(reason: presentationModel.degradationState.reason)
                    .padding(.top, 20)
                    .padding(.trailing, 20)
            }
        }
        .onAppear {
            presentationModel.bind(to: store)
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

                filterChips
            }
        }
    }

    private var filterChips: some View {
        let activeFilter = InjectionTypeFilter(rawValue: typeFilterRaw) ?? .all
        return WrappingPillLayout(spacing: 6, lineSpacing: 6) {
            ForEach(InjectionTypeFilter.allCases, id: \.rawValue) { filter in
                Button {
                    typeFilterRaw = filter.rawValue
                } label: {
                    Text(filter.label)
                        .font(.system(size: 11, weight: .semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            Capsule()
                                .fill(activeFilter == filter ? Color.brainBarAccent.opacity(0.22) : Color.brainBarTextPrimary.opacity(0.06))
                        )
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: 280, alignment: .trailing)
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
                    .background(Capsule().fill(Color.brainBarTextPrimary.opacity(0.06)))

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
        LazyVStack(alignment: .leading, spacing: 16, pinnedViews: [.sectionHeaders]) {
            ForEach(snapshot.burstSections) { section in
                Section {
                    ForEach(section.bursts) { burst in
                        burstCard(burst)
                    }
                } header: {
                    ribbonHeader(section.bucket)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func burstCard(_ burst: InjectionPresentation.Burst) -> some View {
        let isExpanded = expandedBurstIDs.contains(burst.id)
        return VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        let leadEvent = burst.events.first
                        Text(leadEvent?.primaryKind.glyph ?? "📄")
                            .font(.system(size: 16, weight: .bold, design: .rounded))
                            .foregroundStyle(.blue)
                        Text(burst.summaryTitle)
                            .font(.system(size: 18, weight: .semibold, design: .rounded))
                            .lineLimit(2)
                    }
                    if let leadEvent = burst.events.first {
                        Text(leadEvent.triggeredByText)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    WrappingPillLayout(spacing: 8, lineSpacing: 8) {
                        chip(text: "Session: \(shortSessionID(burst.sessionID))", tint: .blue)
                        chip(text: relativeText(for: burst.endDate), tint: .neutral)
                        chip(text: "\(burst.queryCount) queries", tint: .neutral)
                        chip(text: "\(burst.tokenCount) tok", tint: .green)
                        if let leadEvent = burst.events.first {
                            chip(text: "\(leadEvent.primaryKind.glyph) \(leadEvent.primaryKind.label)", tint: .neutral)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 8) {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("\(burst.chunkCount)")
                            .font(.system(size: 24, weight: .bold, design: .rounded))
                        Text("retrieved chunks")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                    }

                    // QA #56: the expand affordance was a faint text link that hid
                    // the rest of the hits — give it real button weight.
                    Button {
                        toggleBurst(burst.id)
                    } label: {
                        Text(isExpanded ? "Collapse" : "Expand")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.blue)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 6)
                            .background(Capsule().fill(Color.brainBarAccent.opacity(0.14)))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(isExpanded ? "Collapse burst" : "Expand burst")

                    // QA #51: copy a resume command to continue this exact thread.
                    Button {
                        copyContinuation(for: burst)
                    } label: {
                        Label(
                            copiedContinuationBurstID == burst.id ? "Copied" : "Copy to continue",
                            systemImage: copiedContinuationBurstID == burst.id ? "checkmark" : "arrow.right.doc.on.clipboard"
                        )
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Copy command to continue this thread")
                }
            }

            if isExpanded {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(burst.events) { event in
                        eventRow(event)
                    }
                }
                .id(Self.expandedBurstDetailsID(burstID: burst.id))
            } else {
                collapsedChunkPreview(for: burst)
            }
        }
        .padding(18)
        .background(cardBackground)
    }

    private func collapsedChunkPreview(for burst: InjectionPresentation.Burst) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            let previewChunks = InjectionPresentation.previewChunks(for: burst.events, limit: 2)
            if previewChunks.isEmpty {
                ForEach(Array(burst.chunkPreviewIDs.enumerated()), id: \.offset) { _, chunkID in
                    Text(chunkPreviewText(chunkID))
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            } else {
                ForEach(previewChunks) { chunk in
                    Text("\(chunk.kind.glyph) \(chunk.displayText)")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }

            let visibleCount = previewChunks.isEmpty ? burst.chunkPreviewIDs.count : previewChunks.count
            let remaining = burst.remainingChunkCount(after: visibleCount)
            if remaining > 0 {
                Text("+\(remaining) more")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.brainBarTextPrimary.opacity(0.045))
        )
    }

    private func ribbonHeader(_ bucket: InjectionPresentation.RibbonBucket) -> some View {
        // QA #57: the time-range header (e.g. "1-2h ago") had no left padding, so
        // the label clipped against the edge. Pad horizontally and span full width.
        HStack(spacing: 10) {
            Text(bucket.title)
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: true, vertical: false)

            Rectangle()
                .fill(Color.brainBarTextPrimary.opacity(0.08))
                .frame(height: 1)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial)
    }

    private func eventRow(_ event: InjectionEvent) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Text(InjectionPresentation.shortTime(event.timestamp))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 48, alignment: .leading)

                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 7) {
                        Text(event.primaryKind.glyph)
                        Text(event.primaryKind.label)
                            .font(.system(size: 11, weight: .semibold))
                        Text(event.displayTitle)
                            .font(.system(size: 14, weight: .semibold))
                            .lineLimit(2)
                    }

                    Text(event.triggeredByText)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)

                    HStack(spacing: 10) {
                        Text("Session \(shortSessionID(event.sessionID))")
                        Text("\(event.chunkCount) chunks")
                        Text("\(event.tokenCount) tok")
                        Text("\(event.chunks.reduce(0) { $0 + $1.tags.count }) tags")
                    }
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)

                    chunkRibbon(for: event)

                    // QA #55/#56: expanding a burst reveals every hit immediately.
                    // Previously the chunk list hid behind a low-weight "Show hits"
                    // link, so an expanded "3-chunk" burst showed only one.
                    if !event.uniqueChunkIDs.isEmpty {
                        chunkList(for: event)
                    }
                }

                Spacer(minLength: 10)

                Button {
                    if let firstChunk = event.uniqueChunkIDs.first {
                        openConversation(chunkID: firstChunk, title: event.openingModalTitle(forChunkID: firstChunk))
                    }
                } label: {
                    Text(loadingConversationChunkID == event.uniqueChunkIDs.first ? "Opening" : "Open")
                        .font(.system(size: 11, weight: .semibold))
                }
                .buttonStyle(.plain)
                .disabled(event.uniqueChunkIDs.isEmpty || loadingConversationChunkID != nil)
            }

            Rectangle()
                .fill(Color.brainBarTextPrimary.opacity(0.06))
                .frame(height: 1)
        }
    }

    private func chunkRibbon(for event: InjectionEvent) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            WrappingPillLayout(spacing: 8, lineSpacing: 7) {
                ForEach(Array(Set(event.chunks.map(\.kind))).sorted(by: { $0.label < $1.label }), id: \.rawValue) { kind in
                    HStack(spacing: 4) {
                        Circle()
                            .fill(color(for: kind))
                            .frame(width: 6, height: 6)
                        Text(kind.label)
                            .font(.system(size: 10, weight: .medium))
                    }
                }
                if event.chunks.isEmpty {
                    Text("Hit bars show retrieved chunk IDs; source metadata was unavailable.")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            HStack(spacing: 4) {
                ForEach(Array(event.uniqueChunkIDs.enumerated()), id: \.offset) { _, chunkID in
                    let chunk = chunk(for: chunkID, event: event)
                    RoundedRectangle(cornerRadius: 999, style: .continuous)
                        .fill(color(for: chunk?.kind ?? .other))
                        .frame(maxWidth: .infinity)
                        .frame(height: 8)
                        .help(hitHelpText(chunkID: chunkID, chunk: chunk))
                }
            }
        }
        .padding(8)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.brainBarTextPrimary.opacity(0.04))
        )
    }

    private func chunkList(for event: InjectionEvent) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(event.uniqueChunkIDs.enumerated()), id: \.offset) { _, chunkID in
                Button {
                    openConversation(
                        chunkID: chunkID,
                        title: chunk(for: chunkID, event: event)?.kind.modalTitle ?? event.modalTitle
                    )
                } label: {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(color(for: chunk(for: chunkID, event: event)?.kind ?? .other))
                            .frame(width: 8, height: 8)
                        Text(chunkListTitle(chunkID: chunkID, event: event))
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .lineLimit(1)
                        Spacer()
                        Text("Open thread")
                            .font(.system(size: 10, weight: .semibold))
                    }
                }
                .buttonStyle(.plain)
                .disabled(loadingConversationChunkID != nil)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.brainBarTextPrimary.opacity(0.05))
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
                                .fill(Color.brainBarTextPrimary.opacity(0.045))
                        )
                    }
                }
            }

            sideRailCard(title: "Token pressure") {
                VStack(alignment: .leading, spacing: 8) {
                    railMetric(label: "Average", value: averageTokenText(snapshot: snapshot))
                    railMetric(label: "Peak event", value: peakTokenText(snapshot: snapshot))
                    railMetric(label: "Burst count", value: "\(snapshot.summary.burstCount)")
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
                .fill(Color.brainBarTextPrimary.opacity(0.05))
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
            .lineLimit(1)
            .truncationMode(.tail)
            .minimumScaleFactor(0.72)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .frame(maxWidth: 170, alignment: .leading)
            .background(
                Capsule()
                    .fill(tint.color.opacity(tint == .neutral ? 0.08 : 0.14))
            )
    }

    private func color(for kind: InjectionKind) -> Color {
        let index = kind.paletteIndex % accentPalette.count
        return accentPalette[index]
    }

    private func bucketColor(for index: Int, count: Int, snapshot: InjectionPresentation.Snapshot) -> Color {
        guard count > 0 else { return Color.brainBarTextPrimary.opacity(0.08) }
        let progress = Double(index) / Double(max(snapshot.ribbonBuckets.count - 1, 1))
        return Color.brainBarHSB(
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
                Color.brainBarGlassPrimary,
                Color.brainBarAccent.opacity(0.04),
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    private var cardBackground: some View {
        RoundedRectangle(cornerRadius: 20, style: .continuous)
            .fill(Color.brainBarGlassSecondary)
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(Color.brainBarTextPrimary.opacity(0.06), lineWidth: 1)
            )
    }

    private func copyContinuation(for burst: InjectionPresentation.Burst) {
        let command = InjectionContinuation.resumeCommand(sessionID: burst.sessionID)
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(command, forType: .string)

        let update = { copiedContinuationBurstID = burst.id }
        if reduceMotion {
            update()
        } else {
            withAnimation(.easeInOut(duration: 0.16), update)
        }

        let burstID = burst.id
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(2))
            guard copiedContinuationBurstID == burstID else { return }
            if reduceMotion {
                copiedContinuationBurstID = nil
            } else {
                withAnimation(.easeInOut(duration: 0.16)) { copiedContinuationBurstID = nil }
            }
        }
    }

    private func toggleBurst(_ burstID: String) {
        let update = {
            if expandedBurstIDs.contains(burstID) {
                expandedBurstIDs.remove(burstID)
            } else {
                expandedBurstIDs.insert(burstID)
                pendingExpandedBurstScrollID = Self.expandedBurstDetailsID(burstID: burstID)
            }
        }

        if reduceMotion {
            update()
        } else {
            withAnimation(.easeInOut(duration: 0.16), update)
        }
    }

    nonisolated static func expandedBurstDetailsID(burstID: String) -> String {
        "expanded-burst-details-\(burstID)"
    }

    private func shortSessionID(_ sessionID: String) -> String {
        guard sessionID.count > 12 else { return sessionID }
        let prefix = sessionID.prefix(10)
        return "\(prefix)…"
    }

    private func relativeText(for date: Date) -> String {
        let seconds = max(Date().timeIntervalSince(date), 0)
        if seconds < 60 {
            return "just now"
        }
        if seconds < 60 * 60 {
            return "\(Int(seconds / 60))m ago"
        }
        if seconds < 24 * 60 * 60 {
            return "\(Int(seconds / 3600))h ago"
        }
        return "\(Int(seconds / 86_400))d ago"
    }

    private func chunkPreviewText(_ chunkID: String) -> String {
        let limit = 80
        guard chunkID.count > limit else { return chunkID }
        return "\(chunkID.prefix(limit - 1))…"
    }

    private func makePresentation(now: Date = Date()) -> InjectionPresentation.Snapshot {
        InjectionPresentation.snapshot(
            events: presentationModel.events,
            filterText: filterText,
            typeFilter: InjectionTypeFilter(rawValue: typeFilterRaw) ?? .all,
            now: now,
            bucketCount: 24
        )
    }

    private func chunk(for chunkID: String, event: InjectionEvent) -> InjectionChunk? {
        event.chunks.first { $0.id == chunkID }
    }

    private func chunkListTitle(chunkID: String, event: InjectionEvent) -> String {
        guard let chunk = chunk(for: chunkID, event: event), !chunk.displayText.isEmpty else {
            return chunkID
        }
        return "\(chunk.kind.glyph) \(chunk.displayText)"
    }

    private func hitHelpText(chunkID: String, chunk: InjectionChunk?) -> String {
        guard let chunk else {
            return "Retrieved chunk \(chunkID)"
        }
        return "\(chunk.kind.label) · \(chunk.id) · \(chunk.displayText)"
    }

    private func openConversation(chunkID: String, title: String) {
        guard loadingConversationChunkID == nil else { return }
        loadingConversationChunkID = chunkID
        Task {
            do {
                let conversation = try await store.expandedConversationAsync(chunkID: chunkID)
                guard loadingConversationChunkID == chunkID else { return }
                conversationSelection.open(conversation, title: title)
                loadingConversationChunkID = nil
            } catch {
                if loadingConversationChunkID == chunkID {
                    loadingConversationChunkID = nil
                }
            }
        }
    }
}

private struct ConversationLoadingOverlay: View {
    let onClose: () -> Void

    var body: some View {
        ZStack {
            Rectangle()
                .fill(Color.brainBarBlack.opacity(0.18))
                .contentShape(Rectangle())
                .onTapGesture(perform: onClose)

            VStack(spacing: 12) {
                ProgressView()
                    .controlSize(.small)
                Text("Opening conversation")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            .padding(18)
            .background(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color.brainBarGlassPrimary)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(Color.brainBarTextPrimary.opacity(0.08), lineWidth: 1)
            )
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .zIndex(29)
    }
}

struct InjectionConversationSelection: Equatable {
    static let defaultTitle = "Conversation"

    var conversation: BrainDatabase.ExpandedConversation?
    var title = defaultTitle

    mutating func open(_ conversation: BrainDatabase.ExpandedConversation, title: String) {
        self.conversation = conversation
        self.title = title
    }

    mutating func close() {
        conversation = nil
        title = Self.defaultTitle
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
