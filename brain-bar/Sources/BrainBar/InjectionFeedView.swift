import AppKit
import Combine
import SwiftUI

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
        state = InjectionFeedPresentationState(
            events: store.events,
            degradationState: store.degradationState,
            loadState: store.loadState
        )

        Publishers.CombineLatest3(store.$events, store.$degradationState, store.$loadState)
            .map { events, degradationState, loadState in
                InjectionFeedPresentationState(
                    events: events,
                    degradationState: degradationState,
                    loadState: loadState
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
    private let store: InjectionStore?
    private let fixture: InjectionFeedFixture?
    @Binding var filterText: String
    @StateObject private var presentationModel = InjectionFeedPresentationModel()
    @State private var expandedBurstIDs: Set<String> = []
    @State private var pendingExpandedBurstScrollID: String?
    @State private var copiedContinuationBurstID: String?
    @State private var conversationSelection = InjectionConversationSelection()
    @State private var loadingConversationChunkID: String?
    @State private var actionReceipt: InjectionActionReceipt? = nil
    @State private var groupingDisclosureExpanded = false
    @AppStorage("brainbar.injectionFeed.typeFilter") private var typeFilterRaw = InjectionTypeFilter.all.rawValue

    init(store: InjectionStore, filterText: Binding<String>) {
        self.store = store
        self.fixture = nil
        _filterText = filterText
    }

    init(fixture: InjectionFeedFixture) {
        self.store = nil
        self.fixture = fixture
        _filterText = .constant(fixture.filterText)
        _expandedBurstIDs = State(initialValue: fixture.expandedBurstIDs)
        _actionReceipt = State(initialValue: fixture.actionReceipt)
    }

    init(disconnectedAt now: Date) {
        self.init(
            fixture: InjectionFeedFixture(
                events: [],
                now: now,
                connectionState: .disconnected
            )
        )
    }

    private let accentPalette: [Color] = [
        .brainBarAccent,
        BrainBarStateTheme.active.theme.swiftUIColor,
        BrainBarStateTheme.degraded.theme.swiftUIColor,
        .brainBarAccentViolet,
        .brainBarAccentBright,
    ]

    var body: some View {
        let presentationState = effectivePresentationState
        let snapshot = makePresentation()
        let surfaceState = InjectionFeedSurfaceState.resolve(
            snapshot: snapshot,
            presentationState: presentationState,
            connectionState: fixture?.connectionState ?? .connected,
            filterActive: !filterText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || effectiveTypeFilter != .all
        )
        GeometryReader { proxy in
            let wideLayout = proxy.size.width >= 960

            ScrollViewReader { scrollProxy in
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        header(snapshot: snapshot)
                        if surfaceState.showsContent {
                            if case .degraded(let reason, true) = surfaceState {
                                degradationNotice(reason: reason)
                            }
                            overviewStrip(snapshot: snapshot)

                            if wideLayout {
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
                        } else {
                            stateCard(surfaceState)
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
        .overlay(alignment: .bottomTrailing) {
            if let actionReceipt {
                actionReceiptBadge(actionReceipt)
                    .padding(20)
            }
        }
        .onAppear {
            if let store {
                presentationModel.bind(to: store)
            }
        }
    }

    private func header(snapshot: InjectionPresentation.Snapshot) -> some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Injections")
                    .font(.system(size: 26, weight: .semibold, design: .rounded))
                Text("Retrieval activity grouped into literal retrieval bursts.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            Spacer(minLength: 16)

            VStack(alignment: .trailing, spacing: 8) {
                TextField("Filter injections", text: $filterText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .frame(minWidth: 220, maxWidth: 280)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Self.filterControlFillColor)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(Self.filterControlBorderColor, lineWidth: 1)
                    )
                    .accessibilityIdentifier(Self.filterSearchAccessibilityID)
                    .accessibilityLabel("Filter retrieval bursts")

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
        let activeFilter = effectiveTypeFilter
        return Picker("Retrieval type", selection: typeFilterBinding) {
            ForEach(InjectionTypeFilter.allCases, id: \.rawValue) { filter in
                Text(filter.label).tag(filter.rawValue)
            }
        }
        .pickerStyle(.menu)
        .labelsHidden()
        .accessibilityLabel("Retrieval type: \(activeFilter.label)")
        .accessibilityIdentifier(Self.filterTypeAccessibilityID)
        .frame(maxWidth: 280, alignment: .trailing)
    }

    private func overviewStrip(snapshot: InjectionPresentation.Snapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            InjectionSummaryView(events: snapshot.windowEvents)

            DisclosureGroup(isExpanded: $groupingDisclosureExpanded) {
                Text(Self.burstGroupingDisclosure)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } label: {
                Text("Retrieval bursts")
                    .font(.system(size: 12, weight: .semibold))
            }
            .accessibilityIdentifier(Self.groupingDisclosureAccessibilityID)

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
                        Text(burst.queryTitle)
                            .font(.system(size: 18, weight: .semibold, design: .rounded))
                            .lineLimit(2)
                    }
                    if burst.selectedResultSummary.caseInsensitiveCompare(burst.queryTitle) != .orderedSame {
                        Text("Selected result · \(burst.selectedResultSummary)")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    WrappingPillLayout(spacing: 8, lineSpacing: 8) {
                        chip(text: burst.sourceLabel, tint: .blue)
                        if burst.projectLabel != "Project unavailable" {
                            chip(text: burst.projectLabel, tint: .neutral)
                        }
                        chip(text: burst.timestampLabel, tint: .neutral)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 8) {
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("\(burst.resultCount)")
                            .font(.system(size: 24, weight: .bold, design: .rounded))
                        Text(Self.burstChunkCounterLabel)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.trailing)
                            .lineLimit(2)
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
                    .accessibilityIdentifier("\(Self.burstActionAccessibilityID).expand.\(burst.id)")

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
                    .accessibilityIdentifier("\(Self.burstActionAccessibilityID).copy.\(burst.id)")
                }
            }

            if isExpanded {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(burst.events) { event in
                        eventRow(event, selectedResultChunkID: burst.selectedResultProvenance?.chunkID)
                    }
                }
                .id(Self.expandedBurstDetailsID(burstID: burst.id))
            } else {
                collapsedChunkPreview(for: burst)
            }
        }
        .padding(18)
        .background(cardBackground)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Retrieval burst: \(burst.queryTitle), \(burst.resultCount) results, \(burst.timestampLabel)")
    }

    private func collapsedChunkPreview(for burst: InjectionPresentation.Burst) -> some View {
        VStack(alignment: .leading, spacing: 7) {
            let previewChunks = burst.additionalResultPreviews
            if previewChunks.isEmpty {
                Text("Additional result details are available when expanded.")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
            } else {
                ForEach(previewChunks) { chunk in
                    Text("\(chunk.kind.glyph) \(chunk.displayText)")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }

            let remaining = burst.remainingCollapsedResultCount
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

    private func eventRow(_ event: InjectionEvent, selectedResultChunkID: String?) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Text(InjectionPresentation.shortTime(event.timestamp))
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 48, alignment: .leading)

                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 7) {
                        Text(event.primaryKind.glyph)
                        if let kindLabel = event.expandedRowKindLabel {
                            Text(kindLabel)
                                .font(.system(size: 11, weight: .semibold))
                        }
                        Text(event.displayTitle)
                            // QA Part B: the thread name read too small next to the
                            // 14pt body — bump to 15 and let it span the row so it
                            // is the clear primary label of the expanded row.
                            .font(.system(size: 15, weight: .semibold))
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    if let triggeredBy = event.expandedRowTriggeredByText {
                        Text(triggeredBy)
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }

                    WrappingPillLayout(spacing: 8, lineSpacing: 6) {
                        Text("Timestamp \(event.timestamp)")
                        Text("Source \(event.primaryKind.label)")
                        if !event.claudeProjectPath.isEmpty {
                            Text("Project \(event.claudeProjectPath)")
                        }
                        Text("Session \(event.sessionID)")
                        Text("Event ID \(event.id)")
                        if !event.claudeConversationID.isEmpty {
                            Text("Conversation \(event.claudeConversationID)")
                        }
                        Text("Mode \(event.mode)")
                        Text("\(event.uniqueChunkIDs.count) results")
                        Text("\(event.tokenCount) tok")
                    }
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                    chunkRibbon(for: event)

                    // QA #55/#56: expanding a burst reveals every hit immediately.
                    // Previously the chunk list hid behind a low-weight "Show hits"
                    // link, so an expanded "3-chunk" burst showed only one.
                    if !event.uniqueChunkIDs.isEmpty {
                        chunkList(for: event, selectedResultChunkID: selectedResultChunkID)
                    }
                }

                Spacer(minLength: 10)

                // QA Part B: "open thread" button was a bare text label crammed
                // against the trailing edge — it clipped at narrow widths. Give it
                // a real pill affordance and a fixed intrinsic width so it never
                // truncates ("Opening" is longer than "Open").
                Button {
                    if let firstChunk = event.uniqueChunkIDs.first {
                        openConversation(chunkID: firstChunk, title: event.openingModalTitle(forChunkID: firstChunk))
                    }
                } label: {
                    Text(loadingConversationChunkID == event.uniqueChunkIDs.first ? "Opening" : "Open thread")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundStyle(.blue)
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Capsule().fill(Color.brainBarAccent.opacity(0.14)))
                }
                .buttonStyle(.plain)
                .disabled(event.uniqueChunkIDs.isEmpty || loadingConversationChunkID != nil)
                .accessibilityLabel("Open thread")
                .accessibilityIdentifier("\(Self.burstActionAccessibilityID).open.\(event.id)")
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
                if let statusText = event.chunkRibbonStatusText {
                    Text(statusText)
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

    private func chunkList(for event: InjectionEvent, selectedResultChunkID: String?) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(event.uniqueChunkIDs.enumerated()), id: \.offset) { _, chunkID in
                let resultChunk = chunk(for: chunkID, event: event)
                let provenance = InjectionPresentation.ResultProvenance(
                    chunkID: chunkID,
                    eventID: event.id,
                    sessionID: event.sessionID,
                    timestamp: event.timestamp,
                    chunk: resultChunk
                )
                Button {
                    openConversation(
                        chunkID: chunkID,
                        title: resultChunk?.kind.modalTitle ?? event.modalTitle
                    )
                } label: {
                    HStack(alignment: .top, spacing: 8) {
                        Circle()
                            .fill(color(for: resultChunk?.kind ?? .other))
                            .frame(width: 8, height: 8)
                            .padding(.top, 4)
                        VStack(alignment: .leading, spacing: 5) {
                            Text(chunkListTitle(chunkID: chunkID, event: event))
                                .font(.system(size: 11, weight: .medium))
                                .lineLimit(2)
                            WrappingPillLayout(spacing: 6, lineSpacing: 5) {
                                ForEach(
                                    provenance.expandedMetadataLabels(
                                        isSelected: chunkID == selectedResultChunkID
                                    ),
                                    id: \.self
                                ) { label in
                                    Text(label)
                                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
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
                            Text(session.displayLabel)
                                .font(.system(size: 11, weight: .semibold))
                                .lineLimit(2)
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
                    let trimmedFilter = filterText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmedFilter.isEmpty {
                        Text("Filtered by “\(trimmedFilter)”")
                            .font(.system(size: 12, weight: .medium))
                    }
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

    @ViewBuilder
    private func stateCard(_ state: InjectionFeedSurfaceState) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            switch state {
            case .loading:
                ProgressView()
                    .controlSize(.small)
                Text("Loading retrieval activity")
                    .font(.system(size: 18, weight: .semibold))
                Text("Waiting for the first read-only injections snapshot.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            case .disconnected:
                Text("Injection feed disconnected")
                    .font(.system(size: 18, weight: .semibold))
                Text("BrainBar has no read-only injection store connection for this session.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            case .empty:
                Text("No retrieval activity yet")
                    .font(.system(size: 18, weight: .semibold))
                Text("The feed is connected and healthy; no injections were returned.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            case .noMatches:
                Text("No injections match the current filters")
                    .font(.system(size: 18, weight: .semibold))
                Text("Clear the text or type filter to show retrieval activity.")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
            case .degraded(let reason, false):
                Text("Injection feed read failed")
                    .font(.system(size: 18, weight: .semibold))
                Text(reason)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            case .content, .degraded(_, true):
                EmptyView()
            }
        }
        .padding(22)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(cardBackground)
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier(Self.surfaceStateAccessibilityID)
    }

    private func degradationNotice(reason: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 3) {
                Text("Feed degraded · showing the last good retrievals")
                    .font(.system(size: 12, weight: .semibold))
                Text(reason)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.orange.opacity(0.1))
        )
    }

    private func actionReceiptBadge(_ receipt: InjectionActionReceipt) -> some View {
        Label(
            receipt.message,
            systemImage: receipt.kind == .success ? "checkmark.circle.fill" : "exclamationmark.triangle.fill"
        )
        .font(.system(size: 11, weight: .semibold))
        .foregroundStyle(receipt.kind == .success ? Color.green : Color.orange)
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(
            Capsule()
                .fill(Color.brainBarGlassPrimary)
                .shadow(color: Color.black.opacity(0.12), radius: 8, y: 3)
        )
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier(Self.actionReceiptAccessibilityID)
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
        let command = InjectionContinuation.resumeCommand(
            conversationID: burst.claudeConversationID,
            fallbackSessionID: burst.sessionID,
            projectPath: burst.claudeProjectPath
        )
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        let copied = pasteboard.setString(command, forType: .string)
        actionReceipt = .copyResult(copied: copied)
        guard copied else {
            return
        }

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
            if actionReceipt?.message == "Resume command copied" {
                actionReceipt = nil
            }
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

    private var effectivePresentationState: InjectionFeedPresentationState {
        fixture?.presentationState ?? presentationModel.state
    }

    private var effectiveTypeFilter: InjectionTypeFilter {
        fixture?.typeFilter ?? InjectionTypeFilter(rawValue: typeFilterRaw) ?? .all
    }

    private var typeFilterBinding: Binding<String> {
        guard let fixture else { return $typeFilterRaw }
        return .constant(fixture.typeFilter.rawValue)
    }

    private func makePresentation(now: Date = Date()) -> InjectionPresentation.Snapshot {
        let effectiveNow = fixture?.now ?? now
        return InjectionPresentation.snapshot(
            events: effectivePresentationState.events,
            filterText: filterText,
            typeFilter: effectiveTypeFilter,
            now: effectiveNow,
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
        guard let store else {
            actionReceipt = .disconnectedThread
            return
        }
        loadingConversationChunkID = chunkID
        Task {
            do {
                let conversation = try await store.expandedConversationAsync(chunkID: chunkID)
                guard loadingConversationChunkID == chunkID else { return }
                conversationSelection.open(conversation, title: title)
                loadingConversationChunkID = nil
                actionReceipt = .threadOpenResult(errorDescription: nil)
            } catch {
                if loadingConversationChunkID == chunkID {
                    loadingConversationChunkID = nil
                }
                actionReceipt = .threadOpenResult(errorDescription: error.localizedDescription)
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

extension InjectionFeedView {
    nonisolated static var filterControlsUseAccentTint: Bool { false }

    nonisolated static let burstChunkCounterLabel = "memories surfaced into context"
    nonisolated static let burstGroupingDisclosure =
        "Retrieval bursts group the same session and trigger topic while consecutive events remain under 60 minutes."
    nonisolated static let filterSearchAccessibilityID = "brainbar.injections.filter.search"
    nonisolated static let filterTypeAccessibilityID = "brainbar.injections.filter.type"
    nonisolated static let groupingDisclosureAccessibilityID = "brainbar.injections.grouping"
    nonisolated static let burstActionAccessibilityID = "brainbar.injections.burst.action"
    nonisolated static let surfaceStateAccessibilityID = "brainbar.injections.state"
    nonisolated static let actionReceiptAccessibilityID = "brainbar.injections.action.receipt"

    static var filterControlFillColor: Color {
        Color.brainBarTextPrimary.opacity(0.06)
    }

    static var filterControlSelectedFillColor: Color {
        Color.brainBarTextPrimary.opacity(0.11)
    }

    static var filterControlBorderColor: Color {
        Color.brainBarTextPrimary.opacity(0.14)
    }

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
