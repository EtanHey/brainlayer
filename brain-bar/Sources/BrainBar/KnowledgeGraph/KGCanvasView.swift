import SwiftUI

enum KGCanvasMetrics {
    static let sidebarWidth: CGFloat = 320
    static let canvasPadding: CGFloat = 18

    static func sidebarVisible(windowWidth: CGFloat, selectedEntityVisible: Bool) -> Bool {
        _ = windowWidth
        return selectedEntityVisible
    }

    static func drawableSize(windowSize: CGSize, sidebarVisible: Bool) -> CGSize {
        CGSize(
            width: max(windowSize.width - (sidebarVisible ? sidebarWidth : 0) - canvasPadding * 2, 0),
            height: max(windowSize.height - canvasPadding * 2, 0)
        )
    }
}

struct KGCanvasView: View {
    @ObservedObject var viewModel: KGViewModel
    let isActive: Bool
    private let reduceMotionOverride: Bool?
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var offset: CGSize = .zero
    @State private var lastDragOffset: CGSize = .zero
    @State private var scale: CGFloat = 1.0
    @State private var lastScale: CGFloat = 1.0
    @State private var canvasSize: CGSize = .zero
    @State private var minimumImportance: Double = 6
    @State private var altitudeLevel: Double = Double(KGAltitudeTier.signal.rawValue)
    @State private var hasLoadedGraph = false
    @State private var simulationController = KGSimulationController()
    @State private var isFindPresented: Bool
    @State private var findQuery: String
    @State private var highlightedFindResultID: String?
    @GestureState private var toolbarInteractionActive = false

    init(
        viewModel: KGViewModel,
        isActive: Bool,
        initialFindPresented: Bool = false,
        initialFindQuery: String = "",
        reduceMotionOverride: Bool? = nil
    ) {
        self.viewModel = viewModel
        self.isActive = isActive
        self.reduceMotionOverride = reduceMotionOverride
        _isFindPresented = State(initialValue: initialFindPresented)
        _findQuery = State(initialValue: initialFindQuery)
    }

    private var shouldReduceMotion: Bool {
        reduceMotionOverride ?? reduceMotion
    }

    private var atlas: KGAtlasPresentation.Snapshot {
        KGAtlasPresentation.snapshot(
            nodes: viewModel.nodes,
            edges: viewModel.edges,
            selectedNodeId: viewModel.selectedNodeId,
            minimumImportance: minimumImportance,
            mode: viewModel.layoutMode,
            altitude: altitudeLevel
        )
    }

    private var findResults: [KGAtlasPresentation.FindResult] {
        KGAtlasPresentation.findResults(
            nodes: atlas.visibleNodes,
            edges: atlas.visibleEdges,
            query: findQuery
        )
    }

    var body: some View {
        GeometryReader { geo in
            let sidebarVisible = KGCanvasMetrics.sidebarVisible(
                windowWidth: geo.size.width,
                selectedEntityVisible: viewModel.selectedNodeId != nil
            )

            HStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    atlasBackground
                    graphCanvas(snapshot: atlas)
                    atlasControls(snapshot: atlas)
                    atlasOverview(snapshot: atlas)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                if sidebarVisible {
                    KGSidebarView(
                        entity: viewModel.selectedEntity,
                        chunks: viewModel.selectedEntityChunks,
                        chunkTotal: viewModel.selectedEntityChunkTotal,
                        isLoadingChunks: viewModel.isLoadingSelectedEntityChunks,
                        canLoadMoreChunks: viewModel.selectedEntityCanLoadMoreChunks,
                        files: viewModel.selectedEntityFiles,
                        fileTotal: viewModel.selectedEntityFileTotal,
                        isLoadingFiles: viewModel.isLoadingSelectedEntityFiles,
                        canLoadMoreFiles: viewModel.selectedEntityCanLoadMoreFiles,
                        hasChunkLoadError: viewModel.selectedEntityChunkSidebarLoadFailed,
                        hasFileLoadError: viewModel.selectedEntityFileSidebarLoadFailed,
                        onOpenConversation: { viewModel.openConversation(chunkID: $0) },
                        onOpenFile: { viewModel.openSourceFile($0) },
                        onSelectRelation: { relation in
                            if viewModel.selectRelatedEntity(from: relation) {
                                stopSimulation()
                            }
                        },
                        onLoadMoreChunks: { viewModel.loadMoreChunks() },
                        onLoadMoreFiles: { viewModel.loadMoreFiles() },
                        onClose: { viewModel.selectNode(id: nil) }
                    )
                }
            }
            .background(pageBackground)
            .overlay {
                if let conversation = viewModel.selectedConversation {
                    ChunkConversationOverlay(
                        conversation: conversation,
                        onClose: { viewModel.selectedConversation = nil }
                    )
                }
            }
            .task(id: isActive) {
                guard isActive else {
                    stopSimulation()
                    return
                }

                let loaded = await viewModel.loadGraphRepeatedly(onSuccessfulLoad: {
                    hasLoadedGraph = true
                    if shouldReduceMotion {
                        _ = viewModel.tick(reduceMotionEnabled: true)
                    } else if !viewModel.isLayoutPinned {
                        startSimulation()
                    }
                })
                if loaded {
                    hasLoadedGraph = true
                }
            }
            .onChange(of: reduceMotion) { _, enabled in
                if reduceMotionOverride ?? enabled {
                    _ = viewModel.tick(reduceMotionEnabled: true)
                    stopSimulation()
                } else {
                    startSimulation()
                }
            }
            .onChange(of: viewModel.nodes.count) { _, _ in
                startSimulation()
            }
            .onChange(of: viewModel.layoutMode) { _, _ in
                startSimulation()
            }
            .onChange(of: viewModel.isLayoutPinned) { _, pinned in
                if !pinned {
                    startSimulation()
                }
            }
            .onDisappear {
                stopSimulation()
            }
        }
    }

    private func graphCanvas(snapshot: KGAtlasPresentation.Snapshot) -> some View {
        Canvas { context, size in
            var ctx = context
            ctx.translateBy(x: offset.width + size.width / 2, y: offset.height + size.height / 2)
            ctx.scaleBy(x: scale, y: scale)
            ctx.translateBy(x: -size.width / 2, y: -size.height / 2)

            var environment = EnvironmentValues()
            environment.colorScheme = colorScheme
            let nodeIndex = Dictionary(uniqueKeysWithValues: snapshot.visibleNodes.map { ($0.id, $0) })
            let labelledNodeIDs = KGAtlasPresentation.priorityLabelNodeIDs(
                nodes: snapshot.visibleNodes,
                selectedNodeID: viewModel.selectedNodeId,
                scale: scale
            )

            for edge in snapshot.visibleEdges {
                guard let source = nodeIndex[edge.sourceId], let target = nodeIndex[edge.targetId] else { continue }
                let highlighted = viewModel.selectedNodeId == edge.sourceId || viewModel.selectedNodeId == edge.targetId
                KGEdgeRenderer.draw(
                    edge: KGAtlasPresentation.lineOnlyEdge(edge),
                    sourcePos: source.position,
                    targetPos: target.position,
                    isHighlighted: highlighted,
                    in: &ctx,
                    environment: environment
                )
            }

            for node in snapshot.visibleNodes {
                KGNodeRenderer.draw(
                    node: node,
                    isSelected: node.id == viewModel.selectedNodeId,
                    showsLabel: labelledNodeIDs.contains(node.id),
                    in: &ctx,
                    environment: environment
                )
            }
        }
        .background(
            GeometryReader { geo in
                Color.brainBarClear
                    // This reports the actual drawable Canvas size after the
                    // sidebar and padding have been applied.
                    .onAppear { setCanvas(size: geo.size) }
                    .onChange(of: geo.size) { _, newSize in
                        resizeCanvas(size: newSize)
                    }
            }
        )
        .overlay { ScrollWheelZoomView(scale: $scale) }
        .allowsHitTesting(!toolbarInteractionActive)
        .gesture(tapGesture(snapshot: snapshot))
        .gesture(dragGesture)
        .gesture(magnifyGesture)
        .padding(KGCanvasMetrics.canvasPadding)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Knowledge atlas canvas")
        .accessibilityValue(KGAtlasPresentation.canvasAccessibilityValue(
            visibleEntityCount: snapshot.visibleNodes.count,
            visibleRelationshipCount: snapshot.visibleEdges.count
        ))
        .accessibilityHint(KGAtlasPresentation.canvasAccessibilityHint)
        .accessibilityIdentifier("knowledge-graph-canvas")
        .accessibilityAction(named: Text("Find entity")) {
            presentFindEntity()
        }
    }

    private var atlasBackground: some View {
        ZStack {
            LinearGradient(
                colors: colorScheme == .dark
                    ? [
                        .brainBarBackgroundAbyss,
                        .brainBarBackgroundBase,
                    ]
                    : [
                        .brainBarGraphCanvasLightTop,
                        .brainBarGraphCanvasLightBottom,
                    ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            VStack(spacing: 64) {
                ForEach(0..<4, id: \.self) { _ in
                    Rectangle()
                        .fill(Color.brainBarTextPrimary.opacity(colorScheme == .dark ? 0.05 : 0.04))
                        .frame(height: 1)
                }
            }
            .padding(.horizontal, 26)
        }
    }

    private func atlasControls(snapshot: KGAtlasPresentation.Snapshot) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            atlasToolbar(snapshot: snapshot)

            if isFindPresented {
                KGFindEntityPanel(
                    query: $findQuery,
                    results: findResults,
                    highlightedResultID: highlightedFindResultID,
                    onHighlight: { highlightedFindResultID = $0 },
                    onSelect: selectFindResult,
                    onDismiss: dismissFindEntity
                )
                .frame(width: 420)
            }
        }
        .padding(20)
    }

    private func atlasToolbar(snapshot: KGAtlasPresentation.Snapshot) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Knowledge atlas")
                        .font(.system(size: 20, weight: .bold, design: .rounded))
                    Text("\(snapshot.visibleNodes.count) entities · \(snapshot.visibleEdges.count) visible relations")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 12)

                labelChip(statusChipText(snapshot: snapshot))
            }

            HStack(spacing: 10) {
                Picker("Graph mode", selection: Binding(
                    get: { viewModel.layoutMode },
                    set: { viewModel.setLayoutMode($0) }
                )) {
                    ForEach(KGAtlasMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .accessibilityHint(modeEffectDescription(snapshot: snapshot))

                Button {
                    if isFindPresented {
                        dismissFindEntity()
                    } else {
                        presentFindEntity()
                    }
                } label: {
                    Label("Find entity", systemImage: "magnifyingglass")
                        .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.bordered)
                .keyboardShortcut("f", modifiers: .command)
                .accessibilityIdentifier("knowledge-graph-find-entity")
            }

            Text(modeEffectDescription(snapshot: snapshot))
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("knowledge-graph-mode-effect")

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(viewModel.layoutMode == .importance ? "Minimum importance" : "Altitude")
                        .font(.system(size: 11, weight: .semibold))
                    Spacer()
                    Text(altitudeReadout(snapshot: snapshot))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                altitudeSlider
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(snapshot.regions) { region in
                        HStack(spacing: 6) {
                            Circle()
                                .fill(region.nodes.first?.color ?? .secondary)
                                .frame(width: 8, height: 8)
                            Text(region.title)
                                .font(.system(size: 11, weight: .semibold))
                            Text("\(region.nodes.count)")
                                .font(.system(size: 10, weight: .medium, design: .monospaced))
                                .foregroundStyle(.secondary)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            Capsule()
                                .fill(Color.brainBarTextPrimary.opacity(0.07))
                        )
                    }
                }
            }
        }
        .padding(16)
        .frame(maxWidth: 420)
        .background(toolbarBackground)
        .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .onTapGesture {}
        .simultaneousGesture(toolbarInteractionGesture)
    }

    private func atlasOverview(snapshot: KGAtlasPresentation.Snapshot) -> some View {
        VStack(alignment: .trailing, spacing: 10) {
            if let selectedEntity = viewModel.selectedEntity {
                VStack(alignment: .trailing, spacing: 4) {
                    Text("Focus")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .foregroundStyle(.secondary)
                    Text(selectedEntity.name)
                        .font(.system(size: 14, weight: .bold))
                    Text(selectedEntity.entityType.capitalized)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .trailing, spacing: 6) {
                Text("Atlas regions")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.secondary)

                ForEach(snapshot.regions) { region in
                    HStack(spacing: 8) {
                        Text(region.title)
                            .font(.system(size: 11, weight: .semibold))
                        Text("\(region.nodes.count)")
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(14)
        .background(toolbarBackground)
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
    }

    private func tapGesture(snapshot: KGAtlasPresentation.Snapshot) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                let point = canvasPoint(from: value.location, in: canvasSize)
                if let node = nodeAt(point: point, visibleNodes: snapshot.visibleNodes) {
                    stopSimulation()
                    viewModel.selectNode(id: node.id)
                } else {
                    viewModel.selectNode(id: nil)
                }
            }
    }

    private var dragGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                offset = CGSize(
                    width: lastDragOffset.width + value.translation.width,
                    height: lastDragOffset.height + value.translation.height
                )
            }
            .onEnded { _ in
                lastDragOffset = offset
            }
    }

    private var magnifyGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                scale = max(0.35, min(4.0, lastScale * value.magnification))
            }
            .onEnded { _ in
                lastScale = scale
            }
    }

    private func canvasPoint(from screenPoint: CGPoint, in size: CGSize) -> CGPoint {
        let cx = size.width / 2
        let cy = size.height / 2
        return CGPoint(
            x: (screenPoint.x - offset.width - cx) / scale + cx,
            y: (screenPoint.y - offset.height - cy) / scale + cy
        )
    }

    private func nodeAt(point: CGPoint, visibleNodes: [KGNode]) -> KGNode? {
        for node in visibleNodes {
            let dx = point.x - node.position.x
            let dy = point.y - node.position.y
            let dist = sqrt(dx * dx + dy * dy)
            if dist <= node.radius + 4 {
                return node
            }
        }
        return nil
    }

    private func setCanvas(size: CGSize) {
        guard size != .zero else { return }
        canvasSize = size
        viewModel.updateCanvasSize(size)
    }

    private func resizeCanvas(size: CGSize) {
        setCanvas(size: size)
    }

    private var toolbarInteractionGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .updating($toolbarInteractionActive) { _, state, _ in
                state = true
            }
    }

    private var toolbarBackground: some View {
        RoundedRectangle(cornerRadius: 18, style: .continuous)
            .fill(Color.brainBarGlassPrimary.opacity(colorScheme == .dark ? 0.82 : 0.9))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.brainBarTextPrimary.opacity(0.08), lineWidth: 1)
            )
    }

    private var pageBackground: some View {
        Color.brainBarGlassPrimary
    }

    private func labelChip(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(Color.brainBarTextPrimary.opacity(0.08))
            )
    }

    @ViewBuilder
    private var altitudeSlider: some View {
        switch viewModel.layoutMode {
        case .importance:
            Slider(value: $minimumImportance, in: 0...10, step: 1)
                .accessibilityLabel("Minimum entity importance")
                .accessibilityValue("\(Int(minimumImportance)) out of 10")
        case .tieredAltitude:
            Slider(value: $altitudeLevel, in: 0...Double(KGAltitudeTier.allCases.count - 1), step: 1)
                .accessibilityLabel("Visible altitude tiers")
                .accessibilityValue(atlas.activeAltitudeTier.title)
        }
    }

    private func statusChipText(snapshot: KGAtlasPresentation.Snapshot) -> String {
        switch viewModel.layoutMode {
        case .importance:
            return "importance ≥ \(Int(minimumImportance))"
        case .tieredAltitude:
            return "\(snapshot.activeAltitudeTier.title) \(Int(altitudeLevel) + 1)/\(KGAltitudeTier.allCases.count)"
        }
    }

    private func altitudeReadout(snapshot: KGAtlasPresentation.Snapshot) -> String {
        switch viewModel.layoutMode {
        case .importance:
            return "importance \(Int(minimumImportance))/10"
        case .tieredAltitude:
            return "\(snapshot.activeAltitudeTier.caption) · lower reveals more"
        }
    }

    private func modeEffectDescription(snapshot: KGAtlasPresentation.Snapshot) -> String {
        KGAtlasPresentation.modeEffectDescription(
            mode: viewModel.layoutMode,
            minimumImportance: minimumImportance,
            altitudeTier: snapshot.activeAltitudeTier
        )
    }

    private func presentFindEntity() {
        isFindPresented = true
        highlightedFindResultID = findResults.first?.id
    }

    private func dismissFindEntity() {
        isFindPresented = false
        highlightedFindResultID = nil
    }

    private func selectFindResult(_ id: String) {
        guard atlas.visibleNodes.contains(where: { $0.id == id }) else { return }
        stopSimulation()
        viewModel.selectNode(id: id)
        dismissFindEntity()
    }

    private func startSimulation() {
        guard isActive, !shouldReduceMotion, !viewModel.isLayoutPinned, hasLoadedGraph, viewModel.nodes.count > 1 else { return }
        simulationController.setActive(true)
        simulationController.start {
            viewModel.tick(reduceMotionEnabled: shouldReduceMotion)
        }
    }

    private func stopSimulation() {
        simulationController.setActive(false)
    }
}

struct KGFindEntityPanel: View {
    @Binding var query: String
    let results: [KGAtlasPresentation.FindResult]
    let highlightedResultID: String?
    let onHighlight: (String?) -> Void
    let onSelect: (String) -> Void
    let onDismiss: () -> Void
    @FocusState private var queryFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)

                TextField("Find entity", text: $query)
                    .textFieldStyle(.plain)
                    .focused($queryFocused)
                    .accessibilityLabel("Find entity")
                    .accessibilityHint("Searches the entities currently visible in the atlas")
                    .accessibilityIdentifier("knowledge-graph-find-field")

                Button(action: onDismiss) {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Close Find entity")
            }
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.brainBarTextPrimary.opacity(0.06))
            )

            if results.isEmpty {
                Text("No visible entities match “\(query)”.")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 12)
                    .accessibilityIdentifier("knowledge-graph-find-empty")
            } else {
                ScrollView(.vertical, showsIndicators: true) {
                    LazyVStack(alignment: .leading, spacing: 6) {
                        ForEach(results) { result in
                            Button {
                                onSelect(result.id)
                            } label: {
                                HStack(alignment: .center, spacing: 10) {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(result.name)
                                            .font(.system(size: 13, weight: .semibold))
                                            .lineLimit(1)
                                        Text("\(result.entityTypeTitle) · \(result.relationshipSummary)")
                                            .font(.system(size: 10, weight: .medium))
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                    }
                                    Spacer(minLength: 8)
                                    Image(systemName: "arrow.right.circle.fill")
                                        .foregroundStyle(.secondary)
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(
                                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                                        .fill(result.id == highlightedResultID
                                            ? Color.brainBarAccent.opacity(0.16)
                                            : Color.brainBarTextPrimary.opacity(0.035))
                                )
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(result.accessibilityLabel)
                            .accessibilityHint("Selects this entity and opens its details")
                            .accessibilityIdentifier("knowledge-graph-find-result-\(result.id)")
                            .onHover { hovering in
                                if hovering {
                                    onHighlight(result.id)
                                }
                            }
                        }
                    }
                }
                .frame(maxHeight: 250)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.brainBarGlassPrimary.opacity(0.96))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(Color.brainBarTextPrimary.opacity(0.1), lineWidth: 1)
                )
        )
        .shadow(color: .black.opacity(0.12), radius: 16, y: 8)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Find entity results")
        .accessibilityIdentifier("knowledge-graph-find-results")
        .onAppear {
            queryFocused = true
            if highlightedResultID == nil {
                onHighlight(results.first?.id)
            }
        }
        .onChange(of: results.map(\.id)) { _, ids in
            if !ids.contains(highlightedResultID ?? "") {
                onHighlight(ids.first)
            }
        }
        .onMoveCommand { direction in
            let move: KGAtlasPresentation.FindMove?
            switch direction {
            case .up: move = .up
            case .down: move = .down
            default: move = nil
            }
            guard let move else { return }
            onHighlight(KGAtlasPresentation.nextFindResultID(
                currentID: highlightedResultID,
                move: move,
                results: results
            ))
        }
        .onSubmit {
            if let id = highlightedResultID ?? results.first?.id {
                onSelect(id)
            }
        }
        .onExitCommand(perform: onDismiss)
    }
}
