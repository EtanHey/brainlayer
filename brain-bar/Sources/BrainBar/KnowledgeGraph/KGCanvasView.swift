import SwiftUI

enum KGCanvasMetrics {
    static let sidebarWidth: CGFloat = 320
    static let canvasPadding: CGFloat = 18

    static func sidebarVisible(windowWidth: CGFloat, selectedEntityVisible: Bool) -> Bool {
        windowWidth >= 980 || selectedEntityVisible
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
    @GestureState private var toolbarInteractionActive = false

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

    var body: some View {
        GeometryReader { geo in
            let sidebarVisible = KGCanvasMetrics.sidebarVisible(
                windowWidth: geo.size.width,
                selectedEntityVisible: viewModel.selectedEntity != nil
            )

            HStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    atlasBackground
                    graphCanvas(snapshot: atlas)
                    atlasToolbar(snapshot: atlas)
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
                    if reduceMotion {
                        _ = viewModel.tick(reduceMotionEnabled: true)
                    } else {
                        startSimulation()
                    }
                })
                if loaded {
                    hasLoadedGraph = true
                }
            }
            .onChange(of: reduceMotion) { _, enabled in
                if enabled {
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
            let labelledNodeIDs = labelledNodeIDs(for: snapshot.visibleNodes)

            for region in snapshot.regions {
                drawRegionBackdrop(region: region, in: &ctx, environment: environment)
            }

            for edge in snapshot.visibleEdges {
                guard let source = nodeIndex[edge.sourceId], let target = nodeIndex[edge.targetId] else { continue }
                let highlighted = viewModel.selectedNodeId == edge.sourceId || viewModel.selectedNodeId == edge.targetId
                KGEdgeRenderer.draw(
                    edge: edge,
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
                Color.clear
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
    }

    private var atlasBackground: some View {
        ZStack {
            LinearGradient(
                colors: colorScheme == .dark
                    ? [
                        Color(red: 0.06, green: 0.07, blue: 0.10),
                        Color(red: 0.09, green: 0.11, blue: 0.14),
                    ]
                    : [
                        Color(red: 0.95, green: 0.95, blue: 0.92),
                        Color(red: 0.90, green: 0.92, blue: 0.94),
                    ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            VStack(spacing: 64) {
                ForEach(0..<4, id: \.self) { _ in
                    Rectangle()
                        .fill(Color.primary.opacity(colorScheme == .dark ? 0.05 : 0.04))
                        .frame(height: 1)
                }
            }
            .padding(.horizontal, 26)
        }
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

            Picker("Graph mode", selection: Binding(
                get: { viewModel.layoutMode },
                set: { viewModel.setLayoutMode($0) }
            )) {
                ForEach(KGAtlasMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Altitude")
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
                                .fill(Color.primary.opacity(0.07))
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
        .padding(20)
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

    private func drawRegionBackdrop(
        region: KGAtlasPresentation.Region,
        in context: inout GraphicsContext,
        environment: EnvironmentValues
    ) {
        guard !region.nodes.isEmpty else { return }

        let xs = region.nodes.map { $0.position.x }
        let ys = region.nodes.map { $0.position.y }
        guard let minX = xs.min(), let maxX = xs.max(), let minY = ys.min(), let maxY = ys.max() else {
            return
        }

        let rect = CGRect(
            x: minX - 64,
            y: minY - 52,
            width: max((maxX - minX) + 128, 180),
            height: max((maxY - minY) + 112, 132)
        )

        let tint = region.nodes.first?.color ?? .secondary
        context.fill(
            Ellipse().path(in: rect),
            with: .color(tint.opacity(colorScheme == .dark ? 0.14 : 0.10))
        )

        let label = Text(region.title.uppercased())
            .font(.system(size: 10, weight: .semibold, design: .monospaced))
            .foregroundColor(Color.primary.opacity(0.55))
        context.draw(
            context.resolve(label),
            at: CGPoint(x: rect.midX, y: rect.minY - 14),
            anchor: .center
        )
    }

    private func tapGesture(snapshot: KGAtlasPresentation.Snapshot) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                let point = canvasPoint(from: value.location, in: canvasSize)
                if let node = nodeAt(point: point, visibleNodes: snapshot.visibleNodes) {
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
            .fill(Color(nsColor: .windowBackgroundColor).opacity(colorScheme == .dark ? 0.82 : 0.9))
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke(Color.primary.opacity(0.08), lineWidth: 1)
            )
    }

    private var pageBackground: some View {
        Color(nsColor: .windowBackgroundColor)
    }

    private func labelChip(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(Color.primary.opacity(0.08))
            )
    }

    @ViewBuilder
    private var altitudeSlider: some View {
        switch viewModel.layoutMode {
        case .importance:
            Slider(value: $minimumImportance, in: 0...10, step: 1)
        case .tieredAltitude:
            Slider(value: $altitudeLevel, in: 0...Double(KGAltitudeTier.allCases.count - 1), step: 1)
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

    private func labelledNodeIDs(for nodes: [KGNode]) -> Set<String> {
        if scale >= 1.15 {
            return Set(nodes.map(\.id))
        }

        var labelled = Set<String>()
        if let selectedNodeId = viewModel.selectedNodeId {
            labelled.insert(selectedNodeId)
        }

        for node in nodes where node.name.localizedCaseInsensitiveCompare("Etan Heyman") == .orderedSame {
            labelled.insert(node.id)
        }

        let maxLabels = scale < 0.7 ? 18 : 36
        for node in nodes
            .sorted(by: { lhs, rhs in
                if lhs.importance == rhs.importance {
                    return lhs.linkedChunkCount > rhs.linkedChunkCount
                }
                return lhs.importance > rhs.importance
            })
            .prefix(maxLabels) {
            labelled.insert(node.id)
        }
        return labelled
    }

    private func startSimulation() {
        guard isActive, !reduceMotion, hasLoadedGraph, viewModel.nodes.count > 1 else { return }
        simulationController.setActive(true)
        simulationController.start {
            viewModel.tick(reduceMotionEnabled: reduceMotion)
        }
    }

    private func stopSimulation() {
        simulationController.setActive(false)
    }
}
