import SwiftUI

struct KGCanvasView: View {
    @ObservedObject var viewModel: KGViewModel

    @State private var offset: CGSize = .zero
    @State private var lastDragOffset: CGSize = .zero
    @State private var scale: CGFloat = 1.0
    @State private var lastScale: CGFloat = 1.0
    @State private var draggedNodeId: String?
    @State private var timerActive = true
    @State private var canvasSize: CGSize = .zero

    var body: some View {
        HStack(spacing: 0) {
            graphCanvas
            if viewModel.selectedEntity != nil {
                KGSidebarView(
                    entity: viewModel.selectedEntity,
                    chunks: viewModel.selectedEntityChunks,
                    onClose: { viewModel.selectNode(id: nil) }
                )
            }
        }
        .onAppear {
            viewModel.loadGraph()
            startSimulation()
        }
        .onDisappear { timerActive = false }
    }

    private var graphCanvas: some View {
        Canvas { context, size in
            var ctx = context
            // Apply pan + zoom transform
            ctx.translateBy(x: offset.width + size.width / 2, y: offset.height + size.height / 2)
            ctx.scaleBy(x: scale, y: scale)
            ctx.translateBy(x: -size.width / 2, y: -size.height / 2)

            let environment = EnvironmentValues()
            let nodeIndex = Dictionary(uniqueKeysWithValues: viewModel.nodes.map { ($0.id, $0) })

            // Draw edges first (behind nodes)
            for edge in viewModel.edges {
                guard let src = nodeIndex[edge.sourceId], let tgt = nodeIndex[edge.targetId] else { continue }
                let highlighted = viewModel.selectedNodeId == edge.sourceId || viewModel.selectedNodeId == edge.targetId
                KGEdgeRenderer.draw(
                    edge: edge, sourcePos: src.position, targetPos: tgt.position,
                    isHighlighted: highlighted, in: &ctx, environment: environment
                )
            }

            // Draw nodes
            for node in viewModel.nodes {
                KGNodeRenderer.draw(
                    node: node,
                    isSelected: node.id == viewModel.selectedNodeId,
                    in: &ctx, environment: environment
                )
            }
        }
        .background(
            GeometryReader { geo in
                Color.black.opacity(0.85)
                    .onAppear {
                        canvasSize = geo.size
                        viewModel.canvasCenter = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                    }
                    .onChange(of: geo.size) { _, newSize in
                        canvasSize = newSize
                        viewModel.canvasCenter = CGPoint(x: newSize.width / 2, y: newSize.height / 2)
                    }
            }
        )
        .overlay { ScrollWheelZoomView(scale: $scale) }
        .gesture(tapGesture)
        .gesture(dragGesture)
        .gesture(magnifyGesture)
        .overlay(alignment: .topLeading) { statsOverlay }
    }

    // MARK: - Gestures

    private var tapGesture: some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                let point = canvasPoint(from: value.location, in: canvasSize)
                if let node = viewModel.nodeAt(point: point) {
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
                scale = max(0.2, min(5.0, lastScale * value.magnification))
            }
            .onEnded { _ in
                lastScale = scale
            }
    }

    // MARK: - Coordinate transform

    private func canvasPoint(from screenPoint: CGPoint, in size: CGSize) -> CGPoint {
        // Inverse of the canvas transform:
        //   translate(offset + size/2) → scale → translate(-size/2)
        let cx = size.width / 2
        let cy = size.height / 2
        return CGPoint(
            x: (screenPoint.x - offset.width - cx) / scale + cx,
            y: (screenPoint.y - offset.height - cy) / scale + cy
        )
    }

    // MARK: - Simulation timer

    private func startSimulation() {
        Task { @MainActor in
            while timerActive {
                try? await Task.sleep(for: .milliseconds(33)) // ~30fps
                viewModel.tick()
            }
        }
    }

    // MARK: - Stats overlay

    private var statsOverlay: some View {
        Text("\(viewModel.nodes.count) nodes · \(viewModel.edges.count) edges")
            .font(.caption2)
            .foregroundColor(.white.opacity(0.6))
            .padding(6)
    }
}
