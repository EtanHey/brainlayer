import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

@MainActor
final class KGSnapshotTests: XCTestCase {
    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    func testAtlasToolbarWrapsEveryRegionLegendChipWithoutHiddenOverflow() throws {
        let source = try String(
            contentsOf: packageRoot.appendingPathComponent("Sources/BrainBar/KnowledgeGraph/KGCanvasView.swift"),
            encoding: .utf8
        )
        let toolbarStart = try XCTUnwrap(source.range(of: "private func atlasToolbar"))
        let toolbarEnd = try XCTUnwrap(
            source.range(of: "private func atlasOverview", range: toolbarStart.upperBound..<source.endIndex)
        )
        let toolbar = String(source[toolbarStart.lowerBound..<toolbarEnd.lowerBound])

        XCTAssertTrue(toolbar.contains("WrappingPillLayout"))
        XCTAssertFalse(toolbar.contains("ScrollView(.horizontal, showsIndicators: false)"))
    }

    func testRendersDeterministicAtlasStatesToIsolatedDirectory() throws {
        guard let renderDirectory = ProcessInfo.processInfo.environment["BRAINBAR_RENDER_DIR"] else {
            throw XCTSkip("Set BRAINBAR_RENDER_DIR to render graph QA artifacts")
        }

        let destination = URL(fileURLWithPath: renderDirectory, isDirectory: true)
        try FileManager.default.createDirectory(
            at: destination,
            withIntermediateDirectories: true
        )

        try assertDeterministicRender(
            named: "graph-tiered-atlas.png",
            in: destination
        ) {
            snapshotView(viewModel: makeViewModel(mode: .tieredAltitude))
        }

        try assertDeterministicRender(
            named: "graph-importance-atlas.png",
            in: destination
        ) {
            snapshotView(viewModel: makeViewModel(mode: .importance))
        }

        try assertDeterministicRender(
            named: "graph-find-entity.png",
            in: destination
        ) {
            KGCanvasView(
                viewModel: makeViewModel(mode: .tieredAltitude),
                isActive: false,
                initialFindPresented: true,
                initialFindQuery: "brain"
            )
        }

        try assertDeterministicRender(
            named: "graph-selected-details.png",
            in: destination
        ) {
            let selected = makeViewModel(mode: .tieredAltitude)
            selected.selectedNodeId = "project-brainlayer"
            selected.selectedEntity = EntityCard(
                id: "project-brainlayer",
                name: "BrainLayer",
                entityType: "project",
                description: "Durable memory infrastructure for the golems ecosystem.",
                relations: [
                    .init(
                        relationType: "built_by",
                        targetName: "Etan Heyman",
                        targetEntityId: "person-etan",
                        direction: "outgoing"
                    ),
                    .init(
                        relationType: "uses",
                        targetName: "MCP",
                        targetEntityId: "tool-mcp",
                        direction: "outgoing"
                    ),
                ],
                importance: 9,
                altitudeTierTitle: "Summit"
            )
            return snapshotView(viewModel: selected)
        }

        try assertDeterministicRender(
            named: "graph-reduced-motion.png",
            in: destination
        ) {
            KGCanvasView(
                viewModel: makeViewModel(mode: .tieredAltitude),
                isActive: false,
                reduceMotionOverride: true
            )
        }

        try assertDeterministicRender(
            named: "graph-dismissed-sidebar.png",
            in: destination
        ) {
            let dismissed = makeViewModel(mode: .tieredAltitude)
            dismissed.selectedNodeId = nil
            dismissed.selectedEntity = nil
            return snapshotView(viewModel: dismissed)
        }

        let expectedNames = [
            "graph-tiered-atlas.png",
            "graph-importance-atlas.png",
            "graph-find-entity.png",
            "graph-selected-details.png",
            "graph-reduced-motion.png",
            "graph-dismissed-sidebar.png",
        ]
        for name in expectedNames {
            let data = try Data(contentsOf: destination.appendingPathComponent(name))
            XCTAssertGreaterThan(data.count, 10_000, "Expected a substantive PNG for \(name)")
        }
    }

    private func makeViewModel(mode: KGAtlasMode) -> KGViewModel {
        let viewModel = KGViewModel(graphReader: KGSnapshotGraphReader())
        let canvasSize = CGSize(width: 1_120, height: 760)
        viewModel.updateCanvasSize(canvasSize)
        viewModel.nodes = KGAtlasLayout.seededNodes(
            [
                node(id: "person-etan", name: "Etan Heyman", type: "person", importance: 10, chunks: 240),
                node(id: "project-brainlayer", name: "BrainLayer", type: "project", importance: 9, chunks: 180),
                node(id: "tool-mcp", name: "MCP", type: "tool", importance: 8, chunks: 120),
                node(id: "agent-codex", name: "Codex", type: "agent", importance: 7, chunks: 90),
                node(id: "company-openai", name: "OpenAI", type: "company", importance: 6, chunks: 60),
                node(id: "topic-retrieval", name: "Retrieval quality", type: "topic", importance: 5, chunks: 35),
                node(id: "decision-wal", name: "Bound WAL growth", type: "decision", importance: 4, chunks: 20),
            ],
            canvasSize: canvasSize,
            mode: mode
        )
        viewModel.edges = [
            KGEdge(sourceId: "person-etan", targetId: "project-brainlayer", relationType: "builds"),
            KGEdge(sourceId: "project-brainlayer", targetId: "tool-mcp", relationType: "exposes"),
            KGEdge(sourceId: "project-brainlayer", targetId: "agent-codex", relationType: "supports"),
            KGEdge(sourceId: "agent-codex", targetId: "company-openai", relationType: "created_by"),
            KGEdge(sourceId: "project-brainlayer", targetId: "topic-retrieval", relationType: "improves"),
            KGEdge(sourceId: "decision-wal", targetId: "project-brainlayer", relationType: "protects"),
        ]
        viewModel.layoutMode = mode
        return viewModel
    }

    private func snapshotView(viewModel: KGViewModel) -> KGCanvasView {
        KGCanvasView(
            viewModel: viewModel,
            isActive: false
        )
    }

    private func node(
        id: String,
        name: String,
        type: String,
        importance: Double,
        chunks: Int
    ) -> KGNode {
        KGNode(
            id: id,
            name: name,
            entityType: type,
            importance: importance,
            linkedChunkCount: chunks
        )
    }

    private func assertDeterministicRender<V: View>(
        named name: String,
        in directory: URL,
        makeView: () -> V
    ) throws {
        let first = try render(makeView(), named: name)
        let second = try render(makeView(), named: name)
        XCTAssertEqual(first, second, "Expected byte-stable fixed-fixture render for \(name)")
        try first.write(to: directory.appendingPathComponent(name))
    }

    private func render<V: View>(_ view: V, named name: String) throws -> Data {
        let host = NSHostingView(
            rootView: view
                .frame(width: 1_120, height: 760)
                .environment(\.colorScheme, .dark)
        )
        host.frame = NSRect(x: 0, y: 0, width: 1_120, height: 760)
        host.layoutSubtreeIfNeeded()
        RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.2))
        host.layoutSubtreeIfNeeded()

        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
            throw KGSnapshotRenderError.bitmapUnavailable(name)
        }
        host.cacheDisplay(in: host.bounds, to: bitmap)

        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            throw KGSnapshotRenderError.encodingFailed(name)
        }
        return png
    }
}

private enum KGSnapshotRenderError: Error {
    case bitmapUnavailable(String)
    case encodingFailed(String)
}

private final class KGSnapshotGraphReader: KnowledgeGraphReading, @unchecked Sendable {
    func fetchKGEntities(limit: Int) throws -> [BrainDatabase.KGEntityRow] { [] }
    func fetchKGRelations(limit: Int) throws -> [BrainDatabase.KGRelationRow] { [] }
}
