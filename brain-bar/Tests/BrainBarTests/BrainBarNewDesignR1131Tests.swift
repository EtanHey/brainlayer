import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// Revision 1.13.1 (the M1 capture of 1.13): the waiting state after Reconnect is one fact every surface reads, the
/// dropdown's item included; the button hit area is rectangular as hygiene; the agents' panel covers
/// no heading; Copied never shows two labels in one frame.
@MainActor
final class BrainBarNewDesignR1131Tests: XCTestCase {
    typealias Fixture = BrainBarDesignFixture
    typealias Say = BrainBarMenuPresenter.SurfaceSay
    typealias Kind = BrainBarMenuPresentation.ActionKind

    // MARK: 1. one waiting fact

    /// The surfaces that offer Reconnect, each pressed through its own entry.
    enum Surface: String, CaseIterable { case row, station, panel, dropdown, settings, backups }

    static func menu(_ store: NDStore) -> NSMenu {
        NDMenuBuilder(store: store, perform: { action in
            if case let .row(kind) = action { store.perform(kind) }
        }, openWindow: { _ in }).build()
    }

    static func reconnectIndex(_ menu: NSMenu) -> Int? {
        menu.items.firstIndex { item in
            guard case let .row(kind)? = item.representedObject as? BrainBarMenuAction else { return false }
            return kind == .reconnectDrive || kind == .connectDrive
        }
    }

    static func press(_ surface: Surface, _ store: NDStore) throws {
        let r1 = store.r1
        switch surface {
        case .row: store.perform(try XCTUnwrap(r1.problemRows.first { $0.action?.kind == .reconnectDrive }?.action).kind)
        case .station, .panel: store.perform(try XCTUnwrap(r1.stationItems[.drive]?.action).kind)
        case .dropdown:
            // The status item holds its builder for as long as it lives; an item's target is weak.
            let builder = NDMenuBuilder(store: store, perform: { action in
                if case let .row(kind) = action { store.perform(kind) }
            }, openWindow: { _ in })
            let menu = builder.build()
            menu.performActionForItem(at: try XCTUnwrap(Self.reconnectIndex(menu), "the dropdown offers Reconnect"))
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
            withExtendedLifetime(builder) {}
        case .settings: store.reconnectDrive()
        case .backups: store.perform(try XCTUnwrap(r1.items.first { $0.id == "drive" }?.action).kind)
        }
    }

    /// Whether each surface, built now, waits (shows the waiting words and offers no Reconnect).
    static func waits(_ surface: Surface, _ store: NDStore) -> Bool {
        let r1 = store.r1, i = store.inputs
        switch surface {
        case .row:
            guard let action = r1.problemRows.first(where: { $0.action?.kind == .reconnectDrive })?.action else { return false }
            return NDDriveWaiting.shows(i, for: action)
        case .station: return r1.pipeline.status[.drive]?.text == Say.stationWaiting
        case .panel: return NDDriveWaiting.shows(i, for: r1.stationItems[.drive]?.action)
        case .dropdown:
            let menu = Self.menu(store)
            return Self.reconnectIndex(menu) == nil && menu.items.contains { $0.title == Say.waitingForBrowser }
        case .settings:
            guard let action = NDSettingsView.driveAction(r1) else { return false }
            return NDDriveWaiting.shows(i, for: action)
        case .backups:
            guard let action = r1.items.first(where: { $0.id == "drive" })?.action else { return false }
            return NDDriveWaiting.shows(i, for: action)
        }
    }

    /// A press in any one surface, then every surface built afterwards waits: one fact (the store's), read by all six.
    /// On 9bcd6388 the dropdown built afterwards still offered "Reconnect Google Drive" as its item (type-select, Return
    /// and VoiceOver reached it) while its view drew the waiting line.
    func testAPressInAnySurfaceMakesEverySurfaceBuiltAfterwardsWait() throws {
        for pressed in Surface.allCases {
            let store = NDStore(fixture: Fixture.inputs(.driveSignedOut), page: .overview)
            for surface in Surface.allCases {
                XCTAssertFalse(Self.waits(surface, store), "before any press, \(surface) offers Reconnect")
            }
            try Self.press(pressed, store)
            for surface in Surface.allCases {
                XCTAssertTrue(Self.waits(surface, store), "pressed in \(pressed): \(surface) built afterwards waits")
            }
        }
    }

    /// The live path: the one fact is the Drive model's sign-in in progress, sampled by the one store every surface reads.
    func testTheLiveWaitingFactIsTheDriveSignInInProgress() async throws {
        final class Held: BrainLayerCLIRunning, @unchecked Sendable {
            let gate = DispatchSemaphore(value: 0)
            func run(_ arguments: [String], timeout: TimeInterval) -> BrainLayerCLIResult? {
                gate.wait()
                return .init(terminationStatus: 0, stdout: #"{"status":"cancelled"}"#)
            }
        }
        let runtime = BrainBarRuntime()
        let runner = Held()
        runtime.driveAuth.install(runner: runner)
        let store = NDStore(runtime: runtime, workspace: BrainBarRecordingWorkspace())
        store.setIconActive(true)
        defer { store.setIconActive(false) }
        XCTAssertFalse(store.inputs.driveReconnecting)
        store.perform(.reconnectDrive)
        for _ in 0..<200 where !store.inputs.driveReconnecting { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(runtime.driveAuth.isReconnecting)
        XCTAssertTrue(store.inputs.driveReconnecting, "the store every surface reads waits")
        store.reconnectDrive()  // a second surface's press starts nothing new
        runner.gate.signal()
        for _ in 0..<200 where store.inputs.driveReconnecting { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(runtime.driveAuth.isReconnecting)
        XCTAssertFalse(store.inputs.driveReconnecting, "the sign-in ended: every surface offers Reconnect again")
    }

    // MARK: 2. the exact centre line

    static func vertices(_ path: Path) -> [CGPoint] {
        var points: [CGPoint] = []
        path.forEach { element in
            switch element {
            case let .move(to: p), let .line(to: p): points.append(p)
            case let .curve(to: p, _, _), let .quadCurve(to: p, _): points.append(p)
            case .closeSubpath: break
            }
        }
        return points
    }

    /// Geometry hygiene only: Path.contains checks the rectangle, not a delivered SwiftUI click.
    /// The missed centre clicks remain unexplained.
    func testTheButtonTakesAClickOnItsExactCentreLine() throws {
        for height in [20.0, 22.0, 24.0] {
            for width in [60.0, 86.5, 117.0, 196.0] {
                let rect = CGRect(x: 0, y: 0, width: width, height: height)
                let hit = NDButtonStyle.HitShape().path(in: rect)
                let centre = CGPoint(x: rect.midX, y: rect.midY)
                XCTAssertTrue(hit.contains(centre), "\(width)x\(height)")
                for dy in [-0.5, 0.5] { XCTAssertTrue(hit.contains(CGPoint(x: centre.x, y: centre.y + dy))) }
                XCTAssertFalse(Self.vertices(hit).contains { abs($0.y - rect.midY) < 0.001 }, "\(width)x\(height): a vertex on the centre line")
                let drawn = RoundedRectangle(cornerRadius: 6, style: .continuous).path(in: rect)
                XCTAssertTrue(Self.vertices(drawn).contains { abs($0.y - rect.midY) < 0.001 }, "the drawn shape has a centre-line vertex; this does not establish a click failure")
            }
        }
        let source = try Self.source("NDWindow.swift")
        XCTAssertTrue(source.contains(".contentShape(HitShape())"), "the button hit-tests the rectangle")
        for file in try FileManager.default.contentsOfDirectory(atPath: Self.newDesign.path) where file.hasSuffix(".swift") {
            XCTAssertFalse(try Self.source(file).contains(".contentShape(RoundedRectangle"), "\(file): a rounded hit area")
        }
    }

    // MARK: 3. small ones

    /// No station's panel covers the "Pipeline" heading, in any fixture (the agents' did, over its left).
    func testNoPanelCoversThePipelineHeading() {
        let L = NDPipelineLayout.self
        for name in Fixture.Name.allCases {
            let store = NDStore(fixture: Fixture.inputs(name), page: .overview)
            let chart = NDPipelineChart(state: store.r1.pipeline, memories: store.model.memories, items: store.r1.stationItems)
            for station in NDStation.allCases {
                let panel = chart.panel(station, content: AnyView(NDStationPopover(store: store, station: station)))
                XCTAssertFalse(panel.outer.intersects(L.pipelineHeading), "\(name.rawValue) \(station): \(panel.outer)")
            }
        }
    }

    /// Copied changes face in two halves (out, swap, in), never a cross-fade; under Reduce Motion it swaps at once.
    func testCopiedNeverShowsTwoLabelsInOneFrame() throws {
        XCTAssertEqual(NDCopyReportButton.steps(reduceMotion: false), [.fadeOut, .swap, .fadeIn])
        XCTAssertEqual(NDCopyReportButton.steps(reduceMotion: true), [.swap])
        let source = try Self.source("NDStationPanel.swift")
        XCTAssertFalse(source.contains("withAnimation(Self.fade) { copied"), "the label's text is never animated")
        XCTAssertTrue(source.contains("swap.disablesAnimations = true"), "the swap itself carries no animation")
    }

    static let newDesign = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().appendingPathComponent("Sources/BrainBar/NewDesign")

    static func source(_ file: String) throws -> String {
        try String(contentsOf: newDesign.appendingPathComponent(file), encoding: .utf8)
    }
}
