import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

/// w307 proof renders: the agents column at rest and hovered, for the running-agent counts the owner asked to see. Off
/// screen only (`NDRenderHarness.host`), and only when `BRAINBAR_W307_RENDER=<absolute dir>` names where to write them.
@MainActor final class BrainBarAgentsPerHarnessRenderTests: XCTestCase {
    typealias Fixture = BrainBarDesignFixture

    /// The states the PR's table names: every harness idle; a mix; 2- and 3-digit counts; a count on a faulted provider;
    /// the monitor unable to measure.
    static let scenarios: [(name: String, fixture: Fixture.Name, running: [AgentFamily: Int?])] = [
        ("zero", .calm, [.claude: 0, .codex: 0, .cursor: 0, .gemini: 0]),
        ("mixed", .calm, [.claude: 7, .codex: 2, .cursor: 0, .gemini: 1]),
        ("wide", .calm, [.claude: 128, .codex: 12, .cursor: 3, .gemini: 999]),
        ("fault", .agentNotRead, [.claude: 4, .codex: 23, .cursor: 0, .gemini: 0]),
        // #1899 R1 (N2): a count beside the faulted provider's ring and badge (the fixture faults Cursor).
        ("fault-count", .agentNotRead, [.claude: 0, .codex: 0, .cursor: 12, .gemini: 0]),
        ("unmeasured", .calm, [.claude: nil, .codex: nil, .cursor: nil, .gemini: nil]),
    ]

    /// The fixture with these running counts (nil: liveness unknown).
    static func inputs(_ fixture: Fixture.Name, _ running: [AgentFamily: Int?]) -> NDInputs {
        var i = Fixture.inputs(fixture)
        i.agents = NDAgents(agents: i.agents.agents.map { agent in
            var a = agent
            a.running = running[agent.family] ?? nil
            return a
        })
        return i
    }

    func testRenderTheAgentsColumnStates() throws {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_W307_RENDER"], NSString(string: path).isAbsolutePath else {
            throw XCTSkip("set BRAINBAR_W307_RENDER=<absolute dir> to write the w307 renders")
        }
        let dir = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (suffix, appearance) in NDRenderHarness.appearances {
            for scenario in Self.scenarios {
                let store = NDStore(fixture: Self.inputs(scenario.fixture, scenario.running), page: .overview)
                _ = try NDRenderHarness.hover(store, station: .agents, name: "agents-\(scenario.name)-hover-\(suffix)",
                                              appearance: appearance, in: dir)
            }
        }
    }
}
