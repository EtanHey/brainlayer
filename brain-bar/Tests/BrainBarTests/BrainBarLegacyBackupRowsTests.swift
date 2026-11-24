import AppKit
import SwiftUI
import XCTest
@testable import BrainBar

@MainActor final class BrainBarLegacyBackupRowsTests: XCTestCase {
    private let now = BrainBarDesignFixture.now
    private let legacy = #"{"already_covered_files":4,"discovered_file_count":4,"message":"No new files","skipped_active_count":0,"status":"no-op","uploaded":false,"verified":false}"#

    private func upload(_ hours: Double) -> String {
        let at = ISO8601DateFormatter().string(from: now.addingTimeInterval(-hours * 3600))
        return "{\"attempted_at\":\"\(at)\",\"uploaded\":true,\"verified\":true}"
    }

    private var legacyHead: [String] { [legacy, legacy, legacy, upload(400), upload(250), upload(6), upload(3)] }

    private func parse(_ lines: [String]) throws -> (attempts: [NDAttempt]?, skipped: Int) {
        try XCTUnwrap(NDBackupHistoryReader.parseConversations(
            .init(lines: lines.map { Substring($0) }, unreadable: false), now: now))
    }

    private func store(_ attempts: [NDAttempt]?) -> NDStore {
        var inputs = BrainBarDesignFixture.inputs(.calm)
        inputs.backupHistory = .read(.init(local: [], drive: [], conversations: attempts, integrity: []))
        return NDStore(fixture: inputs, page: .backups)
    }

    func testLegacyHeadBeforeOldRowsPreservesRecentAttempts() throws {
        let result = try parse(legacyHead)
        XCTAssertEqual(result.skipped, 0)
        XCTAssertEqual(result.attempts?.map(\.at), [now.addingTimeInterval(-6 * 3600), now.addingTimeInterval(-3 * 3600)])
        let store = store(result.attempts)
        let lane = try XCTUnwrap(NDBackupsPage(store: store).lanes(store.model).first { $0.id == "transcripts" })
        XCTAssertFalse(lane.unknown)
        XCTAssertEqual(lane.events.filter { $0.kind == .ok }.count, 2)
    }

    func testUndatedBetweenRecentRowsAndAtEOFStayUnread() throws {
        for lines in [[upload(6), legacy, upload(3)], [upload(400), legacy], [legacy]] {
            let result = try parse(lines)
            XCTAssertNil(result.attempts)
            XCTAssertEqual(result.skipped, 1)
        }
    }

    func testUndatedErrorBeforeOldRowsStaysUnread() throws {
        for error in [#""inventory failed""#, "true", "7"] {
            let result = try parse(["{\"error\":\(error)}", upload(400), upload(3)])
            XCTAssertNil(result.attempts)
            XCTAssertEqual(result.skipped, 1)
        }
        for error in [#"" ""#, "null"] {
            XCTAssertEqual(try parse(["{\"error\":\(error)}", upload(400), upload(3)]).skipped, 0)
        }
    }

    func testNextDatedRowControlsAgeAndWindowBoundaryStaysUnread() throws {
        let boundary = NDBackupHistoryReader.windowS / 3600
        for lines in [[upload(400), legacy, upload(3)], [legacy, upload(boundary)], [legacy, upload(-3)]] {
            XCTAssertNil(try parse(lines).attempts)
        }
        XCTAssertEqual(try parse([legacy, upload(boundary + 1), upload(3)]).skipped, 0)
        XCTAssertNil(try parse(["{broken", upload(400), upload(3)]).attempts)
        XCTAssertNil(try parse([#"{"attempted_at":"broken"}"#, upload(400), upload(3)]).attempts)
        for key in ["ts", "created_at", "at"] {
            XCTAssertNil(try parse(["{\"\(key)\":\"broken\"}", upload(400), upload(3)]).attempts)
        }
    }

    func testDrawnBackupsPageMatchesCleanHistoryInBothAppearances() throws {
        NDMarkView.captureClock = Date(timeIntervalSinceReferenceDate: 0.3)
        defer { NDMarkView.captureClock = nil }
        let parsed = store(try parse(legacyHead).attempts)
        let clean = store(try parse([upload(6), upload(3)]).attempts)
        let unread = store(nil)
        let size = NSSize(width: NDGrid.windowWidth, height: NDPageHeight.window(clean, page: .backups, screen: nil))
        for (_, appearance) in NDRenderHarness.appearances {
            func png(_ store: NDStore) throws -> Data? {
                try NDRenderHarness.host(NDWindowRoot(store: store, drawsTrafficLights: true)
                    .frame(width: size.width, height: size.height), size: size, appearance: appearance)
                    .representation(using: .png, properties: [:])
            }
            let drawn = try XCTUnwrap(png(parsed))
            XCTAssertEqual(drawn, try png(clean))
            XCTAssertNotEqual(drawn, try png(unread))
        }
    }

    func testRenderRequestedLegacyHeadProof() throws {
        guard let path = ProcessInfo.processInfo.environment["LEGACY_BACKUP_PROOF"] else { return }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let store = store(try parse(legacyHead).attempts)
        for (suffix, appearance) in NDRenderHarness.appearances {
            _ = try NDRenderHarness.window(store, name: "backups-legacy-head-\(suffix)", appearance: appearance, in: directory)
        }
    }
}
