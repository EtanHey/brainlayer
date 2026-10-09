import XCTest
@testable import BrainBar

final class BrainBarBackupLogPreviewTests: XCTestCase {
    private func present(_ text: String) -> BrainBarBackupLogPreview {
        .present(.read(Data(text.utf8), modifiedAt: .distantPast))
    }

    func testEmptyUnreadableAndChangingLogsAreDistinct() {
        XCTAssertEqual(present(" \n\t"), .empty)
        XCTAssertEqual(BrainBarBackupLogPreview.present(.unreadable), .unavailable("Log could not be read. It may not exist yet or access may be denied."))
        XCTAssertEqual(BrainBarBackupLogPreview.present(.changedWhileReading), .unavailable("Log changed while reading. Refresh to try again."))
    }

    func testLatestEntriesRetainFailuresAndDiscardOlderHistory() {
        let lines = (0 ..< 25).map { #"{"status":"failed","error_type":"TimeoutError","error":"upload timed out","timestamp":"entry-\#($0)"}"# }
        guard case let .entries(text, truncated) = present(lines.joined(separator: "\n")) else { return XCTFail() }
        XCTAssertTrue(truncated)
        XCTAssertFalse(text.contains("Time: entry-4 ·"))
        XCTAssertTrue(text.contains("Time: entry-5 ·"))
        XCTAssertTrue(text.contains("Time: entry-24 ·"))
        XCTAssertTrue(text.contains("TimeoutError"))
        XCTAssertTrue(text.contains("upload timed out"))
        XCTAssertEqual(text.components(separatedBy: "\n\n").count, 20)
    }

    func testByteTruncationDropsPartialReceiptAndPreservesNewestFailure() {
        let text = String(repeating: "x", count: BrainBarBackupLogPreview.byteLimit + 20) + "\n" + #"{"error_type":"OSError","error":"disk full","verified":false}"#
        guard case let .entries(shown, truncated) = present(text) else { return XCTFail() }
        XCTAssertTrue(truncated)
        XCTAssertFalse(shown.contains("xxx"))
        XCTAssertTrue(shown.contains("disk full"))
        XCTAssertTrue(shown.contains("Verified: No"))
        XCTAssertEqual(present(String(repeating: "x", count: BrainBarBackupLogPreview.byteLimit + 1)),
                       .unavailable("Latest entry exceeds the preview limit. Reveal the log for details."))
    }

    func testReceiptAllowlistAndRedactionKeepErrorType() {
        let input = #"{"access_token":"short-secret","nested":{"secret":"private"},"db":"/Users/private/db","error_type":"PermissionError","error":"failed /Users/private/settings.env password=abc123 https://host/?token=secret"}"#
        guard case let .entries(text, _) = present(input) else { return XCTFail() }
        for secret in ["short-secret", "nested", "/Users", "abc123", "https://", "token=secret"] {
            XCTAssertFalse(text.contains(secret), secret)
        }
        XCTAssertTrue(text.contains("PermissionError"))
        XCTAssertTrue(text.contains("failed"))
    }

    func testProviderTokensAndUnlabeledEntropyNeverEnterPreview() {
        for value in ["ya29.shortsecret", "1//shortsecret", "GOCSPX-shortsecret", "AKIA1234567890ABCDEF", "abcDEFG0123456789XYZzyx987654321qwe"] {
            XCTAssertFalse(BrainBarBackupLogPreview.redact(value).contains(value))
        }
        XCTAssertFalse(BrainBarBackupLogPreview.redact(#""token":"shortsecret""#).contains("shortsecret"))
        XCTAssertEqual(BrainBarBackupLogPreview.redact("disk\nfull"), "disk full")
        XCTAssertEqual(BrainBarBackupLogPreview.redact(#"failed "/Users/Private Person/config.env""#), "failed [private path]")
        XCTAssertEqual(BrainBarBackupLogPreview.redact(#"failed /Users/Private Person/config.env"#), "failed [private path]")
        XCTAssertFalse(BrainBarBackupLogPreview.redact(#""password":"short secret""#).contains("secret"))
        XCTAssertTrue(BrainBarBackupLogPreview.redact("FAILED abcDEFG0123456789XYZzyx987654321qwe").contains("Failure"))
        XCTAssertEqual(BrainBarBackupLogPreview.redact(#"failed C:\Users\Private\config.env"#), "failed [private path]")
    }

    func testMalformedOrUnstructuredFailuresStayVisibleWithRedaction() {
        let raw = "FAILED /Users/private/config token=shortsecret"
        guard case let .entries(text, truncated) = present(raw) else { return XCTFail() }
        XCTAssertFalse(text.contains("shortsecret"))
        XCTAssertTrue(text.contains("FAILED"))
        XCTAssertFalse(truncated)
    }

    func testRealLargeSyntheticFileIsBoundedAndMissingFileIsUnavailable() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        let record = #"{"status":"failed","error":"disk full"}"# + "\n"
        try Data(String(repeating: record, count: 50_000).utf8).write(to: url)
        guard case let .entries(text, truncated) = BrainBarBackupLogPreview.read(url) else { return XCTFail() }
        XCTAssertTrue(truncated)
        XCTAssertEqual(text.components(separatedBy: "\n\n").count, 20)
        XCTAssertLessThan(text.utf8.count, BrainBarBackupLogPreview.byteLimit)
        try FileManager.default.removeItem(at: url)
        guard case .unavailable = BrainBarBackupLogPreview.read(url) else { return XCTFail() }
    }

    @MainActor
    func testBlockedReadsDoNotBlockMainActorAndNewestRequestWins() async throws {
        let reads = PreviewReads()
        defer { reads.release(0); reads.release(1) }
        let model = BrainBarBackupLogPreviewModel(paths: paths(), read: { reads.read($0) })
        model.refresh()
        try await reads.waitForCalls(1)
        XCTAssertTrue(model.refreshing)
        XCTAssertEqual(model.preview, .unread)
        // This main-actor turn can execute while the synchronous reader is blocked.
        model.kind = .transcripts
        model.refresh()
        try await reads.waitForCalls(2)
        reads.release(1)
        for _ in 0 ..< 100 where model.refreshing { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertEqual(model.preview, .unavailable("newest"))
        reads.release(0)
        for _ in 0 ..< 20 { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertEqual(model.preview, .unavailable("newest"))
        XCTAssertFalse(model.refreshing)
    }

    @MainActor
    func testSelectionAndRevealUseInjectedJobLogPaths() async throws {
        let workspace = PreviewWorkspace()
        let paths = paths()
        let model = BrainBarBackupLogPreviewModel(paths: paths, workspace: workspace, read: { .unavailable($0.lastPathComponent) })
        model.reveal()
        model.kind = .transcripts
        model.refresh()
        model.reveal()
        for _ in 0 ..< 100 where model.refreshing { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertEqual(workspace.urls, [paths.databaseLog, paths.archiveLog])
        XCTAssertEqual(model.preview, .unavailable("transcripts.log"))
    }

    private func paths() -> BrainBarBackupSources.Paths {
        let root = URL(fileURLWithPath: "/synthetic")
        return .init(launchAgents: root, databaseLog: root.appendingPathComponent("database.log"),
                     archiveLog: root.appendingPathComponent("transcripts.log"), maintenanceLog: root,
                     snapshotDirectory: root, archiveDirectory: root)
    }
}

private final class PreviewReads: @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    private let gates = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
    func read(_ url: URL) -> BrainBarBackupLogPreview {
        let index = lock.withLock { let index = calls; calls += 1; return index }
        gates[index].wait()
        return .unavailable(index == 0 ? "oldest" : "newest")
    }
    func release(_ index: Int) { gates[index].signal() }
    func waitForCalls(_ count: Int) async throws {
        for _ in 0 ..< 100 {
            if lock.withLock({ calls >= count }) { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("Reader did not start")
    }
}

private final class PreviewWorkspace: BrainBarWorkspaceActing, @unchecked Sendable {
    var urls: [URL] = []
    func reveal(_ url: URL) { urls.append(url) }
    func copy(_ text: String) { XCTFail("Preview must not copy raw content") }
}
