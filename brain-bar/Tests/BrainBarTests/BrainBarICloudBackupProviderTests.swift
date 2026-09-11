import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import BrainBar

private final class FakeICloudResources: @unchecked Sendable, BrainBarICloudResourceProviding {
    let sourceLabel = "injected_test"
    private let lock = NSLock()
    private var values: [String: BrainBarICloudResourceState]
    var downloads: [String] = []
    var downloadError: Error?

    init(values: [URL: BrainBarICloudResourceState]) {
        self.values = Dictionary(uniqueKeysWithValues: values.map { ($0.key.standardizedFileURL.path, $0.value) })
    }

    func inspect(_ url: URL) throws -> BrainBarICloudResourceState {
        lock.lock(); defer { lock.unlock() }
        return values[url.standardizedFileURL.path] ?? BrainBarICloudResourceState(isUbiquitous: nil, isUploading: nil, isUploaded: nil, isDownloaded: nil)
    }

    func startDownloading(_ url: URL) throws {
        lock.lock(); defer { lock.unlock() }
        downloads.append(url.standardizedFileURL.path)
        if let downloadError { throw downloadError }
        let old = values[url.standardizedFileURL.path]
        values[url.standardizedFileURL.path] = BrainBarICloudResourceState(isUbiquitous: old?.isUbiquitous, isUploading: old?.isUploading, isUploaded: old?.isUploaded, isDownloaded: true)
    }
}

private enum FixtureError: Error { case missing }

private struct Fixture {
    let root: URL
    let generation: URL
    let destination: URL
    let input: BrainBarICloudBackupInput
    let artifactURLs: [URL]
}

private func fileIdentity(_ url: URL) -> (Int64, Int64) {
    var info = stat()
    precondition(lstat(url.path, &info) == 0)
    return (Int64(info.st_dev), Int64(info.st_ino))
}

private func hex(_ data: Data) -> String { Data(SHA256.hash(data: data)).map { String(format: "%02x", $0) }.joined() }

private func fixture() throws -> Fixture {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("icloud-test-\(UUID().uuidString)")
    let destination = root.appendingPathComponent("selected")
    let generation = destination.appendingPathComponent(".brainlayer-backup-test")
    try FileManager.default.createDirectory(at: generation, withIntermediateDirectories: true)
    let files = [("brainlayer.db", Data("db\n".utf8)), ("jsonl.bundle.tar", Data("bundle\n".utf8))]
    for (name, data) in files { try data.write(to: generation.appendingPathComponent(name)) }
    let db = files[0].1
    let bundle = files[1].1
    let destinationID = fileIdentity(destination)
    let json: [String: Any] = [
        "status": "local_verified", "generation": generation.lastPathComponent,
        "destination_path": destination.standardizedFileURL.path,
        "destination_identity": ["st_dev": destinationID.0, "st_ino": destinationID.1],
        "db_snapshot": "brainlayer.db", "db_snapshot_bytes": db.count, "db_snapshot_sha256": hex(db),
        "jsonl_bundle": "jsonl.bundle.tar", "jsonl_bundle_bytes": bundle.count, "jsonl_bundle_sha256": hex(bundle),
        "local_verified": true, "remote_verified": false, "cleanup_permitted": false,
    ]
    try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys]).write(to: generation.appendingPathComponent("receipt.complete.json"))
    let receiptURL = destination.appendingPathComponent(".brainlayer-provider/icloud-provider.receipt.json")
    return Fixture(root: root, generation: generation, destination: destination, input: BrainBarICloudBackupInput(generationDirectory: generation, selectedDestination: destination, providerReceiptURL: receiptURL), artifactURLs: files.map { generation.appendingPathComponent($0.0) })
}

private func resourceMap(_ fixture: Fixture, uploaded: Bool, downloaded: Bool = true, uploading: Bool = false) -> [URL: BrainBarICloudResourceState] {
    let state = BrainBarICloudResourceState(isUbiquitous: true, isUploading: uploading, isUploaded: uploaded, isDownloaded: downloaded)
    let complete = fixture.generation.appendingPathComponent("receipt.complete.json")
    return Dictionary(uniqueKeysWithValues: ([fixture.destination] + fixture.artifactURLs + [complete]).map { ($0, state) })
}

private final class PivotICloudResources: @unchecked Sendable, BrainBarICloudResourceProviding {
    let sourceLabel = "injected_race"
    private let lock = NSLock()
    private let destination: URL
    private let namespace: URL
    private let outside: URL
    private var pivoted = false

    init(destination: URL, namespace: URL, outside: URL) {
        self.destination = destination
        self.namespace = namespace
        self.outside = outside
    }

    func inspect(_ url: URL) throws -> BrainBarICloudResourceState {
        lock.lock(); defer { lock.unlock() }
        if url.standardizedFileURL == destination.standardizedFileURL && !pivoted {
            try FileManager.default.removeItem(at: namespace)
            try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
            try Data("sentinel\n".utf8).write(to: outside.appendingPathComponent("icloud-provider.receipt.json"))
            try FileManager.default.createSymbolicLink(at: namespace, withDestinationURL: outside)
            pivoted = true
        }
        return BrainBarICloudResourceState(isUbiquitous: false, isUploading: false, isUploaded: false, isDownloaded: true)
    }

    func startDownloading(_ url: URL) throws {}
}

@MainActor
final class BrainBarICloudBackupProviderTests: XCTestCase {
    func testFoundationLocalFixtureIsStagedAndKeepsGeneration() async throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let receipt = try await BrainBarICloudBackupProvider().process(f.input)
        XCTAssertEqual(receipt.state, .staged)
        XCTAssertEqual(receipt.destinationKind, "local")
        XCTAssertEqual(receipt.resourceObservationSource, "foundation")
        XCTAssertFalse(receipt.remoteVerified)
        XCTAssertFalse(receipt.cleanupPermitted)
        XCTAssertTrue(FileManager.default.fileExists(atPath: f.generation.appendingPathComponent("brainlayer.db").path))
    }

    func testUploadedOSStateReportsHashesWithoutRemoteVerification() async throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let provider = BrainBarICloudBackupProvider(resourceProvider: FakeICloudResources(values: resourceMap(f, uploaded: true)))
        let receipt = try await provider.process(f.input)
        XCTAssertEqual(receipt.state, .uploadedOSReported)
        XCTAssertEqual(receipt.artifacts.count, 3)
        XCTAssertTrue(receipt.artifacts.allSatisfy { $0.finalSHA256 == $0.localSHA256 })
        XCTAssertFalse(receipt.remoteVerified)
        XCTAssertFalse(receipt.cleanupPermitted)
    }

    func testPendingMaterializesAndRecordsInjectedSource() async throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let providerResources = FakeICloudResources(values: resourceMap(f, uploaded: false, downloaded: false, uploading: true))
        let provider = BrainBarICloudBackupProvider(resourceProvider: providerResources)
        let receipt = try await provider.process(f.input)
        XCTAssertEqual(receipt.state, .uploadPending)
        XCTAssertEqual(receipt.resourceObservationSource, "injected_test")
        XCTAssertEqual(providerResources.downloads.count, 3)
    }

    func testChangedBytesFailAndRetainOriginalGeneration() async throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        try Data("tampered\n".utf8).write(to: f.artifactURLs[0])
        let receipt = try await BrainBarICloudBackupProvider(resourceProvider: FakeICloudResources(values: resourceMap(f, uploaded: true))).process(f.input)
        XCTAssertEqual(receipt.state, .failed)
        XCTAssertTrue(receipt.failureReason?.contains("digest_mismatch") == true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: f.generation.path))
    }

    func testNamespacePivotCannotRedirectReceiptOutsideDestination() async throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        _ = try await BrainBarICloudBackupProvider().process(f.input)
        let outside = f.root.appendingPathComponent("outside")
        let pivot = PivotICloudResources(destination: f.destination, namespace: f.destination.appendingPathComponent(".brainlayer-provider"), outside: outside)
        do {
            _ = try await BrainBarICloudBackupProvider(resourceProvider: pivot).process(f.input)
            XCTFail("accepted namespace pivot")
        } catch { }
        XCTAssertEqual(try Data(contentsOf: outside.appendingPathComponent("icloud-provider.receipt.json")), Data("sentinel\n".utf8))
    }

    func testPlaceholderAndDeadlineFailClosed() async throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let placeholder = f.generation.appendingPathComponent("jsonl.bundle.tar")
        try FileManager.default.removeItem(at: placeholder)
        try Data("placeholder\n".utf8).write(to: f.generation.appendingPathComponent("snapshot.icloud"))
        let providerResources = resourceMap(f, uploaded: true)
        let changedReceipt = try JSONSerialization.data(withJSONObject: [
            "status": "local_verified", "generation": f.generation.lastPathComponent,
            "destination_path": f.destination.standardizedFileURL.path,
            "destination_identity": ["st_dev": fileIdentity(f.destination).0, "st_ino": fileIdentity(f.destination).1],
            "db_snapshot": "brainlayer.db", "db_snapshot_bytes": 3, "db_snapshot_sha256": hex(Data("db\n".utf8)),
            "jsonl_bundle": "snapshot.icloud", "jsonl_bundle_bytes": 12, "jsonl_bundle_sha256": hex(Data("placeholder\n".utf8)),
            "local_verified": true, "remote_verified": false, "cleanup_permitted": false,
        ], options: [.sortedKeys])
        try changedReceipt.write(to: f.generation.appendingPathComponent("receipt.complete.json"))
        let placeholderReceipt = try await BrainBarICloudBackupProvider(resourceProvider: FakeICloudResources(values: providerResources)).process(f.input)
        XCTAssertEqual(placeholderReceipt.state, .failed)
        XCTAssertTrue(placeholderReceipt.failureReason?.contains("icloud_placeholder") == true)

        let deadlineReceipt = try await BrainBarICloudBackupProvider().process(BrainBarICloudBackupInput(generationDirectory: f.generation, selectedDestination: f.destination, providerReceiptURL: f.input.providerReceiptURL, deadline: Date(timeIntervalSince1970: 0)))
        XCTAssertEqual(deadlineReceipt.failureReason, "deadline_exceeded")
    }

    func testReceiptOutputCannotOverwriteGenerationOrFollowAlias() async throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let db = f.generation.appendingPathComponent("brainlayer.db")
        let complete = f.generation.appendingPathComponent("receipt.complete.json")
        let dbBefore = try Data(contentsOf: db)
        let completeBefore = try Data(contentsOf: complete)
        for unsafeURL in [db, complete] {
            do {
                _ = try await BrainBarICloudBackupProvider().process(BrainBarICloudBackupInput(generationDirectory: f.generation, selectedDestination: f.destination, providerReceiptURL: unsafeURL))
                XCTFail("accepted unsafe provider receipt URL")
            } catch { }
        }
        XCTAssertEqual(try Data(contentsOf: db), dbBefore)
        XCTAssertEqual(try Data(contentsOf: complete), completeBefore)

        let namespace = f.destination.appendingPathComponent(".brainlayer-provider")
        try FileManager.default.createSymbolicLink(at: namespace, withDestinationURL: f.generation)
        defer { try? FileManager.default.removeItem(at: namespace) }
        do {
            _ = try await BrainBarICloudBackupProvider().process(f.input)
            XCTFail("accepted symlinked provider receipt namespace")
        } catch { }
        XCTAssertEqual(try Data(contentsOf: db), dbBefore)
    }
}
