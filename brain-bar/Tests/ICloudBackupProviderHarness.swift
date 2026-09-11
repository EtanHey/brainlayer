import CryptoKit
import Darwin
import Foundation

private struct Identity {
    let device: Int64
    let inode: Int64
}

private func identity(_ url: URL) -> Identity {
    var info = stat()
    precondition(lstat(url.path, &info) == 0, "missing \(url.path)")
    return Identity(device: Int64(info.st_dev), inode: Int64(info.st_ino))
}

private func digest(_ data: Data) -> String {
    Data(SHA256.hash(data: data)).map { String(format: "%02x", $0) }.joined()
}

private func bytes(_ url: URL) throws -> Data { try Data(contentsOf: url) }

private func log(_ message: String) { FileHandle.standardError.write(Data((message + "\n").utf8)) }

private func assertTrue(_ condition: @autoclosure () -> Bool, _ message: String) {
    precondition(condition(), message)
}

private final class FakeResources: @unchecked Sendable, BrainBarICloudResourceProviding {
    let sourceLabel = "injected_test"
    private let lock = NSLock()
    private var values: [String: BrainBarICloudResourceState]
    var downloads: [String] = []
    var downloadError: Error?

    init(_ values: [URL: BrainBarICloudResourceState]) {
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

private final class PivotResources: @unchecked Sendable, BrainBarICloudResourceProviding {
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

private func fakeMap(generation: URL, destination: URL, artifacts: [URL], uploaded: Bool, downloaded: Bool, uploading: Bool) -> [URL: BrainBarICloudResourceState] {
    let state = BrainBarICloudResourceState(isUbiquitous: true, isUploading: uploading, isUploaded: uploaded, isDownloaded: downloaded)
    return Dictionary(uniqueKeysWithValues: ([destination] + artifacts).map { ($0, state) })
}

private func makeGeneration(root: URL) throws -> (URL, URL) {
    let destination = root.appendingPathComponent("selected")
    let generation = destination.appendingPathComponent(".brainlayer-backup-20260911-genuine")
    try FileManager.default.createDirectory(at: generation, withIntermediateDirectories: true)
    let db = Data("generated-db\n".utf8)
    let bundle = Data("generated-jsonl-bundle\n".utf8)
    try db.write(to: generation.appendingPathComponent("brainlayer.db"))
    try bundle.write(to: generation.appendingPathComponent("jsonl.bundle.tar"))
    let receipt: [String: Any] = [
        "status": "local_verified",
        "generation": generation.lastPathComponent,
        "destination_path": destination.standardizedFileURL.path,
        "destination_identity": ["st_dev": identity(destination).device, "st_ino": identity(destination).inode],
        "db_snapshot": "brainlayer.db",
        "db_snapshot_bytes": db.count,
        "db_snapshot_sha256": digest(db),
        "jsonl_bundle": "jsonl.bundle.tar",
        "jsonl_bundle_bytes": bundle.count,
        "jsonl_bundle_sha256": digest(bundle),
        "local_verified": true,
        "remote_verified": false,
        "cleanup_permitted": false,
    ]
    let receiptData = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
    try receiptData.write(to: generation.appendingPathComponent("receipt.complete.json"))
    return (generation, destination)
}

@main
struct ICloudBackupProviderHarness {
    static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("brainlayer-icloud-harness-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let (generation, destination) = try makeGeneration(root: root)
        let providerReceipt = destination.appendingPathComponent(".brainlayer-provider/icloud-provider.receipt.json")
        let input = BrainBarICloudBackupInput(generationDirectory: generation, selectedDestination: destination, providerReceiptURL: providerReceipt)
        let receipt = try await BrainBarICloudBackupProvider(resourceProvider: BrainBarFoundationICloudResourceProvider()).process(input)

        assertTrue(receipt.state == .staged, "expected local fixture to be staged, got \(receipt.state)")
        assertTrue(receipt.destinationKind == "local", "expected local destination kind")
        assertTrue(receipt.resourceObservationSource == "foundation", "expected Foundation observation label")
        assertTrue(receipt.remoteVerified == false, "remote verification must remain false")
        assertTrue(receipt.cleanupPermitted == false, "cleanup must remain forbidden")
        assertTrue(receipt.artifacts.count == 3, "expected db, bundle, and complete receipt")
        assertTrue(FileManager.default.fileExists(atPath: generation.appendingPathComponent("brainlayer.db").path), "source generation was removed")
        assertTrue(FileManager.default.fileExists(atPath: providerReceipt.path), "provider receipt was not persisted")
        let persisted = try Data(contentsOf: providerReceipt)
        assertTrue(persisted.count < 64 * 1024, "provider receipt is not compact")
        let dbURL = generation.appendingPathComponent("brainlayer.db")
        let completeURL = generation.appendingPathComponent("receipt.complete.json")
        let dbBefore = try bytes(dbURL)
        let completeBefore = try bytes(completeURL)
        let dbPathInput = BrainBarICloudBackupInput(generationDirectory: generation, selectedDestination: destination, providerReceiptURL: dbURL)
        do {
            _ = try await BrainBarICloudBackupProvider().process(dbPathInput)
            preconditionFailure("provider accepted a source artifact as receipt output")
        } catch { }
        let dbAfterUnsafe = try bytes(dbURL)
        assertTrue(dbAfterUnsafe == dbBefore, "DB changed through an unsafe receipt path")
        let completePathInput = BrainBarICloudBackupInput(generationDirectory: generation, selectedDestination: destination, providerReceiptURL: completeURL)
        do {
            _ = try await BrainBarICloudBackupProvider().process(completePathInput)
            preconditionFailure("provider accepted complete receipt as receipt output")
        } catch { }
        let completeAfterUnsafe = try bytes(completeURL)
        assertTrue(completeAfterUnsafe == completeBefore, "complete receipt changed through an unsafe path")
        let expiredUnsafe = BrainBarICloudBackupInput(generationDirectory: generation, selectedDestination: destination, providerReceiptURL: dbURL, deadline: Date(timeIntervalSince1970: 0))
        do {
            _ = try await BrainBarICloudBackupProvider().process(expiredUnsafe)
            preconditionFailure("expired unsafe output path was accepted")
        } catch { }
        let dbAfterDeadline = try bytes(dbURL)
        assertTrue(dbAfterDeadline == dbBefore, "expired deadline changed the DB")
        let namespace = destination.appendingPathComponent(".brainlayer-provider")
        try FileManager.default.removeItem(at: namespace)
        try FileManager.default.createSymbolicLink(at: namespace, withDestinationURL: generation)
        defer { try? FileManager.default.removeItem(at: namespace) }
        do {
            _ = try await BrainBarICloudBackupProvider().process(input)
            preconditionFailure("provider accepted a symlinked receipt namespace")
        } catch { }
        let dbAfterAlias = try bytes(dbURL)
        assertTrue(dbAfterAlias == dbBefore, "symlinked namespace changed the DB")
        try FileManager.default.removeItem(at: namespace)
        let repeatFirst = try await BrainBarICloudBackupProvider().process(input)
        let repeatSecond = try await BrainBarICloudBackupProvider().process(input)
        assertTrue(repeatFirst.state == .staged && repeatSecond.state == .staged, "valid provider status update did not repeat")

        let outside = root.appendingPathComponent("outside")
        let pivot = PivotResources(destination: destination, namespace: namespace, outside: outside)
        do {
            _ = try await BrainBarICloudBackupProvider(resourceProvider: pivot).process(input)
            preconditionFailure("provider accepted a namespace pivot")
        } catch { }
        let outsideAfter = try bytes(outside.appendingPathComponent("icloud-provider.receipt.json"))
        assertTrue(outsideAfter == Data("sentinel\n".utf8), "namespace pivot changed an outside receipt")
        try FileManager.default.removeItem(at: namespace)

        let uploadedRoot = FileManager.default.temporaryDirectory.appendingPathComponent("brainlayer-icloud-uploaded-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: uploadedRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: uploadedRoot) }
        let (uploadedGeneration, uploadedDestination) = try makeGeneration(root: uploadedRoot)
        let uploadedArtifacts = ["brainlayer.db", "jsonl.bundle.tar", "receipt.complete.json"].map { uploadedGeneration.appendingPathComponent($0) }
        let uploadedResources = FakeResources(fakeMap(generation: uploadedGeneration, destination: uploadedDestination, artifacts: uploadedArtifacts, uploaded: true, downloaded: true, uploading: false))
        let uploadedInput = BrainBarICloudBackupInput(generationDirectory: uploadedGeneration, selectedDestination: uploadedDestination, providerReceiptURL: uploadedDestination.appendingPathComponent(".brainlayer-provider/icloud-provider.receipt.json"))
        let uploaded = try await BrainBarICloudBackupProvider(resourceProvider: uploadedResources).process(uploadedInput)
        assertTrue(uploaded.state == .uploadedOSReported, "expected OS-uploaded state")
        assertTrue(uploaded.artifacts.allSatisfy { $0.localSHA256 == $0.finalSHA256 }, "final hashes did not match local hashes")
        assertTrue(uploaded.remoteVerified == false && uploaded.cleanupPermitted == false, "OS state must not authorize cleanup")

        let pendingRoot = FileManager.default.temporaryDirectory.appendingPathComponent("brainlayer-icloud-pending-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: pendingRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: pendingRoot) }
        let (pendingGeneration, pendingDestination) = try makeGeneration(root: pendingRoot)
        let pendingArtifacts = ["brainlayer.db", "jsonl.bundle.tar", "receipt.complete.json"].map { pendingGeneration.appendingPathComponent($0) }
        let pendingResources = FakeResources(fakeMap(generation: pendingGeneration, destination: pendingDestination, artifacts: pendingArtifacts, uploaded: false, downloaded: false, uploading: true))
        let pending = try await BrainBarICloudBackupProvider(resourceProvider: pendingResources).process(BrainBarICloudBackupInput(generationDirectory: pendingGeneration, selectedDestination: pendingDestination, providerReceiptURL: pendingDestination.appendingPathComponent(".brainlayer-provider/icloud-provider.receipt.json")))
        assertTrue(pending.state == .uploadPending && pendingResources.downloads.count == 3, "pending upload did not request bounded materialization")

        log("ICLOUD_PROVIDER_HARNESS_OK local=staged uploaded=uploaded_os_reported pending=upload_pending source=foundation/injected_test receipt_bytes=\(persisted.count)")
    }
}
