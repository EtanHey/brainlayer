import CryptoKit
import Darwin
import Foundation

public struct BrainBarICloudResourceState: Codable, Equatable, Sendable {
    public let isUbiquitous: Bool?
    public let isUploading: Bool?
    public let isUploaded: Bool?
    public let isDownloaded: Bool?
    public let errorDescription: String?

    public init(isUbiquitous: Bool?, isUploading: Bool?, isUploaded: Bool?, isDownloaded: Bool?, errorDescription: String? = nil) {
        self.isUbiquitous = isUbiquitous
        self.isUploading = isUploading
        self.isUploaded = isUploaded
        self.isDownloaded = isDownloaded
        self.errorDescription = errorDescription
    }
}

public protocol BrainBarICloudResourceProviding: Sendable {
    /// Identifies whether observations came from Foundation or an injected test provider.
    var sourceLabel: String { get }
    func inspect(_ url: URL) throws -> BrainBarICloudResourceState
    func startDownloading(_ url: URL) throws
}

public struct BrainBarFoundationICloudResourceProvider: BrainBarICloudResourceProviding, Sendable {
    public let sourceLabel = "foundation"

    public init() {}

    public func inspect(_ url: URL) throws -> BrainBarICloudResourceState {
        let keys: Set<URLResourceKey> = [
            .isUbiquitousItemKey,
            .ubiquitousItemIsUploadingKey,
            .ubiquitousItemIsUploadedKey,
            .ubiquitousItemDownloadingStatusKey,
        ]
        let values = try url.resourceValues(forKeys: keys)
        return BrainBarICloudResourceState(
            isUbiquitous: values.isUbiquitousItem ?? false,
            isUploading: values.ubiquitousItemIsUploading,
            isUploaded: values.ubiquitousItemIsUploaded,
            isDownloaded: values.ubiquitousItemDownloadingStatus.map { $0 == .current }
        )
    }

    public func startDownloading(_ url: URL) throws {
        try FileManager.default.startDownloadingUbiquitousItem(at: url)
    }
}

public enum BrainBarICloudBackupState: String, Codable, Equatable, Sendable {
    case staged
    case uploadPending = "upload_pending"
    case uploadedOSReported = "uploaded_os_reported"
    case failed
}

public struct BrainBarICloudBackupInput: Sendable {
    public let generationDirectory: URL
    public let selectedDestination: URL
    public let providerReceiptURL: URL
    public let deadline: Date?

    public init(generationDirectory: URL, selectedDestination: URL, providerReceiptURL: URL, deadline: Date? = nil) {
        self.generationDirectory = generationDirectory
        self.selectedDestination = selectedDestination
        self.providerReceiptURL = providerReceiptURL
        self.deadline = deadline
    }
}

public struct BrainBarICloudBackupArtifact: Codable, Equatable, Sendable {
    public let name: String
    public let bytes: Int64
    public let localSHA256: String
    public let finalSHA256: String?
    public let resourceState: BrainBarICloudResourceState?

    public init(name: String, bytes: Int64, localSHA256: String, finalSHA256: String?, resourceState: BrainBarICloudResourceState?) {
        self.name = name
        self.bytes = bytes
        self.localSHA256 = localSHA256
        self.finalSHA256 = finalSHA256
        self.resourceState = resourceState
    }
}

public struct BrainBarICloudBackupProviderReceipt: Codable, Equatable, Sendable {
    public let schema: Int
    public let state: BrainBarICloudBackupState
    public let destinationKind: String
    public let selectedDestinationPath: String
    public let generation: String?
    public let artifacts: [BrainBarICloudBackupArtifact]
    public let remoteVerified: Bool
    public let cleanupPermitted: Bool
    public let observedAt: Date
    public let failureReason: String?
    public let resourceObservationSource: String?

    public init(schema: Int = 1, state: BrainBarICloudBackupState, destinationKind: String = "icloud", selectedDestinationPath: String, generation: String?, artifacts: [BrainBarICloudBackupArtifact], remoteVerified: Bool = false, cleanupPermitted: Bool = false, observedAt: Date = Date(), failureReason: String? = nil, resourceObservationSource: String? = nil) {
        self.schema = schema
        self.state = state
        self.destinationKind = destinationKind
        self.selectedDestinationPath = selectedDestinationPath
        self.generation = generation
        self.artifacts = artifacts
        self.remoteVerified = remoteVerified
        self.cleanupPermitted = cleanupPermitted
        self.observedAt = observedAt
        self.failureReason = failureReason
        self.resourceObservationSource = resourceObservationSource
    }
}

public struct BrainBarICloudBackupProvider: Sendable {
    public static let maxReceiptBytes = 256 * 1024
    public static let hashChunkBytes = 1024 * 1024
    private let resourceProvider: any BrainBarICloudResourceProviding
    private let now: @Sendable () -> Date

    public init(resourceProvider: any BrainBarICloudResourceProviding = BrainBarFoundationICloudResourceProvider(), now: @escaping @Sendable () -> Date = Date.init) {
        self.resourceProvider = resourceProvider
        self.now = now
    }

    public func process(_ input: BrainBarICloudBackupInput) async throws -> BrainBarICloudBackupProviderReceipt {
        try Task.checkCancellation()
        if let deadline = input.deadline, now() >= deadline {
            return try persist(failure: "deadline_exceeded", input: input, generation: nil, artifacts: [])
        }
        do {
            let checked = try await Task.detached(priority: .utility) {
                try Self.validateAndHash(input)
            }.value
            try Task.checkCancellation()
            if let deadline = input.deadline, now() >= deadline {
                return try persist(failure: "deadline_exceeded", input: input, generation: checked.generation, artifacts: checked.artifacts)
            }
            let destinationState = try resourceProvider.inspect(input.selectedDestination)
            if let error = destinationState.errorDescription {
                return try persist(failure: "destination_resource_error: \(error)", input: input, generation: checked.generation, artifacts: checked.artifacts)
            }
            guard destinationState.isUbiquitous != nil else {
                return try persist(failure: "destination_resource_state_unknown", input: input, generation: checked.generation, artifacts: checked.artifacts)
            }
            guard destinationState.isUbiquitous == true else {
                return try persist(state: .staged, input: input, generation: checked.generation, artifacts: checked.artifacts, destinationKind: "local")
            }
            let finalDirectory: URL
            if input.generationDirectory.deletingLastPathComponent().standardizedFileURL == input.selectedDestination.standardizedFileURL {
                finalDirectory = input.generationDirectory
            } else {
                finalDirectory = try copyGeneration(checked: checked, input: input)
            }
            let finalArtifacts = try await inspectAndHashFinal(checked: checked, finalDirectory: finalDirectory, input: input)
            let allUploaded = finalArtifacts.allSatisfy { $0.resourceState?.isUploaded == true }
            let uploading = finalArtifacts.contains { $0.resourceState?.isUploading == true || $0.resourceState?.isUploaded != true }
            return try persist(
                state: allUploaded ? .uploadedOSReported : (uploading ? .uploadPending : .uploadPending),
                input: input,
                generation: checked.generation,
                artifacts: finalArtifacts,
                destinationKind: "icloud"
            )
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return try persist(failure: Self.failureMessage(error), input: input, generation: nil, artifacts: [])
        }
    }

    private struct CheckedGeneration: Sendable {
        let generation: String
        let destinationPath: String
        let artifacts: [BrainBarICloudBackupArtifact]
        let sourceURLs: [URL]
        let receiptURL: URL
    }

    private struct CompleteReceipt: Decodable, Sendable {
        let status: String
        let generation: String
        let destinationPath: String
        let destinationIdentity: DestinationIdentity
        let dbSnapshot: String
        let dbSnapshotBytes: Int64
        let dbSnapshotSHA256: String
        let jsonlBundle: String
        let jsonlBundleBytes: Int64
        let jsonlBundleSHA256: String
        let localVerified: Bool
        let remoteVerified: Bool
        let cleanupPermitted: Bool

        enum CodingKeys: String, CodingKey {
            case status, generation, destinationPath = "destination_path", destinationIdentity = "destination_identity"
            case dbSnapshot = "db_snapshot", dbSnapshotBytes = "db_snapshot_bytes", dbSnapshotSHA256 = "db_snapshot_sha256"
            case jsonlBundle = "jsonl_bundle", jsonlBundleBytes = "jsonl_bundle_bytes", jsonlBundleSHA256 = "jsonl_bundle_sha256"
            case localVerified = "local_verified", remoteVerified = "remote_verified", cleanupPermitted = "cleanup_permitted"
        }
    }

    private struct DestinationIdentity: Decodable, Sendable {
        let stDev: Int64
        let stIno: Int64
        enum CodingKeys: String, CodingKey { case stDev = "st_dev", stIno = "st_ino" }
    }

    private static func validateAndHash(_ input: BrainBarICloudBackupInput) throws -> CheckedGeneration {
        let generationInfo = try regularIdentity(input.generationDirectory, directory: true)
        _ = generationInfo
        let destinationInfo = try regularIdentity(input.selectedDestination, directory: true)
        let receiptData = try boundedData(input.generationDirectory.appendingPathComponent("receipt.complete.json"))
        let decoder = JSONDecoder()
        let receipt = try decoder.decode(CompleteReceipt.self, from: receiptData)
        let generation = input.generationDirectory.lastPathComponent
        guard receipt.status == "local_verified", receipt.localVerified, !receipt.remoteVerified, !receipt.cleanupPermitted else { throw ProviderFailure.invalidReceipt("receipt is not complete and local-only") }
        guard receipt.generation == generation else { throw ProviderFailure.invalidReceipt("generation identity mismatch") }
        guard URL(fileURLWithPath: receipt.destinationPath).standardizedFileURL == input.selectedDestination.standardizedFileURL else { throw ProviderFailure.invalidReceipt("destination path mismatch") }
        guard receipt.destinationIdentity.stDev == destinationInfo.device, receipt.destinationIdentity.stIno == destinationInfo.inode else { throw ProviderFailure.invalidReceipt("destination identity mismatch") }
        let names = [receipt.dbSnapshot, receipt.jsonlBundle, "receipt.complete.json"]
        guard Set(names).count == names.count, names.allSatisfy(safeName) else { throw ProviderFailure.invalidReceipt("artifact name is unsafe") }
        let specs: [(String, Int64, String?)] = [(receipt.dbSnapshot, receipt.dbSnapshotBytes, receipt.dbSnapshotSHA256), (receipt.jsonlBundle, receipt.jsonlBundleBytes, receipt.jsonlBundleSHA256), ("receipt.complete.json", Int64(receiptData.count), nil)]
        var artifacts: [BrainBarICloudBackupArtifact] = []
        var urls: [URL] = []
        for (name, expectedBytes, expectedHash) in specs {
            let url = input.generationDirectory.appendingPathComponent(name)
            let identity = try regularIdentity(url, directory: false)
            let actual = try hashAndSize(url)
            guard identity.device == generationInfo.device else { throw ProviderFailure.sourceChanged(name) }
            guard actual.bytes == expectedBytes else { throw ProviderFailure.digestMismatch(name) }
            if let expectedHash, actual.sha256 != expectedHash { throw ProviderFailure.digestMismatch(name) }
            artifacts.append(BrainBarICloudBackupArtifact(name: name, bytes: actual.bytes, localSHA256: actual.sha256, finalSHA256: nil, resourceState: nil))
            urls.append(url)
        }
        return CheckedGeneration(generation: generation, destinationPath: input.selectedDestination.standardizedFileURL.path, artifacts: artifacts, sourceURLs: urls, receiptURL: input.generationDirectory.appendingPathComponent("receipt.complete.json"))
    }

    private func inspectAndHashFinal(checked: CheckedGeneration, finalDirectory: URL, input: BrainBarICloudBackupInput) async throws -> [BrainBarICloudBackupArtifact] {
        var result: [BrainBarICloudBackupArtifact] = []
        for localArtifact in checked.artifacts {
            try Task.checkCancellation()
            if let deadline = input.deadline, now() >= deadline { throw ProviderFailure.deadline }
            let finalURL = finalDirectory.appendingPathComponent(localArtifact.name)
            if finalURL.pathExtension == "icloud" { throw ProviderFailure.placeholder(finalURL.lastPathComponent) }
            var state = try resourceProvider.inspect(finalURL)
            if let error = state.errorDescription { throw ProviderFailure.resourceError(finalURL.lastPathComponent, error) }
            guard state.isUbiquitous == true else { throw ProviderFailure.resourceState(finalURL.lastPathComponent) }
            if state.isDownloaded != true {
                try resourceProvider.startDownloading(finalURL)
                state = try resourceProvider.inspect(finalURL)
                if state.isDownloaded != true { throw ProviderFailure.materialization(finalURL.lastPathComponent) }
            }
            let actual = try await Task.detached(priority: .utility) { try Self.hashAndSize(finalURL) }.value
            guard actual.bytes == localArtifact.bytes, actual.sha256 == localArtifact.localSHA256 else { throw ProviderFailure.digestMismatch(finalURL.lastPathComponent) }
            result.append(BrainBarICloudBackupArtifact(name: localArtifact.name, bytes: actual.bytes, localSHA256: localArtifact.localSHA256, finalSHA256: actual.sha256, resourceState: state))
        }
        return result
    }

    private func copyGeneration(checked: CheckedGeneration, input: BrainBarICloudBackupInput) throws -> URL {
        let final = input.selectedDestination.appendingPathComponent(checked.generation)
        guard !FileManager.default.fileExists(atPath: final.path) else { throw ProviderFailure.staleDestination }
        let temporary = input.selectedDestination.appendingPathComponent(".brainlayer-icloud-upload-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
        do {
            for artifact in checked.artifacts {
                let source = input.generationDirectory.appendingPathComponent(artifact.name)
                let destination = temporary.appendingPathComponent(artifact.name)
                _ = try Self.regularIdentity(source, directory: false)
                try FileManager.default.copyItem(at: source, to: destination)
            }
            let marker = temporary.appendingPathComponent("icloud-upload.complete")
            try Data("complete\n".utf8).write(to: marker, options: .atomic)
            try FileManager.default.moveItem(at: temporary, to: final)
            return final
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
    }

    private func persist(state: BrainBarICloudBackupState, input: BrainBarICloudBackupInput, generation: String?, artifacts: [BrainBarICloudBackupArtifact], destinationKind: String = "icloud") throws -> BrainBarICloudBackupProviderReceipt {
        let receipt = BrainBarICloudBackupProviderReceipt(state: state, destinationKind: destinationKind, selectedDestinationPath: input.selectedDestination.standardizedFileURL.path, generation: generation, artifacts: artifacts, observedAt: now(), resourceObservationSource: resourceProvider.sourceLabel)
        try write(receipt, to: input.providerReceiptURL)
        return receipt
    }

    private func persist(failure: String, input: BrainBarICloudBackupInput, generation: String?, artifacts: [BrainBarICloudBackupArtifact]) throws -> BrainBarICloudBackupProviderReceipt {
        let receipt = BrainBarICloudBackupProviderReceipt(state: .failed, selectedDestinationPath: input.selectedDestination.standardizedFileURL.path, generation: generation, artifacts: artifacts, observedAt: now(), failureReason: failure, resourceObservationSource: resourceProvider.sourceLabel)
        try write(receipt, to: input.providerReceiptURL)
        return receipt
    }

    private func write(_ receipt: BrainBarICloudBackupProviderReceipt, to url: URL) throws {
        let data = try JSONEncoder().encode(receipt)
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let temporary = directory.appendingPathComponent(".icloud-provider-\(UUID().uuidString).tmp")
        try data.write(to: temporary, options: .atomic)
        if FileManager.default.fileExists(atPath: url.path) {
            _ = try FileManager.default.replaceItemAt(url, withItemAt: temporary, backupItemName: nil, options: .usingNewMetadataOnly)
        } else {
            try FileManager.default.moveItem(at: temporary, to: url)
        }
    }

    private enum ProviderFailure: Error, CustomStringConvertible {
        case invalidReceipt(String), digestMismatch(String), sourceChanged(String), staleDestination, placeholder(String), materialization(String), resourceError(String, String), resourceState(String), deadline
        var description: String {
            switch self {
            case let .invalidReceipt(value): "invalid_receipt: \(value)"
            case let .digestMismatch(value): "digest_mismatch: \(value)"
            case let .sourceChanged(value): "source_changed: \(value)"
            case .staleDestination: "stale_or_mismatched_destination"
            case let .placeholder(value): "icloud_placeholder: \(value)"
            case let .materialization(value): "materialization_timeout: \(value)"
            case let .resourceError(name, error): "resource_error: \(name): \(error)"
            case let .resourceState(name): "resource_state_unknown_or_non_icloud: \(name)"
            case .deadline: "deadline_exceeded"
            }
        }
    }

    private static func safeName(_ name: String) -> Bool {
        !name.isEmpty && name != "." && name != ".." && !name.contains("/") && !name.contains("\\") && !name.hasPrefix(".")
    }

    private static func boundedData(_ url: URL) throws -> Data {
        let identity = try regularIdentity(url, directory: false)
        guard identity.bytes <= maxReceiptBytes else { throw ProviderFailure.invalidReceipt("receipt exceeds byte cap") }
        return try Data(contentsOf: url, options: [.mappedIfSafe])
    }

    private struct Identity: Sendable { let device: Int64; let inode: Int64; let bytes: Int64 }
    private static func regularIdentity(_ url: URL, directory: Bool) throws -> Identity {
        var info = stat()
        guard lstat(url.path, &info) == 0 else { throw ProviderFailure.sourceChanged(url.lastPathComponent) }
        let mode = info.st_mode & S_IFMT
        guard directory ? mode == S_IFDIR : mode == S_IFREG else { throw ProviderFailure.sourceChanged(url.lastPathComponent) }
        return Identity(device: Int64(info.st_dev), inode: Int64(info.st_ino), bytes: Int64(info.st_size))
    }

    private struct DigestSize: Sendable { let sha256: String; let bytes: Int64 }
    private static func hashAndSize(_ url: URL) throws -> DigestSize {
        let before = try regularIdentity(url, directory: false)
        let handle = try FileHandle(forReadingFrom: url)
        var digest = SHA256(); var size: Int64 = 0
        defer { try? handle.close() }
        while let data = try handle.read(upToCount: hashChunkBytes), !data.isEmpty {
            digest.update(data: data); size += Int64(data.count)
        }
        let after = try regularIdentity(url, directory: false)
        guard before.device == after.device, before.inode == after.inode, before.bytes == after.bytes, size == after.bytes else { throw ProviderFailure.sourceChanged(url.lastPathComponent) }
        return DigestSize(sha256: Data(digest.finalize()).map { String(format: "%02x", $0) }.joined(), bytes: size)
    }

    private static func failureMessage(_ error: Error) -> String { (error as? ProviderFailure)?.description ?? "provider_error: \(error)" }
}
