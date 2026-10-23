import Foundation
import XCTest
@testable import BrainBar

final class PendingStoreDirectoryTests: XCTestCase {
    private func fixture() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("brainbar-directory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    func testRootDoesNotCreateOrSyncAnAncestor() throws {
        let root = URL(fileURLWithPath: "/")
        print("QUEUE_ROOT_WALK root=\(root.absoluteString) parent=\(root.deletingLastPathComponent().absoluteString)")
        try BrainDatabase.createDurableQueueDirectory(at: root, sync: { _ in
            XCTFail("The filesystem root is already the durable anchor")
        }, create: { _ in
            XCTFail("The filesystem root must never be created")
        })
    }

    func testDotDotQueuePathTerminatesAndSyncsItsParents() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root.appendingPathComponent("existing"), withIntermediateDirectories: false)
        let queue = root.appendingPathComponent("existing/../nested/queue")
        var synced: [String] = []
        try BrainDatabase.createDurableQueueDirectory(at: queue, sync: { directory in
            XCTAssertFalse(directory.pathComponents.contains(".."))
            synced.append(directory.path)
        })
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("nested/queue").path))
        XCTAssertTrue(synced.contains(root.standardizedFileURL.path))
        XCTAssertTrue(synced.contains(root.appendingPathComponent("nested").standardizedFileURL.path))
    }

    func testConcurrentDirectoryCreatorStillRequiresParentSync() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let queue = root.appendingPathComponent("queue")
        var raced = false
        var synced: [String] = []
        try BrainDatabase.createDurableQueueDirectory(at: queue, sync: { synced.append($0.path) }, create: { directory in
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            raced = true // The peer wins after fileExists, before our mkdir.
            throw CocoaError(.fileWriteFileExists)
        })
        XCTAssertTrue(raced)
        XCTAssertTrue(FileManager.default.fileExists(atPath: queue.path))
        XCTAssertTrue(synced.contains(root.standardizedFileURL.path))
    }

    func testConcurrentFileAndMissingDirectoryRemainErrors() throws {
        for createFile in [false, true] {
            let root = try fixture()
            defer { try? FileManager.default.removeItem(at: root) }
            let queue = root.appendingPathComponent("queue")
            var synced: [String] = []
            XCTAssertThrowsError(try BrainDatabase.createDurableQueueDirectory(
                at: queue, sync: { synced.append($0.path) }, create: { directory in
                    if createFile { try Data("peer file".utf8).write(to: directory) }
                    throw CocoaError(.fileWriteFileExists)
                }
            ))
            XCTAssertFalse(synced.contains(root.standardizedFileURL.path))
        }
    }

    func testConcurrentCreatorDoesNotWaiveFailedBarrierOnRetry() throws {
        let root = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let queue = root.appendingPathComponent("queue")
        var attempts = 0
        let sync: (URL) throws -> Void = { directory in
            if directory.path == root.standardizedFileURL.path {
                attempts += 1
                if attempts == 1 { throw CocoaError(.fileWriteUnknown) }
            }
        }
        XCTAssertThrowsError(try BrainDatabase.createDurableQueueDirectory(at: queue, sync: sync, create: { directory in
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            throw CocoaError(.fileWriteFileExists)
        }))
        try BrainDatabase.createDurableQueueDirectory(at: queue, sync: sync)
        XCTAssertEqual(attempts, 2, "Existing directory visibility cannot discharge the failed parent barrier")
    }
}
