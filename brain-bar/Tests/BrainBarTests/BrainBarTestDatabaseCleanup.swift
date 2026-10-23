import Foundation
import XCTest
@testable import BrainBar

typealias BrainBarFixtureCleanupResult = FixtureShutdownResult

@discardableResult
func finishBrainBarSocketFixture(
    _ server: BrainBarServer,
    database: BrainDatabase,
    path: String,
    deadline: DispatchTime = .now() + 5
) -> BrainBarFixtureCleanupResult {
    guard server.stopAndDrainForTesting(deadline: deadline) == .finished else { return .timedOut }
    database.close()
    removeBrainBarTestDatabase(path)
    return .finished
}

/// Darwin Foundation can ignore TMPDIR; use the runner's explicit suite-owned root.
func brainBarTestTemporaryDirectory() -> URL {
    if let path = ProcessInfo.processInfo.environment["BRAINBAR_TEST_TMP_ROOT"] {
        return URL(fileURLWithPath: path, isDirectory: true)
    }
    return FileManager.default.temporaryDirectory
}

func brainBarTestTemporaryPath() -> String { brainBarTestTemporaryDirectory().path + "/" }

/// Remove only this fixture's database and its two SQLite sidecars, after closing its connections.
func removeBrainBarTestDatabase(_ path: String, file: StaticString = #filePath, line: UInt = #line) {
    for suffix in ["", "-wal", "-shm"] {
        let ownedPath = path + suffix
        guard FileManager.default.fileExists(atPath: ownedPath) else { continue }
        do { try FileManager.default.removeItem(atPath: ownedPath) }
        catch { XCTFail("Unable to remove owned fixture \(ownedPath): \(error)", file: file, line: line) }
    }
}
