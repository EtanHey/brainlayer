import Foundation
@testable import BrainBar

enum RecordingHybridSearchClientError: LocalizedError {
    case injectedFailure

    var errorDescription: String? {
        switch self {
        case .injectedFailure:
            return "injected hybrid search failure"
        }
    }
}

final class RecordingHybridSearchClient: HybridSearchClientProtocol, @unchecked Sendable {
    private let result: Result<HybridSearchResponse, Error>
    private let delay: TimeInterval
    private let lock = NSLock()
    private var recordedRequests: [[String: Any]] = []

    var requests: [[String: Any]] {
        lock.lock()
        defer { lock.unlock() }
        return recordedRequests
    }

    init(response: HybridSearchResponse, delay: TimeInterval = 0) {
        result = .success(response)
        self.delay = delay
    }

    init(error: Error = RecordingHybridSearchClientError.injectedFailure, delay: TimeInterval = 0) {
        result = .failure(error)
        self.delay = delay
    }

    deinit {}

    func search(arguments: [String: Any]) throws -> HybridSearchResponse {
        lock.lock()
        recordedRequests.append(arguments)
        lock.unlock()
        if delay > 0 {
            Thread.sleep(forTimeInterval: delay)
        }
        return try result.get()
    }
}
