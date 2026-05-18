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
    private let lock = NSLock()
    private var recordedRequests: [[String: Any]] = []

    var requests: [[String: Any]] {
        lock.lock()
        defer { lock.unlock() }
        return recordedRequests
    }

    init(response: HybridSearchResponse) {
        result = .success(response)
    }

    init(error: Error = RecordingHybridSearchClientError.injectedFailure) {
        result = .failure(error)
    }

    deinit {}

    func search(arguments: [String: Any]) throws -> HybridSearchResponse {
        lock.lock()
        recordedRequests.append(arguments)
        lock.unlock()
        return try result.get()
    }
}
