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
    private(set) var requests: [[String: Any]] = []

    init(response: HybridSearchResponse) {
        result = .success(response)
    }

    init(error: Error = RecordingHybridSearchClientError.injectedFailure) {
        result = .failure(error)
    }

    deinit {}

    func search(arguments: [String: Any]) throws -> HybridSearchResponse {
        requests.append(arguments)
        return try result.get()
    }
}
