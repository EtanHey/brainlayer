import Foundation

struct KGSearchResult: Equatable {
    struct Fact: Equatable {
        let source: String
        let relation: String
        let target: String
    }

    struct MemoryResult: Equatable {
        let chunkID: String
        let score: Double
        let snippet: String
    }

    let entityName: String
    let query: String
    let facts: [Fact]
    let results: [MemoryResult]
}
