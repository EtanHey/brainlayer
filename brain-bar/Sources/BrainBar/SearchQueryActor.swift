import Foundation

struct SearchQueryCandidate: Equatable, Sendable, Identifiable {
    let id: String
    let previewText: String
    let lexicalScore: Double
    let date: String
    let project: String
    let importance: Int

    init(id: String, previewText: String, lexicalScore: Double, date: String = "", project: String = "", importance: Int = 0) {
        self.id = id
        self.previewText = previewText
        self.lexicalScore = lexicalScore
        self.date = date
        self.project = project
        self.importance = importance
    }
}

actor SearchQueryActor {
    typealias LexicalSearch = @Sendable (_ query: String, _ candidateLimit: Int, _ filters: SearchFilters) async -> [SearchQueryCandidate]
    typealias Rerank = @Sendable (_ candidates: [SearchQueryCandidate], _ query: String) async -> [SearchQueryCandidate]

    private let lexicalSearch: LexicalSearch
    private let rerank: Rerank
    private var latestGeneration = 0

    init(
        lexicalSearch: @escaping LexicalSearch,
        rerank: @escaping Rerank
    ) {
        self.lexicalSearch = lexicalSearch
        self.rerank = rerank
    }

    func search(
        query: String,
        filters: SearchFilters = SearchFilters(),
        candidateLimit: Int = 100
    ) async -> [SearchQueryCandidate]? {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        latestGeneration += 1
        let generation = latestGeneration

        let lexicalCandidates = await lexicalSearch(trimmed, candidateLimit, filters)
        guard generation == latestGeneration else { return nil }

        let boundedCandidates = Array(lexicalCandidates.prefix(candidateLimit))
        let rerankedCandidates = await rerank(boundedCandidates, trimmed)
        guard generation == latestGeneration else { return nil }

        return rerankedCandidates
    }
}
