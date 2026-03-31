import Foundation

@MainActor
final class SearchViewModel: ObservableObject {
    enum MoveDirection {
        case up
        case down
    }

    @Published var queryText = ""
    @Published var results: [SearchResult] = []
    @Published private(set) var filters = SearchFilters()

    var selectedResultID: String? {
        guard let selectedResultIndex, results.indices.contains(selectedResultIndex) else {
            return nil
        }
        return results[selectedResultIndex].id
    }

    private let queryActor: SearchQueryActor
    private var selectedResultIndex: Int?

    init(queryActor: SearchQueryActor) {
        self.queryActor = queryActor
    }

    func updateQuery(_ query: String) async {
        queryText = query

        guard let candidates = await queryActor.search(query: query, filters: filters) else {
            return
        }

        let mappedResults = candidates.map { candidate in
            SearchResult(
                chunkID: candidate.id,
                score: candidate.lexicalScore,
                project: candidate.project,
                date: candidate.date,
                summary: candidate.previewText,
                snippet: candidate.previewText,
                importance: candidate.importance > 0 ? candidate.importance : nil
            )
        }
        applyResults(mappedResults)
    }

    func selectFilter(_ chip: SearchFilterChip) async {
        filters = filters.selecting(chip)
        await updateQuery(queryText)
    }

    func selectResult(id: String) {
        guard let index = results.firstIndex(where: { $0.id == id }) else { return }
        selectedResultIndex = index
    }

    func moveSelection(_ direction: MoveDirection) {
        guard !results.isEmpty else {
            selectedResultIndex = nil
            return
        }

        let currentIndex = selectedResultIndex ?? 0
        switch direction {
        case .up:
            selectedResultIndex = max(currentIndex - 1, 0)
        case .down:
            selectedResultIndex = min(currentIndex + 1, results.count - 1)
        }
    }

    func activateSelectedResult() -> SearchResult? {
        guard let selectedResultID else { return nil }
        return results.first(where: { $0.id == selectedResultID })
    }

    private func applyResults(_ newResults: [SearchResult]) {
        results = newResults
        selectedResultIndex = newResults.isEmpty ? nil : 0
    }
}
