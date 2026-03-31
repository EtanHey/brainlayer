import XCTest
@testable import BrainBar

@MainActor
final class SearchViewModelTests: XCTestCase {
    func testEmptyQueryClearsResultsWithoutSearchWork() async {
        let actor = SearchQueryActor(
            lexicalSearch: { _, _, _ in
                XCTFail("lexical search should not run for an empty query")
                return []
            },
            rerank: { candidates, _ in candidates }
        )
        let model = SearchViewModel(queryActor: actor)
        model.results = [
            SearchResult(chunkID: "old", score: 1, project: "brainlayer", date: "", summary: "old", snippet: "old")
        ]

        await model.updateQuery("   ")

        XCTAssertEqual(model.results, [])
        XCTAssertNil(model.selectedResultID)
    }

    func testUpdatingQueryPublishesMappedResultsAndSelectsFirstRow() async {
        let actor = SearchQueryActor(
            lexicalSearch: { query, _, _ in
                XCTAssertEqual(query, "brainbar")
                return [
                    SearchQueryCandidate(id: "c1", previewText: "BrainBar search window", lexicalScore: 3),
                    SearchQueryCandidate(id: "c2", previewText: "BrainBar quick capture", lexicalScore: 2),
                ]
            },
            rerank: { candidates, _ in candidates }
        )
        let model = SearchViewModel(queryActor: actor)

        await model.updateQuery("brainbar")

        XCTAssertEqual(model.results.map(\.id), ["c1", "c2"])
        XCTAssertEqual(model.selectedResultID, "c1")
    }

    func testChangingPrimaryFilterRequeries() async {
        let queries = QueryLog()
        let filters = FilterLog()
        let actor = SearchQueryActor(
            lexicalSearch: { query, _, searchFilters in
                await queries.append(query)
                await filters.append(searchFilters.primaryChip)
                return [SearchQueryCandidate(id: "c1", previewText: "Result", lexicalScore: 1)]
            },
            rerank: { candidates, _ in candidates }
        )
        let model = SearchViewModel(queryActor: actor)

        await model.updateQuery("vector")
        await model.selectFilter(.important)

        let recordedQueries = await queries.values()
        let recordedFilters = await filters.values()
        XCTAssertEqual(recordedQueries, ["vector", "vector"])
        XCTAssertEqual(recordedFilters, [.all, .important])
        XCTAssertEqual(model.filters.primaryChip, .important)
    }

    func testMoveSelectionNavigatesVisibleResults() async {
        let actor = SearchQueryActor(
            lexicalSearch: { _, _, _ in
                [
                    SearchQueryCandidate(id: "c1", previewText: "One", lexicalScore: 3),
                    SearchQueryCandidate(id: "c2", previewText: "Two", lexicalScore: 2),
                    SearchQueryCandidate(id: "c3", previewText: "Three", lexicalScore: 1),
                ]
            },
            rerank: { candidates, _ in candidates }
        )
        let model = SearchViewModel(queryActor: actor)
        await model.updateQuery("anything")

        model.moveSelection(.down)
        XCTAssertEqual(model.selectedResultID, "c2")

        model.moveSelection(.down)
        XCTAssertEqual(model.selectedResultID, "c3")

        model.moveSelection(.up)
        XCTAssertEqual(model.selectedResultID, "c2")
    }

    func testActivateSelectedResultPublishesActivatedSearchResult() async {
        let actor = SearchQueryActor(
            lexicalSearch: { _, _, _ in
                [SearchQueryCandidate(id: "c1", previewText: "Chosen result", lexicalScore: 1)]
            },
            rerank: { candidates, _ in candidates }
        )
        let model = SearchViewModel(queryActor: actor)
        await model.updateQuery("chosen")

        let activated = model.activateSelectedResult()

        XCTAssertEqual(activated?.id, "c1")
        XCTAssertEqual(activated?.displayText, "Chosen result")
    }
}

private actor QueryLog {
    private var entries: [String] = []

    func append(_ value: String) {
        entries.append(value)
    }

    func values() -> [String] {
        entries
    }
}

private actor FilterLog {
    private var entries: [SearchFilterChip] = []

    func append(_ value: SearchFilterChip) {
        entries.append(value)
    }

    func values() -> [SearchFilterChip] {
        entries
    }
}
