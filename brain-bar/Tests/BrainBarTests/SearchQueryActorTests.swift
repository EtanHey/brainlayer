import XCTest
@testable import BrainBar

private actor IntBox {
    private var value = 0

    func set(_ newValue: Int) {
        value = newValue
    }

    func increment() {
        value += 1
    }

    func get() -> Int {
        value
    }
}

final class SearchQueryActorTests: XCTestCase {
    func testLatestQueryWinsAndOlderResultIsDropped() async {
        let actor = SearchQueryActor(
            lexicalSearch: { query, _, _ in
                if query == "alpha" {
                    try? await Task.sleep(for: .milliseconds(80))
                    return [SearchQueryCandidate(id: "old", previewText: "alpha result", lexicalScore: 1)]
                }
                return [SearchQueryCandidate(id: "new", previewText: "beta result", lexicalScore: 2)]
            },
            rerank: { candidates, _ in candidates }
        )

        async let first = actor.search(query: "alpha")
        try? await Task.sleep(for: .milliseconds(10))
        let second = await actor.search(query: "beta")
        let firstResult = await first

        XCTAssertNil(firstResult, "Superseded query results should be dropped")
        XCTAssertEqual(second?.map(\.id), ["new"])
    }

    func testRerankReceivesOnlyBoundedCandidateSet() async {
        let rerankedCount = IntBox()
        let actor = SearchQueryActor(
            lexicalSearch: { _, _, _ in
                (0..<140).map { index in
                    SearchQueryCandidate(
                        id: "cand-\(index)",
                        previewText: "candidate \(index)",
                        lexicalScore: Double(140 - index)
                    )
                }
            },
            rerank: { candidates, _ in
                await rerankedCount.set(candidates.count)
                return candidates
            }
        )

        let result = await actor.search(query: "bounded", candidateLimit: 75)
        let rerankedTotal = await rerankedCount.get()

        XCTAssertEqual(rerankedTotal, 75)
        XCTAssertEqual(result?.count, 75)
    }

    func testEmptyQuerySkipsLexicalAndRerankWork() async {
        let lexicalCalls = IntBox()
        let rerankCalls = IntBox()
        let actor = SearchQueryActor(
            lexicalSearch: { _, _, _ in
                await lexicalCalls.increment()
                return [SearchQueryCandidate(id: "x", previewText: "x", lexicalScore: 1)]
            },
            rerank: { candidates, _ in
                await rerankCalls.increment()
                return candidates
            }
        )

        let result = await actor.search(query: "   ")
        let lexicalTotal = await lexicalCalls.get()
        let rerankTotal = await rerankCalls.get()

        XCTAssertEqual(result, [])
        XCTAssertEqual(lexicalTotal, 0)
        XCTAssertEqual(rerankTotal, 0)
    }

    func testLexicalSearchReceivesSelectedFilters() async {
        let capturedFilters = FilterBox()
        let actor = SearchQueryActor(
            lexicalSearch: { _, _, filters in
                await capturedFilters.set(filters)
                return [SearchQueryCandidate(id: "f1", previewText: "important result", lexicalScore: 1)]
            },
            rerank: { candidates, _ in candidates }
        )

        _ = await actor.search(
            query: "important",
            filters: SearchFilters(primaryChip: .important)
        )

        let filters = await capturedFilters.get()
        XCTAssertEqual(filters?.primaryChip, .important)
        XCTAssertEqual(filters?.importanceMin, 7)
    }
}

private actor FilterBox {
    private var filters: SearchFilters?

    func set(_ newValue: SearchFilters) {
        filters = newValue
    }

    func get() -> SearchFilters? {
        filters
    }
}
