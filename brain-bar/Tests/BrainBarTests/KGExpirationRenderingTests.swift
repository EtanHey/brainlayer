import XCTest
@testable import BrainBar

final class KGExpirationRenderingTests: XCTestCase {
    func testEntityCardRelationDecodesExpirationDatesFromLookupPayload() throws {
        let payload: [String: Any] = [
            "entity_id": "person-etan",
            "name": "Etan",
            "entity_type": "person",
            "relations": [
                [
                    "relation_type": "cto_of",
                    "target_name": "Domica",
                    "direction": "outgoing",
                    "valid_until": "2026-05-24T00:00:00Z",
                    "expired_at": "2026-05-24T00:00:00Z",
                ]
            ],
        ]

        let card = EntityCard(lookupPayload: payload)
        let relation = try XCTUnwrap(card.relations.first)

        XCTAssertEqual(relation.targetName, "Domica")
        XCTAssertNotNil(relation.validUntil)
        XCTAssertNotNil(relation.expiredAt)
    }

    func testExpirationPillFormatsDateForInlineRendering() throws {
        let date = try XCTUnwrap(KGTemporalDate.parse("2026-05-24T00:00:00Z"))

        XCTAssertEqual(ExpirationPill.formattedDate(date), "May 24, 2026")
        XCTAssertEqual(ExpirationPill.displayText(date: date, label: "expired"), "expired May 24, 2026")
    }

    func testRelationPresentationDimsExpiredAndPillsButKeepsLiveRelationsVisible() throws {
        let expiredAt = try XCTUnwrap(KGTemporalDate.parse("2026-05-24T00:00:00Z"))
        let validUntil = try XCTUnwrap(KGTemporalDate.parse("2026-06-01T00:00:00Z"))
        let now = try XCTUnwrap(KGTemporalDate.parse("2026-05-25T00:00:00Z"))
        let expired = EntityCard.Relation(
            relationType: "cto_of",
            targetName: "Domica",
            expiredAt: expiredAt
        )
        let live = EntityCard.Relation(
            relationType: "builds",
            targetName: "BrainLayer"
        )
        let forwardDated = EntityCard.Relation(
            relationType: "collaborates_with",
            targetName: "Cursor",
            validUntil: validUntil
        )

        XCTAssertTrue(KGRelationPresentation(relation: expired, now: now).isDimmed)
        XCTAssertEqual(KGRelationPresentation(relation: expired, now: now).expirationPill?.label, "expired")
        XCTAssertFalse(KGRelationPresentation(relation: live, now: now).isDimmed)
        XCTAssertNil(KGRelationPresentation(relation: live, now: now).expirationPill)
        XCTAssertFalse(KGRelationPresentation(relation: forwardDated, now: now).isDimmed)
        XCTAssertEqual(KGRelationPresentation(relation: forwardDated, now: now).expirationPill?.label, "until")
    }

    func testRelationPresentationDimsPastValidUntilAsExpiredWhenExpiredAtIsMissing() throws {
        let validUntil = try XCTUnwrap(KGTemporalDate.parse("2026-05-24T00:00:00Z"))
        let now = try XCTUnwrap(KGTemporalDate.parse("2026-05-25T00:00:00Z"))
        let relation = EntityCard.Relation(
            relationType: "cto_of",
            targetName: "Domica",
            validUntil: validUntil
        )

        let presentation = KGRelationPresentation(relation: relation, now: now)

        XCTAssertTrue(presentation.isDimmed)
        XCTAssertEqual(presentation.expirationPill?.label, "expired")
        XCTAssertEqual(presentation.expirationPill?.date, validUntil)
    }
}
