import Foundation

struct KGRelationPresentation: Equatable {
    struct Expiration: Equatable {
        let date: Date
        let label: String
    }

    let expirationPill: Expiration?

    init(relation: EntityCard.Relation) {
        if let expiredAt = relation.expiredAt {
            expirationPill = Expiration(date: expiredAt, label: "expired")
        } else if let validUntil = relation.validUntil {
            expirationPill = Expiration(date: validUntil, label: "until")
        } else {
            expirationPill = nil
        }
    }

    var isDimmed: Bool {
        expirationPill?.label == "expired"
    }
}
