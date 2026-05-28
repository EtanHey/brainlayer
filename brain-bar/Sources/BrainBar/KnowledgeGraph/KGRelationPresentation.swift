import Foundation

struct KGRelationPresentation: Equatable {
    struct Expiration: Equatable {
        let date: Date
        let label: String
    }

    let expirationPill: Expiration?

    init(relation: EntityCard.Relation, now: Date = Date()) {
        if let expiredAt = relation.expiredAt {
            expirationPill = Expiration(date: expiredAt, label: "expired")
        } else if let validUntil = relation.validUntil, validUntil < now {
            expirationPill = Expiration(date: validUntil, label: "expired")
        } else if let validUntil = relation.validUntil {
            expirationPill = Expiration(date: validUntil, label: "until")
        } else {
            expirationPill = nil
        }
    }

    var isDimmed: Bool {
        expirationPill?.label == "expired"
    }

    /// Expired relations are de-emphasized, not badged (QA #18/#19). The
    /// inline pill is therefore reserved for forward-dated ("until") relations.
    var inlinePill: Expiration? {
        guard let expirationPill, expirationPill.label != "expired" else { return nil }
        return expirationPill
    }

    /// De-emphasized footer line shown at the bottom of an expired relation row
    /// instead of a badge. `nil` for live / forward-dated relations.
    var expiredFooterText: String? {
        guard let expirationPill, expirationPill.label == "expired" else { return nil }
        return "Expired \(ExpirationPill.formattedDate(expirationPill.date))"
    }
}
