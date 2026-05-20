import Foundation

struct KGEdge: Identifiable, Equatable, Sendable {
    let sourceId: String
    let targetId: String
    let relationType: String

    var id: String { "\(sourceId)-\(relationType)-\(targetId)" }
}
