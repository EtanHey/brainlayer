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
