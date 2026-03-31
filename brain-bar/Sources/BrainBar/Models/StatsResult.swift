import Foundation

struct StatsResult: Equatable {
    let totalChunks: Int
    let projects: [String]
    let contentTypes: [String]

    init(totalChunks: Int, projects: [String], contentTypes: [String]) {
        self.totalChunks = totalChunks
        self.projects = projects
        self.contentTypes = contentTypes
    }

    init(payload: [String: Any]) {
        totalChunks = payload["total_chunks"] as? Int ?? 0
        projects = payload["projects"] as? [String] ?? []
        contentTypes = payload["content_types"] as? [String] ?? []
    }
}
