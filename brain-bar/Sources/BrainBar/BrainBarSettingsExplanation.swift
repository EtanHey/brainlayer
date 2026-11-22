import Foundation

/// A file's measured backup count, or an explicitly invented capture fixture. Never inferred from a missing read.
struct BrainBarSettingsBackupSummary: Sendable {
    let file: URL
    let count: Int
    var line: String {
        BrainBarMenuPresenter.SurfaceSay.settingsBackups(count,
            path: (file.deletingLastPathComponent().path as NSString).abbreviatingWithTildeInPath)
    }
}

/// Only filenames are read. Revealing settings never tidies or changes the owner's files.
enum BrainBarSettingsExplanation {
    static func model(_ name: String?, namesModel: Bool = true) -> [String] {
        guard let name = name?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty else { return [] }
        return BrainBarMenuPresenter.SurfaceSay.modelExplanation(name, namesModel: namesModel)
    }

    static func backupLine(for file: URL, fileManager: FileManager = .default) -> String? {
        let folder = file.deletingLastPathComponent()
        guard let entries = try? fileManager.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.isRegularFileKey]) else {
            return BrainBarMenuPresenter.SurfaceSay.settingsBackupsUnreadable
        }
        let count = entries.filter { entry in
            let suffix = entry.lastPathComponent.dropFirst(file.lastPathComponent.count)
            return entry.lastPathComponent.hasPrefix(file.lastPathComponent + ".")
                && (suffix.hasPrefix(".bak") || suffix.hasPrefix(".backup") || suffix.hasSuffix(".bak"))
                && (try? entry.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }.count
        return BrainBarSettingsBackupSummary(file: file, count: count).line
    }
}
