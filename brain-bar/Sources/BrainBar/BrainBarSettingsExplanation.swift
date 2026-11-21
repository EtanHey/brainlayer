import Foundation

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
        guard count > 0 else { return nil }
        let path = (folder.path as NSString).abbreviatingWithTildeInPath
        return BrainBarMenuPresenter.SurfaceSay.settingsBackups(count, path: path)
    }
}
