import Foundation

/// Connection-derived identity persisted for the offline scheduled backup.
/// The open-generation token is provenance only; a daemon never compares it
/// with a process-local connection token after restart.
struct BrainLayerScheduledOfflineSourceBinding: Codable, Equatable, Sendable {
    /// Increment on incompatible serialization changes; schedulers refuse an
    /// unknown version rather than guessing a database pathname.
    let version: Int
    let path: String
    let stDev: Int64
    let stIno: Int64
    let sourceOpenGeneration: String

    init(path: String, stDev: Int64, stIno: Int64, sourceOpenGeneration: String) {
        version = 1
        self.path = path
        self.stDev = stDev
        self.stIno = stIno
        self.sourceOpenGeneration = sourceOpenGeneration
    }

    var environmentValue: [String: Any] {
        [
            "version": version,
            "path": path,
            "st_dev": stDev,
            "st_ino": stIno,
            "source_open_generation": sourceOpenGeneration,
        ]
    }
}
