import Foundation

enum BrainBarDaemonStartupMode: Equatable {
    enum ParseOutcome {
        case success(BrainBarDaemonStartupMode)
        case failure(String)
    }

    case normal
    case isolated(dbPath: String, socketPath: String)

    static func parse(arguments: [String], environment: [String: String]) -> ParseOutcome {
        let flags = Array(arguments.dropFirst())
        guard flags == ["--isolated"] || flags.isEmpty else {
            return .failure("unsupported arguments; use --isolated")
        }
        guard flags == ["--isolated"] else { return .success(.normal) }
        guard let dbPath = environment["BRAINBAR_DB_PATH"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              let socketPath = environment["BRAINBAR_SOCKET_PATH"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !dbPath.isEmpty, !socketPath.isEmpty else {
            return .failure("--isolated requires BRAINBAR_DB_PATH and BRAINBAR_SOCKET_PATH")
        }
        let canonicalDB = NSHomeDirectory() + "/.local/share/brainlayer/brainlayer.db"
        guard dbPath != canonicalDB, socketPath != "/tmp/brainbar.sock",
              dbPath.hasPrefix("/"), socketPath.hasPrefix("/") else {
            return .failure("--isolated requires absolute noncanonical DB and socket paths")
        }
        return .success(.isolated(dbPath: dbPath, socketPath: socketPath))
    }

    var dbPath: String? {
        if case let .isolated(path, _) = self { return path }
        return nil
    }

    var socketPath: String? {
        if case let .isolated(_, path) = self { return path }
        return nil
    }
}
