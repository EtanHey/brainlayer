import Darwin
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
        guard let resolvedDB = resolvedPath(dbPath), let resolvedSocket = resolvedPath(socketPath),
              let dbParent = privateParent(of: resolvedDB),
              let socketParent = privateParent(of: resolvedSocket),
              dbParent == socketParent, dbParent != "/tmp",
              isPrivateDirectory(dbParent),
              resolvedDB != resolvedSocket else {
            return .failure("--isolated requires routes beneath the same owned 0700 private directory")
        }
        let canonicalDB = resolvedPath(NSHomeDirectory() + "/.local/share/brainlayer/brainlayer.db")
        let canonicalSocket = resolvedPath("/tmp/brainbar.sock")
        guard resolvedDB != canonicalDB, resolvedSocket != canonicalSocket,
              !sameExistingFile(dbPath, socketPath) else {
            return .failure("--isolated routes resolve to a canonical or aliased path")
        }
        return .success(.isolated(dbPath: dbPath, socketPath: socketPath))
    }

    private static func resolvedPath(_ path: String) -> String? {
        guard path.hasPrefix("/") else { return nil }
        return URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath().path
    }

    private static func privateParent(of path: String) -> String? {
        URL(fileURLWithPath: path).deletingLastPathComponent().path
    }

    private static func isPrivateDirectory(_ path: String) -> Bool {
        var info = stat()
        guard lstat(path, &info) == 0,
              (info.st_mode & S_IFMT) == S_IFDIR,
              (info.st_mode & 0o777) == 0o700 else { return false }
        return info.st_uid == getuid()
    }

    private static func sameExistingFile(_ left: String, _ right: String) -> Bool {
        var leftInfo = stat()
        var rightInfo = stat()
        guard lstat(left, &leftInfo) == 0, lstat(right, &rightInfo) == 0 else { return false }
        return leftInfo.st_dev == rightInfo.st_dev && leftInfo.st_ino == rightInfo.st_ino
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
