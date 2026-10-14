import Darwin
import Foundation

/// Descriptor-based trust boundary. The path is never reopened for parsing.
enum TransportReceiptFile {
    enum Failure: Error { case missing, refused, oversized, unreadable }
    static func read(_ path: String, ownerUID: uid_t = getuid()) throws -> Data {
        let fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { throw errno == ENOENT ? Failure.missing : Failure.refused }
        defer { close(fd) }
        func validate() throws {
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
                  info.st_uid == ownerUID, info.st_mode & 0o7777 == 0o600,
                  info.st_nlink == 1 else { throw Failure.refused }
            guard info.st_size <= 65_536 else { throw Failure.oversized }
        }
        try validate()
        var result = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let n = Darwin.read(fd, &buffer, buffer.count)
            if n < 0, errno == EINTR { continue }
            guard n >= 0 else { throw Failure.unreadable }
            if n == 0 { break }
            result.append(contentsOf: buffer.prefix(n))
            guard result.count <= 65_536 else { throw Failure.oversized }
        }
        try validate()
        return result
    }
}
