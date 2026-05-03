import Darwin
import Foundation
import os

final class BrainBarInstanceLock: @unchecked Sendable {
    enum AcquireError: Error, Equatable {
        case alreadyRunning
        case openFailed(String)
        case lockFailed(String)
    }

    private let fd: Int32
    private let lockPath: String
    private let releaseLock = OSAllocatedUnfairLock(initialState: false)

    private init(fd: Int32, lockPath: String) {
        self.fd = fd
        self.lockPath = lockPath
    }

    deinit {
        release()
    }

    static func acquire(lockPath: String) throws -> BrainBarInstanceLock {
        let directory = URL(fileURLWithPath: lockPath).deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let fd = open(lockPath, O_RDWR | O_CREAT, 0o600)
        guard fd >= 0 else {
            throw AcquireError.openFailed(String(cString: strerror(errno)))
        }

        guard flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            let lockErrno = errno
            let message = String(cString: strerror(lockErrno))
            close(fd)
            if lockErrno == EWOULDBLOCK {
                throw AcquireError.alreadyRunning
            }
            throw AcquireError.lockFailed(message)
        }

        var pidLine = "\(getpid())\n".data(using: .utf8) ?? Data()
        ftruncate(fd, 0)
        lseek(fd, 0, SEEK_SET)
        _ = pidLine.withUnsafeMutableBytes { ptr in
            write(fd, ptr.baseAddress, ptr.count)
        }
        fsync(fd)

        return BrainBarInstanceLock(fd: fd, lockPath: lockPath)
    }

    func release() {
        let shouldRelease = releaseLock.withLock { released -> Bool in
            guard !released else { return false }
            released = true
            return true
        }
        guard shouldRelease else { return }
        flock(fd, LOCK_UN)
        close(fd)
        try? FileManager.default.removeItem(atPath: lockPath)
    }
}
