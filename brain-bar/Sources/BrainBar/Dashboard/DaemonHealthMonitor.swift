import Darwin
import Foundation
import MachO

final class DaemonHealthMonitor: @unchecked Sendable {
    private let targetPID: pid_t
    private let launchTime = ProcessInfo.processInfo.systemUptime

    init(targetPID: pid_t) {
        self.targetPID = targetPID
    }

    func sample() -> DaemonHealthSnapshot? {
        guard targetPID > 0 else { return nil }
        guard kill(targetPID, 0) == 0 else { return nil }

        let isCurrentProcess = targetPID == ProcessInfo.processInfo.processIdentifier
        let rssBytes = isCurrentProcess ? currentResidentSize() : 0
        let uptime = isCurrentProcess ? (ProcessInfo.processInfo.systemUptime - launchTime) : 0
        let openConnections = isCurrentProcess ? countOpenSocketDescriptors() : 0

        return DaemonHealthSnapshot(
            pid: targetPID,
            isResponsive: true,
            rssBytes: rssBytes,
            uptime: uptime,
            openConnections: openConnections,
            lastSeenAt: Date()
        )
    }

    private func currentResidentSize() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)

        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
            }
        }

        guard result == KERN_SUCCESS else { return 0 }
        return UInt64(info.resident_size)
    }

    private func countOpenSocketDescriptors() -> Int {
        let maxDescriptors = Int(getdtablesize())
        guard maxDescriptors > 0 else { return 0 }

        var socketCount = 0
        for fd in 0..<maxDescriptors {
            if fcntl(Int32(fd), F_GETFD) == -1 {
                continue
            }

            var socketType: Int32 = 0
            var length = socklen_t(MemoryLayout<Int32>.size)
            let result = withUnsafeMutablePointer(to: &socketType) { pointer in
                getsockopt(Int32(fd), SOL_SOCKET, SO_TYPE, pointer, &length)
            }

            if result == 0 {
                socketCount += 1
            }
        }

        return socketCount
    }
}
