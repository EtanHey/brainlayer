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
        var fdInfos = Array(repeating: proc_fdinfo(), count: 256)
        let bytesRead = fdInfos.withUnsafeMutableBytes { rawBuffer in
            proc_pidinfo(
                targetPID,
                PROC_PIDLISTFDS,
                0,
                rawBuffer.baseAddress,
                Int32(rawBuffer.count)
            )
        }

        guard bytesRead > 0 else { return 0 }
        let infoCount = Int(bytesRead) / MemoryLayout<proc_fdinfo>.stride
        return fdInfos.prefix(infoCount).reduce(into: 0) { count, info in
            if Int32(info.proc_fdtype) == PROX_FDTYPE_SOCKET {
                count += 1
            }
        }
    }
}
