import Darwin
import Foundation

final class DaemonHealthMonitor: @unchecked Sendable {
    private let targetPID: pid_t

    init(targetPID: pid_t) {
        self.targetPID = targetPID
    }

    func sample() -> DaemonHealthSnapshot? {
        guard targetPID > 0 else { return nil }
        guard kill(targetPID, 0) == 0 else { return nil }

        guard let taskInfo = taskInfo(),
              let bsdInfo = bsdInfo() else {
            return DaemonHealthSnapshot(
                pid: targetPID,
                isResponsive: false,
                rssBytes: 0,
                uptime: 0,
                openConnections: 0,
                lastSeenAt: Date()
            )
        }

        let startTime = Date(
            timeIntervalSince1970: TimeInterval(bsdInfo.pbi_start_tvsec) +
                (TimeInterval(bsdInfo.pbi_start_tvusec) / 1_000_000)
        )

        return DaemonHealthSnapshot(
            pid: targetPID,
            isResponsive: true,
            rssBytes: taskInfo.pti_resident_size,
            uptime: max(0, Date().timeIntervalSince(startTime)),
            openConnections: countOpenSocketDescriptors(),
            lastSeenAt: Date()
        )
    }

    private func taskInfo() -> proc_taskinfo? {
        var info = proc_taskinfo()
        let size = Int32(MemoryLayout.size(ofValue: info))
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            proc_pidinfo(targetPID, PROC_PIDTASKINFO, 0, pointer, size)
        }
        guard result == size else { return nil }
        return info
    }

    private func bsdInfo() -> proc_bsdinfo? {
        var info = proc_bsdinfo()
        let size = Int32(MemoryLayout.size(ofValue: info))
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            proc_pidinfo(targetPID, PROC_PIDTBSDINFO, 0, pointer, size)
        }
        guard result == size else { return nil }
        return info
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
