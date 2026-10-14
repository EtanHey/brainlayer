import Foundation

/// One clock for socket age and receipt provenance; release defaults stay real.
public struct MCPTransportClock {
    public let now: () -> Date
    public let uptime: () -> TimeInterval
    public let publishInterval: TimeInterval
    public init(now: @escaping () -> Date = Date.init,
                uptime: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime },
                publishInterval: TimeInterval = 5) {
        self.now = now; self.uptime = uptime; self.publishInterval = publishInterval
    }

    #if DEBUG
    /// Private debug daemon fixture only. Neither symbol nor key ships in release.
    public static func processFixture() throws -> Self {
        guard let path = ProcessInfo.processInfo.environment["BRAINBAR_RECOVERY_TEST_CLOCK"] else { return Self() }
        func offset() throws -> Double {
            // Atomic fixture replacement can unlink an already-open old FD.
            for _ in 0..<3 {
                if let data = try? TransportReceiptFile.read(path),
                   let value = try? JSONDecoder().decode(Double.self, from: data),
                   value.isFinite, (0...120).contains(value) { return value }
            }
            throw CocoaError(.fileReadNoPermission)
        }
        _ = try offset() // Refuse unsafe fixture evidence before server startup.
        return Self(now: { Date().addingTimeInterval(try! offset()) },
                    uptime: { ProcessInfo.processInfo.systemUptime + (try! offset()) }, publishInterval: 0.02)
    }
    #endif
}
