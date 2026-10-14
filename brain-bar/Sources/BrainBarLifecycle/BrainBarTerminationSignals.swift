import Darwin
import Foundation

/// SIGTERM/SIGINT get one bounded attempt at queue-owned cleanup. SIGKILL,
/// a suspended process or a wedged request queue cannot persist an episode.
public final class BrainBarTerminationSignals: @unchecked Sendable {
    public static let graceSeconds: TimeInterval = 1.5 // Below watchdog TERM→KILL (2s) and daemon ExitTimeOut (5s).
    private let queue = DispatchQueue(label: "com.brainlayer.brainbar.termination", qos: .userInitiated)
    private var sources: [DispatchSourceSignal] = []
    private var stopping = false // Confined to queue; both signal sources share this serial queue.

    public init(shutdown: @escaping @Sendable () -> Void) {
        for number in [SIGTERM, SIGINT] {
            _ = Darwin.signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: queue)
            source.setEventHandler { [weak self] in
                guard let self, !self.stopping else { return }
                self.stopping = true
                DispatchQueue.global(qos: .userInteractive).asyncAfter(deadline: .now() + Self.graceSeconds) {
                    // Never wait on logging/stdio locks on the deadline path.
                    Darwin._exit(0)
                }
                DispatchQueue.global(qos: .userInitiated).async {
                    shutdown()
                    Darwin._exit(0)
                }
            }
            sources.append(source)
            source.resume()
        }
    }
}
