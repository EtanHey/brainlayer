import Foundation

@MainActor
final class KGSimulationController {
    typealias TickHandler = @MainActor () -> CGFloat
    typealias SleepHandler = @Sendable (Duration) async -> Void

    static let defaultFrameDuration: Duration = .milliseconds(33)
    static let defaultEnergyThreshold: CGFloat = 0.01

    private(set) var timerActive = false

    private let frameDuration: Duration
    private let energyThreshold: CGFloat
    private let sleep: SleepHandler
    private var simulationTask: Task<Void, Never>?
    private var isActive = true

    init(
        frameDuration: Duration = KGSimulationController.defaultFrameDuration,
        energyThreshold: CGFloat = KGSimulationController.defaultEnergyThreshold,
        sleep: @escaping SleepHandler = { duration in
            try? await Task.sleep(for: duration)
        }
    ) {
        self.frameDuration = frameDuration
        self.energyThreshold = energyThreshold
        self.sleep = sleep
    }

    func start(tick: @escaping TickHandler) {
        guard isActive else { return }
        guard !timerActive else { return }

        timerActive = true
        simulationTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.simulationTask = nil }

            while self.timerActive && !Task.isCancelled {
                await self.sleep(self.frameDuration)
                guard self.timerActive && !Task.isCancelled else { break }

                if tick() < self.energyThreshold {
                    self.timerActive = false
                }
            }
        }
    }

    func stop() {
        timerActive = false
        simulationTask?.cancel()
        simulationTask = nil
    }

    func setActive(_ active: Bool) {
        isActive = active
        if !active {
            stop()
        }
    }

    deinit {
        simulationTask?.cancel()
    }
}
