import Foundation

enum BrainBarLivePulse {
    static func shouldPulse(previous: [Int], current: [Int]) -> Bool {
        previous != current
    }
}
