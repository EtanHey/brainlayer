import Darwin
import Foundation

public enum BrainBarSignalSafety {
    public static func ignoreSIGPIPE() {
        _ = Darwin.signal(SIGPIPE, SIG_IGN)
    }

    @discardableResult
    public static func write(_ data: Data, to handle: FileHandle, context: String) -> Bool {
        do {
            try handle.write(contentsOf: data)
            return true
        } catch {
            NSLog("[BrainBar] %@ write failed (%@)", context, String(reflecting: type(of: error)))
            return false
        }
    }
}
