// CacheDelete's private request API is used by Apple's DeviceLink host when a
// backup needs space. Keep it in a subprocess: an absent or changed ABI must
// fail the Python preflight, never the backup process holding live DB state.
import Darwin
import Dispatch
import Foundation

typealias PurgeReply = @convention(block) (NSDictionary?) -> Void
typealias PurgeCall = @convention(c) (UnsafeRawPointer, PurgeReply) -> Void

func fail(_ message: String) -> Never {
    fputs("CacheDelete: \(message)\n", stderr)
    exit(1)
}

let framework = "/System/Library/PrivateFrameworks/CacheDelete.framework/CacheDelete"
guard let library = dlopen(framework, RTLD_NOW | RTLD_LOCAL) else {
    fail("framework unavailable")
}
defer { dlclose(library) }
guard let symbol = dlsym(library, "CacheDeletePurgeSpaceWithInfo") else {
    fail("purge request symbol unavailable")
}
if CommandLine.arguments.dropFirst().first == "--probe" {
    print("CacheDelete purge request symbol available")
    exit(0)
}
guard CommandLine.arguments.count == 3,
      let amount = Int64(CommandLine.arguments[2]), amount > 0 else {
    fail("expected volume path and positive byte count")
}

let info: NSDictionary = [
    "CACHE_DELETE_VOLUME": CommandLine.arguments[1],
    "CACHE_DELETE_AMOUNT": NSNumber(value: amount),
    "CACHE_DELETE_URGENCY_LIMIT": NSNumber(value: 3),
]
let request = unsafeBitCast(symbol, to: PurgeCall.self)
let completed = DispatchSemaphore(value: 0)
var reply: NSDictionary?
let callback: PurgeReply = { result in
    reply = result
    completed.signal()
}
request(UnsafeRawPointer(Unmanaged.passUnretained(info).toOpaque()), callback)
guard completed.wait(timeout: .now() + .seconds(150)) == .success else {
    fail("request timed out")
}
guard let reply else {
    fail("request returned no result")
}
if let error = reply["CACHE_DELETE_ERROR"] {
    fail("request returned error: \(error)")
}
print("CacheDelete request completed; raw free space must be remeasured")
