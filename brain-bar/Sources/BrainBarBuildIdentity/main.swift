import CryptoKit
import Foundation

enum Failure: Error { case gitFailed, invalidRoot }
let package = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
let output = URL(fileURLWithPath: CommandLine.arguments[2])
let env = ProcessInfo.processInfo.environment.filter { !$0.key.hasPrefix("GIT_") }
func git(_ args: String...) throws -> Data {
    let process = Process(), pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    process.currentDirectoryURL = package
    process.arguments = args
    process.environment = env.merging(["GIT_OPTIONAL_LOCKS": "0"]) { _, new in new }
    process.standardOutput = pipe
    try process.run()
    let bytes = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else { throw Failure.gitFailed }
    return bytes
}
func string(_ data: Data) -> String { String(decoding: data, as: UTF8.self).trimmingCharacters(in: .newlines) }
func hex(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
func boundIdentity() throws -> [String: Any] {
    let root = URL(fileURLWithPath: string(try git("rev-parse", "--show-toplevel"))).resolvingSymlinksInPath()
    guard root.appendingPathComponent("brain-bar").resolvingSymlinksInPath() == package.resolvingSymlinksInPath() else {
        throw Failure.invalidRoot
    }
    // Run from the repository root, so Git's tracked paths are repository-relative.
    let names = try git("-C", root.path, "ls-files", "-z").split(separator: 0).sorted { $0.lexicographicallyPrecedes($1) }
    var manifest = Data()
    for name in names {
        let relative = String(decoding: name, as: UTF8.self)
        let path = root.appendingPathComponent(relative)
        let values = try path.resourceValues(forKeys: [.isSymbolicLinkKey])
        let bytes = values.isSymbolicLink == true
            ? Data(try FileManager.default.destinationOfSymbolicLink(atPath: path.path).utf8)
            : try Data(contentsOf: path)
        manifest.append(contentsOf: name)
        manifest.append(0)
        manifest.append(contentsOf: hex(bytes).utf8)
        manifest.append(10)
    }
    return [
        "schema_version": 1, "root": root.path,
        "head": string(try git("rev-parse", "HEAD")), "tree": string(try git("rev-parse", "HEAD^{tree}")),
        "dirty": !(try git("-C", root.path, "status", "--porcelain", "-z", "--untracked-files=all")).isEmpty,
        "source_sha256": hex(manifest), "source_files": names.count,
    ]
}
let identity: [String: Any]
do {
    identity = try boundIdentity()
} catch Failure.gitFailed {
    // Archive-based retirement/release builds remain possible. Such executables
    // explicitly lack source proof and are rejected by the UI receipt runner.
    print("warning: BrainBar build-source proof unavailable: Git metadata missing")
    identity = ["schema_version": 1, "root": package.deletingLastPathComponent().path,
                "head": "unavailable", "tree": "unavailable", "dirty": true,
                "source_sha256": "unavailable", "source_files": 0,
                "unbound_reason": "Git metadata unavailable"]
}
let json = String(decoding: try JSONSerialization.data(withJSONObject: identity, options: [.sortedKeys]), as: UTF8.self)
let literal = json.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
let generated = Data("#if DEBUG\nenum BrainBarCompiledSource { static let json = \"\(literal)\" }\n#endif\n".utf8)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let file = output.appendingPathComponent("BrainBarCompiledSource.swift")
if (try? Data(contentsOf: file)) != generated { try generated.write(to: file, options: .atomic) }
