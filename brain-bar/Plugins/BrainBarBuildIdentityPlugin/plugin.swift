import PackagePlugin
import Foundation

@main
struct BrainBarBuildIdentityPlugin: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
        // Prebuild also runs on HEAD-only changes. SwiftPM supplies the source location.
        let script = context.package.directoryURL.appendingPathComponent("Sources/BrainBarBuildIdentity/main.swift")
        return [.prebuildCommand(
            displayName: "Bind BrainBar executable to build source",
            executable: URL(fileURLWithPath: "/usr/bin/xcrun"),
            arguments: ["swift", script.path, context.package.directoryURL.path, context.pluginWorkDirectoryURL.path],
            outputFilesDirectory: context.pluginWorkDirectoryURL
        )]
    }
}
