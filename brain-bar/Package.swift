// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BrainBar",
    platforms: [
        .macOS(.v14),
    ],
    targets: [
        .executableTarget(
            name: "BrainBar",
            path: "Sources/BrainBar",
            linkerSettings: [
                .linkedLibrary("sqlite3"),
            ]
        ),
        .testTarget(
            name: "BrainBarTests",
            dependencies: ["BrainBar"],
            path: "Tests/BrainBarTests"
        ),
    ]
)
