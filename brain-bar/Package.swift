// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "BrainBar",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "BrainBar", targets: ["BrainBar"]),
        .executable(name: "BrainBarDaemon", targets: ["BrainBarDaemon"]),
    ],
    dependencies: [
        .package(url: "https://github.com/groue/GRDB.swift.git", from: "7.5.0"),
    ],
    targets: [
        .executableTarget(
            name: "BrainBar",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            path: "Sources/BrainBar",
            linkerSettings: [
                .linkedLibrary("sqlite3"),
            ]
        ),
        .executableTarget(
            name: "BrainBarDaemon",
            dependencies: [
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            path: "Sources/BrainBarDaemon",
            linkerSettings: [
                .linkedLibrary("sqlite3"),
            ]
        ),
        .testTarget(
            name: "BrainBarTests",
            dependencies: [
                "BrainBar",
                .product(name: "GRDB", package: "GRDB.swift"),
            ],
            path: "Tests/BrainBarTests"
        ),
    ]
)
