// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "PenGPTSwift",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "PenGPT", targets: ["PenGPT"]),
        .executable(name: "pengpt-convert", targets: ["PenGPTConvert"]),
    ],
    targets: [
        .target(name: "PenGPT"),
        .target(name: "PenGPTModelImport", dependencies: ["PenGPT"]),
        .executableTarget(name: "PenGPTConvert", dependencies: ["PenGPTModelImport"]),
        .testTarget(
            name: "PenGPTTests",
            dependencies: ["PenGPT", "PenGPTModelImport"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
