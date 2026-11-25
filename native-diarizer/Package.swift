// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "NixScribeDiarizer",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git",
                 exact: "0.18.4", traits: ["Diarizer"])
    ],
    targets: [
        .executableTarget(name: "NixScribeDiarizer", dependencies: [
            .product(name: "FluidAudio", package: "FluidAudio")
        ])
    ]
)
