// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MOSSTTSKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "MOSSTTSKit",
            targets: ["MOSSTTSKit"]
        ),
    ],
    dependencies: [
        // HuggingFace Transformers Swift (for Hub and Tokenizers)
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.6"),
        // ONNX Runtime Swift (Microsoft 官方)
        // 注意：产品名是 "onnxruntime"（静态库），它暴露 OnnxRuntimeBindings 模块
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager.git", from: "1.24.2"),
    ],
    targets: [
        .target(
            name: "MOSSTTSKit",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            resources: [
                .process("Resources")
            ]
        ),
        .executableTarget(
            name: "MOSSTTSSampleCLI",
            dependencies: ["MOSSTTSKit"],
            path: "Examples/MOSSTTSSampleCLI"
        ),
        .executableTarget(
            name: "MOSSFrameDumpCLI",
            dependencies: ["MOSSTTSKit"],
            path: "Examples/MOSSFrameDumpCLI"
        ),
        .executableTarget(
            name: "MOSSRegressionSamplesCLI",
            dependencies: ["MOSSTTSKit"],
            path: "Examples/MOSSRegressionSamplesCLI"
        ),
        .testTarget(
            name: "MOSSTTSKitTests",
            dependencies: ["MOSSTTSKit"]
        ),
    ]
)
