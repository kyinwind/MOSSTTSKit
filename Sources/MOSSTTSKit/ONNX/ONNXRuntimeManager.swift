// ONNXRuntimeManager.swift
// ONNX Runtime 管理器
//
// 注意：这是一个框架实现
// 完整的 ONNX Runtime 集成需要调用 C API

import Foundation

/// ONNX Runtime 管理器
public final class ONNXRuntimeManager: @unchecked Sendable {
    // MARK: - 单例
    
    public static let shared = ONNXRuntimeManager()
    
    // MARK: - 属性
    
    private let queue = DispatchQueue(label: "com.mossttskit.onnxruntime", qos: .userInitiated)
    
    /// 模型目录
    public let modelDirectory: URL
    
    // MARK: - 初始化
    
    private init() {
        // 默认模型目录
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        self.modelDirectory = caches.appendingPathComponent("MOSSTTSKit/Models", isDirectory: true)
        
        // 创建目录
        try? FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
    }
    
    // MARK: - 模型管理
    
    /// 检查模型文件是否存在
    public func modelExists(name: String) -> Bool {
        let path = modelDirectory.appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: path.path)
    }
    
    /// 获取模型完整路径
    public func modelPath(for name: String) -> URL {
        return modelDirectory.appendingPathComponent(name)
    }
    
    /// 列出所有已下载的模型
    public func listModels() -> [String] {
        guard let contents = try? FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        ) else { return [] }
        
        return contents
            .filter { $0.pathExtension == "onnx" }
            .map { $0.deletingPathExtension().lastPathComponent }
    }
}
