import Foundation
import OnnxRuntimeBindings

/// ONNX Runtime Swift 封装层
/// 
/// 使用 Microsoft 官方的 onnxruntime-swift-package-manager
/// GitHub: https://github.com/microsoft/onnxruntime-swift-package-manager
///
/// 架构:
/// - ORTEnv: ONNX Runtime 环境
/// - ORTSession: 模型推理会话
/// - ORTSessionOptions: 会话配置
/// - ORTValue: 输入/输出张量数据
///
/// 使用示例:
/// ```swift
/// let engine = try await ONNXSwift(modelPath: "/path/to/model.onnx")
/// let result = try await engine.run(inputs: ["input": inputData])
/// ```
public actor ONNXSwift {
    
    // MARK: - Types
    
    /// 推理结果
    public struct InferenceResult: Sendable {
        public let outputTensor: [Float]
        public let shape: [Int64]
        
        public init(outputTensor: [Float], shape: [Int64]) {
            self.outputTensor = outputTensor
            self.shape = shape
        }
    }
    
    /// 错误类型
    public enum ONNXSwiftError: Error, LocalizedError {
        case environmentCreationFailed(String)
        case sessionCreationFailed(String)
        case sessionOptionsCreationFailed
        case valueCreationFailed(String)
        case runFailed(String)
        case modelPathInvalid
        case outputNotFound(String)
        
        public var errorDescription: String? {
            switch self {
            case .environmentCreationFailed(let msg):
                return "ONNX Runtime 环境创建失败: \(msg)"
            case .sessionCreationFailed(let msg):
                return "ONNX 会话创建失败: \(msg)"
            case .sessionOptionsCreationFailed:
                return "会话选项创建失败"
            case .valueCreationFailed(let msg):
                return "张量创建失败: \(msg)"
            case .runFailed(let msg):
                return "推理运行失败: \(msg)"
            case .modelPathInvalid:
                return "模型路径无效"
            case .outputNotFound(let name):
                return "找不到输出张量: \(name)"
            }
        }
    }
    
    // MARK: - Properties
    
    private let env: ORTEnv
    private let session: ORTSession
    
    // MARK: - Initialization
    
    /// 初始化 ONNX Runtime
    /// - Parameters:
    ///   - modelPath: ONNX 模型文件路径
    ///   - options: 可选的会话配置
    public init(modelPath: String, options: ORTSessionOptions? = nil) async throws {
        // 验证路径
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw ONNXSwiftError.modelPathInvalid
        }
        
        // 创建环境
        do {
            self.env = try ORTEnv(loggingLevel: .warning)
        } catch {
            throw ONNXSwiftError.environmentCreationFailed(error.localizedDescription)
        }
        
        // 创建会话
        do {
            // 创建默认选项（如果需要）
            let sessionOptions: ORTSessionOptions
            if let opts = options {
                sessionOptions = opts
            } else {
                sessionOptions = try ORTSessionOptions()
                try sessionOptions.setGraphOptimizationLevel(.all)
            }
            
            self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: sessionOptions)
        } catch {
            throw ONNXSwiftError.sessionCreationFailed(error.localizedDescription)
        }
    }
    
    // MARK: - Inference
    
    /// 运行推理
    /// - Parameters:
    ///   - inputs: 输入张量字典 [name: data]
    ///   - outputName: 输出张量名称（可选）
    /// - Returns: 推理结果
    public func run(inputs: [String: [Float]], outputName: String? = nil) async throws -> InferenceResult {
        // 准备输入张量
        var ortInputs: [String: ORTValue] = [:]
        
        for (name, data) in inputs {
            // 创建张量数据
            let tensorData = NSMutableData(length: data.count * MemoryLayout<Float>.size)!
            data.withUnsafeBytes { buffer in
                tensorData.replaceBytes(in: NSRange(location: 0, length: buffer.count), withBytes: buffer.baseAddress!)
            }
            
            // Shape: [1, sequence_length] - 使用 NSNumber
            let shape: [NSNumber] = [NSNumber(value: 1), NSNumber(value: data.count)]
            
            do {
                let value = try ORTValue(tensorData: tensorData, elementType: .float, shape: shape)
                ortInputs[name] = value
            } catch {
                throw ONNXSwiftError.valueCreationFailed("\(name): \(error.localizedDescription)")
            }
        }
        
        // 默认输出名称
        let targetOutput = outputName ?? "output"
        
        // 运行推理
        var outputs: [String: ORTValue] = [:]
        do {
            outputs = try session.run(withInputs: ortInputs, outputNames: Set<String>([targetOutput]), runOptions: nil)
        } catch {
            throw ONNXSwiftError.runFailed(error.localizedDescription)
        }
        
        // 提取输出
        guard let outputValue = outputs[targetOutput] else {
            throw ONNXSwiftError.outputNotFound(targetOutput)
        }
        
        // 提取数据
        var floatData: [Float] = []
        var shape: [Int64] = [0]
        
        do {
            let tensorData = try outputValue.tensorData()
            let byteCount = tensorData.length
            let floatCount = byteCount / MemoryLayout<Float>.size
            
            floatData = [Float](repeating: 0, count: floatCount)
            tensorData.getBytes(&floatData, length: byteCount)
            
            // 获取形状
            if let typeInfo = try? outputValue.typeInfo(),
               let info = typeInfo.tensorTypeAndShapeInfo {
                shape = info.shape.map { $0.int64Value }
            }
        } catch {
            throw ONNXSwiftError.runFailed("无法读取张量: \(error.localizedDescription)")
        }
        
        return InferenceResult(outputTensor: floatData, shape: shape)
    }
    
    // MARK: - Model Info
    
    /// 获取模型输入名称
    public func getInputInfo() -> [String] {
        (try? session.inputNames()) ?? []
    }
    
    /// 获取模型输出名称
    public func getOutputInfo() -> [String] {
        (try? session.outputNames()) ?? []
    }
}

// MARK: - CoreML Support

extension ONNXSwift {
    
    /// 创建支持 CoreML 的引擎（Apple Silicon Neural Engine）
    public static func createWithCoreML(modelPath: String, enableOnSubgraphs: Bool = true) async throws -> ONNXSwift {
        let options = try ORTSessionOptions()
        
        let coreMLOptions = ORTCoreMLExecutionProviderOptions()
        coreMLOptions.enableOnSubgraphs = enableOnSubgraphs
        try options.appendCoreMLExecutionProvider(with: coreMLOptions)
        
        return try await ONNXSwift(modelPath: modelPath, options: options)
    }
    
    /// CoreML 是否可用
    public static var isCoreMLAvailable: Bool {
        return ORTIsCoreMLExecutionProviderAvailable()
    }
}
