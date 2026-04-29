// MOSSTTSModels.swift
// 数据模型定义

import Foundation
import OnnxRuntimeBindings

// MARK: - 模型家族

/// MOSS-TTS 模型家族
public enum MOSSModelFamily: String, Sendable {
    case moss  // MOSS-TTS-Nano
}

// MARK: - 模型变体

/// MOSS-TTS 模型大小变体
public enum MOSSModelVariant: String, CaseIterable, Sendable {
    case mossTTSNano100M = "100M"
    
    /// 模型家族
    public var family: MOSSModelFamily {
        return .moss
    }
    
    /// 显示名称
    public var displayName: String {
        switch self {
        case .mossTTSNano100M:
            return "MOSS-TTS-Nano 100M"
        }
    }
    
    /// HuggingFace TTS 模型仓库
    public var modelRepo: String {
        switch self {
        case .mossTTSNano100M:
            return "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
        }
    }
    
    /// HuggingFace Audio Tokenizer 仓库
    public var tokenizerRepo: String {
        switch self {
        case .mossTTSNano100M:
            return "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
        }
    }
    
    /// 预估模型大小 (bytes)
    public var estimatedSize: Int64 {
        switch self {
        case .mossTTSNano100M:
            return 708_000_000  // ~708MB (ONNX 模型包)
        }
    }
    
    /// 当前平台默认变体
    public static var defaultForCurrentPlatform: MOSSModelVariant {
        return .mossTTSNano100M
    }
}

// MARK: - 说话人/音色

/// 说话人配置（用于语音克隆）
public struct MOSSSpeaker: Sendable, Codable {
    /// 说话人名称
    public let name: String
    
    /// 参考音频路径（本地文件 URL 或 HuggingFace 路径）
    public let referenceAudioPath: String?
    
    /// 参考音频数据（如果已在内存中）
    public let referenceAudioData: Data?
    
    /// 已缓存的说话人 embedding。
    ///
    /// MOSS-TTS-Nano 的 voice cloning 通常需要先把参考音频编码成 prompt/acoustic codes。
    /// 在真实 ONNX prompt 编码链路完成前，调用方可以把已计算好的 embedding 放在这里，
    /// 包内 API 会稳定传递它，不再悄悄丢弃 speaker 信息。
    public let embedding: [Float]?
    
    /// 参考音频通过 MOSS Audio Tokenizer 编码后的 acoustic codes。
    ///
    /// 这是语音克隆链路里比通用 embedding 更贴近 MOSS-TTS-Nano 的中间表示。
    public let referenceAudioCodes: [[Int32]]?
    
    public init(
        name: String,
        referenceAudioPath: String? = nil,
        referenceAudioData: Data? = nil,
        embedding: [Float]? = nil,
        referenceAudioCodes: [[Int32]]? = nil
    ) {
        self.name = name
        self.referenceAudioPath = referenceAudioPath
        self.referenceAudioData = referenceAudioData
        self.embedding = embedding
        self.referenceAudioCodes = referenceAudioCodes
    }
}

// MARK: - 语音生成结果

/// 语音生成结果
public struct MOSSSpeechResult: Sendable {
    /// 音频采样数据（Float, 范围 -1.0 ~ 1.0）
    public let audioSamples: [Float]
    
    /// 采样率 (Hz)
    public let sampleRate: Int
    
    /// 生成耗时统计
    public let timings: MOSSSpeechTimings
    
    /// 音频时长（秒）
    public var audioDuration: Double {
        Double(audioSamples.count) / Double(sampleRate)
    }
    
    public init(audioSamples: [Float], sampleRate: Int, timings: MOSSSpeechTimings) {
        self.audioSamples = audioSamples
        self.sampleRate = sampleRate
        self.timings = timings
    }
}

// MARK: - 语音生成耗时

/// 语音生成各阶段耗时统计
public struct MOSSSpeechTimings: Sendable {
    /// 模型加载时间
    public var modelLoading: TimeInterval = 0
    
    /// Tokenizer 加载时间
    public var tokenizerLoading: TimeInterval = 0
    
    /// 文本编码时间
    public var tokenization: TimeInterval = 0
    
    /// 预填充阶段时间
    public var prefill: TimeInterval = 0
    
    /// 解码循环总时间
    public var decodingLoop: TimeInterval = 0
    
    /// 音频解码时间
    public var audioDecoding: TimeInterval = 0
    
    /// 完整流程总时间
    public var fullPipeline: TimeInterval = 0
    
    /// 总解码步数
    public var totalDecodingLoops: Double = 0
    
    /// 音频时长（秒）
    public var inputAudioSeconds: TimeInterval = 0
    
    /// 初始化
    public init() {}
    
    /// 合并其他计时数据
    public mutating func merge(_ other: MOSSSpeechTimings) {
        tokenization += other.tokenization
        prefill += other.prefill
        decodingLoop += other.decodingLoop
        audioDecoding += other.audioDecoding
        totalDecodingLoops += other.totalDecodingLoops
    }
}

// MARK: - 模型加载状态

/// 模型加载状态
public enum MOSSModelState: Sendable {
    case unloaded
    case downloading
    case downloaded
    case loading
    case loaded
    case prewarming
    case prewarmed
    case unloading
}

// MARK: - 音频格式

/// 音频格式配置
public struct MOSSAudioFormat: Sendable {
    /// 采样率 (Hz)
    public let sampleRate: Int
    
    /// 声道数
    public let channels: Int
    
    /// 每帧样本数
    public let samplesPerFrame: Int
    
    /// 默认 48kHz 立体声格式
    public static let defaultFormat = MOSSAudioFormat(
        sampleRate: 48000,
        channels: 2,
        samplesPerFrame: 960  // 20ms @ 48kHz
    )
    
    public init(sampleRate: Int, channels: Int, samplesPerFrame: Int) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.samplesPerFrame = samplesPerFrame
    }
}

// MARK: - 生成进度

/// 生成进度回调信息
public struct MOSSProgress: Sendable {
    /// 已生成的音频样本
    public let audioSamples: [Float]
    
    /// 当前步数
    public let currentStep: Int
    
    /// 总步数
    public let totalSteps: Int
    
    /// 进度百分比 (0.0 ~ 1.0)
    public var progress: Double {
        guard totalSteps > 0 else { return 0 }
        return Double(currentStep) / Double(totalSteps)
    }
    
    public init(audioSamples: [Float], currentStep: Int, totalSteps: Int) {
        self.audioSamples = audioSamples
        self.currentStep = currentStep
        self.totalSteps = totalSteps
    }
}

/// 进度回调闭包
/// 返回 false 可取消生成
public typealias MOSSProgressCallback = (@Sendable (MOSSProgress) -> Bool)?

// MARK: - 错误类型

/// MOSSTTSKit 错误类型
public enum MOSSTTSError: Error, LocalizedError {
    case modelNotFound(String)
    case modelLoadFailed(String)
    case tokenizerNotFound(String)
    case inferenceFailed(String)
    case invalidInput(String)
    case generationFailed(String)
    case audioProcessingFailed(String)
    case downloadFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return "模型未找到: \(msg)"
        case .modelLoadFailed(let msg): return "模型加载失败: \(msg)"
        case .tokenizerNotFound(let msg): return "分词器未找到: \(msg)"
        case .inferenceFailed(let msg): return "推理失败: \(msg)"
        case .invalidInput(let msg): return "无效输入: \(msg)"
        case .generationFailed(let msg): return "生成失败: \(msg)"
        case .audioProcessingFailed(let msg): return "音频处理失败: \(msg)"
        case .downloadFailed(let msg): return "下载失败: \(msg)"
        }
    }
}

// MARK: - KV Cache

/// KV Cache 用于自回归模型加速
public struct KVCache: Sendable {
    /// Key cache
    public var keyCache: [Float]
    
    /// Value cache
    public var valueCache: [Float]
    
    /// Cache 形状 [num_layers, 2, batch, num_heads, seq_len, head_dim]
    public let shape: [Int]
    
    public init(shape: [Int] = [24, 2, 1, 4, 0, 64]) {
        self.shape = shape
        self.keyCache = []
        self.valueCache = []
    }
    
    /// 更新 cache
    public mutating func update(keys: [Float], values: [Float], seqLen: Int) {
        keyCache.append(contentsOf: keys)
        valueCache.append(contentsOf: values)
    }
    
    /// 清空 cache
    public mutating func reset() {
        keyCache.removeAll()
        valueCache.removeAll()
    }
}

/// 音频帧
public struct AudioFrame: Sendable {
    /// 音频数据
    public let samples: [Float]
    
    /// 帧索引
    public let index: Int
    
    /// 时间戳（秒）
    public let timestamp: Double
    
    public init(samples: [Float], index: Int, timestamp: Double = 0) {
        self.samples = samples
        self.index = index
        self.timestamp = timestamp
    }
}

/// ONNX Runtime 执行提供者类型
public enum ONNXExecutionProvider: String, Sendable {
    case cpu = "CPU"
    case coreml = "CoreML"
    case cuda = "CUDA"
}

/// ONNX Runtime 会话选项
public struct ONNXSessionOptions: Sendable {
    public var intraOpNumThreads: Int = 4
    public var interOpNumThreads: Int = 4
    public var executionProviders: [ONNXExecutionProvider] = [.cpu]
    
    public init() {}
}

/// ONNX 张量数据类型
public enum ONNXTensorDataType: Sendable {
    case float32
    case int64
    case int32
    case uint8
    case float16
    
    public var size: Int {
        switch self {
        case .float32: return 4
        case .int64: return 8
        case .int32: return 4
        case .uint8: return 1
        case .float16: return 2
        }
    }
}

/// ONNX 张量
public struct ONNXTensor: Sendable {
    public let shape: [Int]
    public let dataType: ONNXTensorDataType
    public let data: Data
    
    public init(shape: [Int], dataType: ONNXTensorDataType, data: Data) {
        self.shape = shape
        self.dataType = dataType
        self.data = data
    }
    
    /// 从 Float 数组创建
    public static func floats(_ values: [Float], shape: [Int]) -> ONNXTensor {
        let data = values.withUnsafeBytes { Data($0) }
        return ONNXTensor(shape: shape, dataType: .float32, data: data)
    }
    
    /// 从 Int64 数组创建
    public static func int64s(_ values: [Int64], shape: [Int]) -> ONNXTensor {
        let data = values.withUnsafeBytes { Data($0) }
        return ONNXTensor(shape: shape, dataType: .int64, data: data)
    }
    
    /// 从 Int32 数组创建
    public static func int32s(_ values: [Int32], shape: [Int]) -> ONNXTensor {
        let data = values.withUnsafeBytes { Data($0) }
        return ONNXTensor(shape: shape, dataType: .int32, data: data)
    }
    
    /// 转换为 Float 数组
    public func toFloats() -> [Float]? {
        guard dataType == .float32 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }
    
    /// 转换为 Int64 数组
    public func toInt64s() -> [Int64]? {
        guard dataType == .int64 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int64.self))
        }
    }
    
    /// 转换为 Int32 数组
    public func toInt32s() -> [Int32]? {
        guard dataType == .int32 else { return nil }
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int32.self))
        }
    }
    
    /// 元素总数
    public var elementCount: Int {
        shape.reduce(1, *)
    }
}

/// ONNX Runtime 会话（框架）
public final class ONNXSession: @unchecked Sendable {
    /// 模型路径
    public let modelPath: String
    
    /// 会话选项
    public let options: ONNXSessionOptions
    
    /// 模型输入名称
    public private(set) var inputNames: [String] = []
    
    /// 模型输出名称
    public private(set) var outputNames: [String] = []
    
    /// 是否已加载
    public private(set) var isLoaded: Bool = false
    
    private let env: ORTEnv
    private let session: ORTSession
    
    public init(modelPath: String, options: ONNXSessionOptions = ONNXSessionOptions()) throws {
        self.modelPath = modelPath
        self.options = options
        
        // 验证文件存在
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw MOSSTTSError.modelNotFound("Model file not found: \(modelPath)")
        }
        
        self.env = try ORTEnv(loggingLevel: .warning)
        
        let ortOptions = try ORTSessionOptions()
        try ortOptions.setGraphOptimizationLevel(.all)
        self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: ortOptions)
        self.inputNames = try session.inputNames()
        self.outputNames = try session.outputNames()
        
        isLoaded = true
    }
    
    /// 执行推理
    public func run(inputs: [String: ONNXTensor], outputs: [String]) throws -> [String: ONNXTensor] {
        guard isLoaded else {
            throw MOSSTTSError.inferenceFailed("Session not loaded")
        }
        
        var ortInputs: [String: ORTValue] = [:]
        for (name, tensor) in inputs {
            let tensorData = NSMutableData(data: tensor.data)
            let shape = tensor.shape.map { NSNumber(value: $0) }
            let value = try ORTValue(
                tensorData: tensorData,
                elementType: try tensor.dataType.ortElementType(),
                shape: shape
            )
            ortInputs[name] = value
        }
        
        let requestedOutputs = outputs.isEmpty ? Set(outputNames) : Set(outputs)
        let ortOutputs = try session.run(
            withInputs: ortInputs,
            outputNames: requestedOutputs,
            runOptions: nil
        )
        
        var result: [String: ONNXTensor] = [:]
        for (name, value) in ortOutputs {
            let typeInfo = try value.typeInfo()
            guard let tensorInfo = typeInfo.tensorTypeAndShapeInfo else {
                throw MOSSTTSError.inferenceFailed("Output is not a tensor: \(name)")
            }
            
            let tensorData = try value.tensorData()
            let data = Data(referencing: tensorData)
            let shape = tensorInfo.shape.map { $0.intValue }
            let dataType = try ONNXTensorDataType(ortElementType: tensorInfo.elementType)
            result[name] = ONNXTensor(shape: shape, dataType: dataType, data: data)
        }
        
        return result
    }
    
    /// 卸载会话
    public func unload() {
        isLoaded = false
        inputNames = []
        outputNames = []
    }
    
    deinit {
        unload()
    }
}

private extension ONNXTensorDataType {
    func ortElementType() throws -> ORTTensorElementDataType {
        switch self {
        case .float32:
            return .float
        case .int64:
            return .int64
        case .int32:
            return .int32
        case .uint8:
            return .uInt8
        case .float16:
            throw MOSSTTSError.invalidInput("ONNX Runtime Swift binding does not expose float16 tensor creation")
        }
    }
    
    init(ortElementType: ORTTensorElementDataType) throws {
        switch ortElementType {
        case .float:
            self = .float32
        case .int64:
            self = .int64
        case .int32:
            self = .int32
        case .uInt8:
            self = .uint8
        default:
            throw MOSSTTSError.inferenceFailed("Unsupported ONNX tensor element type: \(ortElementType)")
        }
    }
}
