import Foundation

/// MOSS-TTS 生成选项
/// 
/// 控制语音合成的各种参数
public struct MOSSTTSOptions: Sendable, Equatable {
    
    // MARK: - Generation Options
    
    /// 采样温度 (0.0 - 1.0)
    /// - 较低值: 更确定性的输出
    /// - 较高值: 更有创造性但可能不稳定
    public var temperature: Float
    
    /// Top-K 采样数量
    /// - 限制考虑的最高概率 token 数量
    /// - 较小的值使输出更确定性
    public var topK: Int
    
    /// 最大生成长度 (token 数量)
    public var maxLength: Int
    
    /// 最大生成音频帧数。
    ///
    /// MOSS Audio Tokenizer 每帧约 80ms (48kHz / 3840 samples)。默认使用较保守的
    /// 32 帧，便于客户端早期接入测试；调用方可按需要调高。
    public var maxGeneratedFrames: Int?

    /// 长文本自动切分的最大 token 数。
    ///
    /// 当文本超过这个预算时，`speak(...)` 会优先按句号/逗号等标点切分，再按 token 预算兜底切分。
    public var maxTextTokensPerChunk: Int

    /// 采样随机种子。
    ///
    /// MOSS-TTS-Nano 官方 ONNX runtime 默认使用固定 seed，因此这里也默认使用 `1234`，
    /// 以获得更稳定、可复现的 fixed-sampling 输出。
    public var seed: UInt64?
    
    /// 批量大小 (用于批处理优化)
    public var batchSize: Int
    
    // MARK: - Audio Options
    
    /// 输出采样率
    public var sampleRate: Int
    
    /// 音频通道数 (1: 单声道, 2: 立体声)
    public var channels: Int
    
    /// 是否启用平滑
    public var enableSmoothing: Bool
    
    // MARK: - Quality Options
    
    /// 推理精度
    public var precision: Precision
    
    /// 是否使用 CoreML 加速 (Apple Silicon)
    public var useCoreML: Bool
    
    /// 线程数 (CPU 模式)
    public var numThreads: Int
    
    // MARK: - Types
    
    public enum Precision: String, Sendable, CaseIterable {
        case float16 = "fp16"
        case float32 = "fp32"
        
        public var description: String {
            switch self {
            case .float16: return "半精度 (FP16)"
            case .float32: return "全精度 (FP32)"
            }
        }
    }
    
    // MARK: - Initialization
    
    /// 默认选项
    public static let `default` = MOSSTTSOptions()
    
    /// 快速模式 (低延迟)
    public static let fast = MOSSTTSOptions(
        temperature: 0.5,
        topK: 20,
        maxLength: 5000,
        maxGeneratedFrames: 16,
        maxTextTokensPerChunk: 75,
        seed: 1234,
        batchSize: 1,
        sampleRate: 24000,
        channels: 1,
        enableSmoothing: false,
        precision: .float32,
        useCoreML: true,
        numThreads: 2
    )
    
    /// 高质量模式
    public static let highQuality = MOSSTTSOptions(
        temperature: 0.7,
        topK: 100,
        maxLength: 20000,
        maxGeneratedFrames: 375,
        maxTextTokensPerChunk: 75,
        seed: 1234,
        batchSize: 4,
        sampleRate: 48000,
        channels: 2,
        enableSmoothing: true,
        precision: .float32,
        useCoreML: true,
        numThreads: 8
    )
    
    /// 公开初始化器
    public init(
        temperature: Float = 0.6,
        topK: Int = 50,
        maxLength: Int = 10000,
        maxGeneratedFrames: Int? = 32,
        maxTextTokensPerChunk: Int = 75,
        seed: UInt64? = 1234,
        batchSize: Int = 1,
        sampleRate: Int = 48000,
        channels: Int = 2,
        enableSmoothing: Bool = true,
        precision: Precision = .float32,
        useCoreML: Bool = true,
        numThreads: Int = 4
    ) {
        self.temperature = temperature
        self.topK = topK
        self.maxLength = maxLength
        self.maxGeneratedFrames = maxGeneratedFrames
        self.maxTextTokensPerChunk = maxTextTokensPerChunk
        self.seed = seed
        self.batchSize = batchSize
        self.sampleRate = sampleRate
        self.channels = channels
        self.enableSmoothing = enableSmoothing
        self.precision = precision
        self.useCoreML = useCoreML
        self.numThreads = numThreads
    }
}

// MARK: - Validation

extension MOSSTTSOptions {
    
    /// 验证选项
    public func validate() -> [ValidationError] {
        var errors: [ValidationError] = []
        
        if temperature < 0 || temperature > 1 {
            errors.append(.invalidTemperature(temperature))
        }
        
        if topK < 1 {
            errors.append(.invalidTopK(topK))
        }
        
        if maxLength < 1 || maxLength > 50000 {
            errors.append(.invalidMaxLength(maxLength))
        }
        
        if let maxGeneratedFrames, maxGeneratedFrames < 1 || maxGeneratedFrames > 50000 {
            errors.append(.invalidMaxGeneratedFrames(maxGeneratedFrames))
        }

        if maxTextTokensPerChunk < 1 || maxTextTokensPerChunk > 50000 {
            errors.append(.invalidMaxTextTokensPerChunk(maxTextTokensPerChunk))
        }
        
        if sampleRate != 24000 && sampleRate != 48000 {
            errors.append(.invalidSampleRate(sampleRate))
        }
        
        if channels < 1 || channels > 2 {
            errors.append(.invalidChannels(channels))
        }
        
        return errors
    }
    
    public enum ValidationError: Error, LocalizedError {
        case invalidTemperature(Float)
        case invalidTopK(Int)
        case invalidMaxLength(Int)
        case invalidMaxGeneratedFrames(Int)
        case invalidMaxTextTokensPerChunk(Int)
        case invalidSampleRate(Int)
        case invalidChannels(Int)
        
        public var errorDescription: String? {
            switch self {
            case .invalidTemperature(let value):
                return "温度必须在 0.0-1.0 之间，当前值: \(value)"
            case .invalidTopK(let value):
                return "Top-K 必须 >= 1，当前值: \(value)"
            case .invalidMaxLength(let value):
                return "最大长度必须在 1-50000 之间，当前值: \(value)"
            case .invalidMaxGeneratedFrames(let value):
                return "最大音频帧数必须在 1-50000 之间，当前值: \(value)"
            case .invalidMaxTextTokensPerChunk(let value):
                return "长文本切分 token 预算必须在 1-50000 之间，当前值: \(value)"
            case .invalidSampleRate(let value):
                return "采样率必须是 24000 或 48000，当前值: \(value)"
            case .invalidChannels(let value):
                return "通道数必须是 1 或 2，当前值: \(value)"
            }
        }
    }
}
