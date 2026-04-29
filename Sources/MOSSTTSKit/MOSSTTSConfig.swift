// MOSSTTSConfig.swift
// 配置管理

import Foundation

/// MOSSTTSKit 配置
public final class MOSSTTSConfig: @unchecked Sendable {
    // MARK: - 模型选择
    
    /// 模型变体
    public var modelVariant: MOSSModelVariant
    
    // MARK: - 模型位置
    
    /// 本地模型文件夹路径（如果已下载）
    public var modelFolder: URL?
    
    /// 下载基础路径（nil 使用 Hub 默认缓存）
    public var downloadBase: URL?
    
    /// HuggingFace 模型仓库 ID
    public var modelRepo: String
    
    /// HuggingFace Tokenizer 仓库 ID
    public var tokenizerRepo: String
    
    // MARK: - 认证
    
    /// HuggingFace API Token（用于私有仓库）
    public var modelToken: String?
    
    /// HuggingFace Hub 端点 URL
    public var modelEndpoint: String
    
    // MARK: - 下载选项
    
    /// 是否下载模型（true = 自动下载，false = 使用本地模型）
    public var download: Bool
    
    /// 使用后台 URLSession 下载
    public var useBackgroundDownloadSession: Bool
    
    /// 指定下载的 git revision
    public var downloadRevision: String?
    
    // MARK: - 生命周期
    
    /// 是否预热模型
    public var prewarm: Bool
    
    /// 是否立即加载模型
    public var loadImmediately: Bool
    
    // MARK: - 日志
    
    /// 是否输出详细日志
    public var verbose: Bool
    
    /// 生成随机种子（用于复现）
    public var seed: UInt64?
    
    // MARK: - 模型 URL
    
    /// 获取 TTS 模型文件夹 URL
    public var ttsModelFolder: URL? {
        guard let folder = modelFolder else { return nil }
        return folder.appendingPathComponent("tts")
    }
    
    /// 获取 Audio Tokenizer 模型文件夹 URL
    public var tokenizerModelFolder: URL? {
        guard let folder = modelFolder else { return nil }
        return folder.appendingPathComponent("audio_tokenizer")
    }
    
    // MARK: - 初始化
    
    public init(
        modelVariant: MOSSModelVariant = .mossTTSNano100M,
        modelFolder: URL? = nil,
        downloadBase: URL? = nil,
        modelRepo: String? = nil,
        tokenizerRepo: String? = nil,
        modelToken: String? = nil,
        modelEndpoint: String = "https://huggingface.co",
        download: Bool = true,
        useBackgroundDownloadSession: Bool = false,
        downloadRevision: String? = nil,
        prewarm: Bool = false,
        loadImmediately: Bool = true,
        verbose: Bool = true,
        seed: UInt64? = nil
    ) {
        self.modelVariant = modelVariant
        self.modelFolder = modelFolder
        self.downloadBase = downloadBase
        self.modelRepo = modelRepo ?? modelVariant.modelRepo
        self.tokenizerRepo = tokenizerRepo ?? modelVariant.tokenizerRepo
        self.modelToken = modelToken
        self.modelEndpoint = modelEndpoint
        self.download = download
        self.useBackgroundDownloadSession = useBackgroundDownloadSession
        self.downloadRevision = downloadRevision
        self.prewarm = prewarm
        self.loadImmediately = loadImmediately
        self.verbose = verbose
        self.seed = seed
    }
}

// MARK: - 常量

/// MOSS-TTS 常量
public enum MOSSConstants {
    /// 默认 HuggingFace 端点
    public static let defaultEndpoint = "https://huggingface.co"
    
    /// TTS 模型目录名
    public static let ttsModelDir = "tts"
    
    /// Audio Tokenizer 模型目录名
    public static let audioTokenizerModelDir = "audio_tokenizer"
    
    /// ONNX 模型文件名
    public enum ONNXFiles {
        // TTS 模型
        public static let prefill = "moss_tts_prefill.onnx"
        public static let decodeStep = "moss_tts_decode_step.onnx"
        public static let localDecoder = "moss_tts_local_decoder.onnx"
        public static let localCachedStep = "moss_tts_local_cached_step.onnx"
        public static let localFixedSampledFrame = "moss_tts_local_fixed_sampled_frame.onnx"
        
        // Audio Tokenizer
        public static let audioEncoder = "moss_audio_tokenizer_encode.onnx"
        public static let audioDecoderFull = "moss_audio_tokenizer_decode_full.onnx"
        public static let audioDecoderStep = "moss_audio_tokenizer_decode_step.onnx"
        
        // 数据文件
        public static let globalShared = "moss_tts_global_shared.data"
        public static let localShared = "moss_tts_local_shared.data"
        public static let audioEncoderData = "moss_audio_tokenizer_encode.data"
        public static let audioDecoderSharedData = "moss_audio_tokenizer_decode_shared.data"
        
        // Tokenizer
        public static let tokenizer = "tokenizer.model"
    }
    
    /// 音频格式
    public enum Audio {
        public static let defaultSampleRate = 48000
        public static let defaultChannels = 2
        public static let defaultSamplesPerFrame = 960  // 20ms @ 48kHz
    }
    
    /// 生成参数默认值
    public enum Defaults {
        public static let maxTokens = 2048
        public static let temperature: Float = 1.0
        public static let topK = 50
        public static let repetitionPenalty: Float = 1.0
    }
}
