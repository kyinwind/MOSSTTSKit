import Foundation
import AVFoundation

/// MOSSTTSKit - MOSS-TTS-Nano ONNX 推理框架
///
/// 一个封装了 ONNX Runtime 的 Swift TTS 框架，支持：
/// - 自动模型下载和缓存
/// - CPU/GPU/CoreML 多后端
/// - 音频流式输出
/// - 文本预处理
public actor MOSSTTSKit {
    
    // MARK: - Types
    
    /// 初始化选项
    public struct InitOptions: Sendable {
        /// 模型变体 (默认: 100M ONNX)
        public var variant: MOSSModelVariant = .mossTTSNano100M
        
        /// 是否自动下载模型 (默认: true)
        public var autoDownload: Bool = true
        
        /// 缓存目录 (默认: 系统缓存)
        public var cacheDir: URL? = nil
        
        /// 生成选项
        public var synthesisOptions: MOSSTTSOptions = .default
        
        /// 进度回调 (下载进度)
        public var progressCallback: (@Sendable (ModelDownloader.DownloadProgress) -> Void)? = nil
        
        public init(
            variant: MOSSModelVariant = .mossTTSNano100M,
            autoDownload: Bool = true,
            cacheDir: URL? = nil,
            synthesisOptions: MOSSTTSOptions = .default,
            progressCallback: (@Sendable (ModelDownloader.DownloadProgress) -> Void)? = nil
        ) {
            self.variant = variant
            self.autoDownload = autoDownload
            self.cacheDir = cacheDir
            self.synthesisOptions = synthesisOptions
            self.progressCallback = progressCallback
        }
    }
    
    // MARK: - Public Properties
    
    /// TTS 模型目录
    public let ttsModelDir: URL
    
    /// Audio Tokenizer 目录
    public let audioTokenizerDir: URL
    
    /// 模型变体
    public let variant: MOSSModelVariant
    
    /// TTS 引擎
    public let engine: MOSSTTSEngine
    
    /// Audio Tokenizer
    public let audioTokenizer: AudioTokenizerONNX
    
    /// Text Tokenizer
    public let textTokenizer: any MOSSTextTokenizer
    
    /// Browser ONNX runtime manifest, when present in the TTS model directory.
    public let browserManifest: MOSSBrowserManifest?
    
    /// 当前选项
    public var options: MOSSTTSOptions
    
    // MARK: - Private Properties
    
    private let downloader: ModelDownloader
    
    // MARK: - Initialization
    
    /// 便捷初始化 - 自动下载模型
    ///
    /// 使用示例:
    /// ```swift
    /// let tts = try await MOSSTTSKit()
    /// let result = try await tts.speak(text: "你好")
    /// ```
    public init(options: InitOptions = InitOptions()) async throws {
        self.variant = options.variant
        self.options = options.synthesisOptions
        self.downloader = ModelDownloader(cacheDir: options.cacheDir)
        
        // 自动下载或使用本地模型
        let modelPaths: ModelPaths
        
        if options.autoDownload {
            modelPaths = try await downloader.downloadModels(
                variant: options.variant,
                progressCallback: options.progressCallback
            )
        } else {
            let isCached = await downloader.isModelCached(variant: options.variant)
            guard isCached else {
                throw MOSSTTSError.modelNotFound(
                    "模型未找到。请设置 autoDownload: true 或先调用 ModelDownloader.downloadModels()"
                )
            }
            
            let ttsDir = await downloader.ttsModelDir(for: options.variant)
            let tokenizerDir = await downloader.tokenizerModelDir(for: options.variant)
            
            modelPaths = ModelPaths(ttsModelDir: ttsDir, audioTokenizerDir: tokenizerDir)
        }
        
        self.ttsModelDir = modelPaths.ttsModelDir
        self.audioTokenizerDir = modelPaths.audioTokenizerDir
        try modelPaths.validate()
        
        // 初始化引擎和 tokenizer
        self.engine = try await MOSSTTSEngine(modelDir: modelPaths.ttsModelDir)
        self.audioTokenizer = try await AudioTokenizerONNX.fromDirectory(modelPaths.audioTokenizerDir.path)
        
        let tokenizerPath = modelPaths.ttsModelDir
            .appendingPathComponent(MOSSConstants.ONNXFiles.tokenizer)
            .path
        let tokenizer = SentencePieceTokenizer(tokenizerPath: tokenizerPath, modelName: options.variant.modelRepo)
        try await tokenizer.load()
        self.textTokenizer = tokenizer
        self.browserManifest = try MOSSBrowserManifest.find(in: modelPaths.ttsModelDir)
    }
    
    /// 从指定路径初始化 (不自动下载)
    ///
    /// - Parameters:
    ///   - ttsModelDir: TTS 模型目录
    ///   - audioTokenizerDir: Audio Tokenizer 目录
    ///   - options: 生成选项
    public init(
        ttsModelDir: URL,
        audioTokenizerDir: URL,
        options: MOSSTTSOptions = .default
    ) async throws {
        self.ttsModelDir = ttsModelDir
        self.audioTokenizerDir = audioTokenizerDir
        self.variant = .mossTTSNano100M
        self.options = options
        self.downloader = ModelDownloader()
        
        guard FileManager.default.fileExists(atPath: ttsModelDir.path) else {
            throw MOSSTTSError.modelNotFound("TTS 模型目录不存在: \(ttsModelDir.path)")
        }
        
        guard FileManager.default.fileExists(atPath: audioTokenizerDir.path) else {
            throw MOSSTTSError.modelNotFound("Audio Tokenizer 目录不存在: \(audioTokenizerDir.path)")
        }
        
        try ModelPaths(
            ttsModelDir: ttsModelDir,
            audioTokenizerDir: audioTokenizerDir
        )
        .validate()
        
        self.engine = try await MOSSTTSEngine(modelDir: ttsModelDir)
        self.audioTokenizer = try await AudioTokenizerONNX.fromDirectory(audioTokenizerDir.path)
        
        let tokenizerPath = ttsModelDir
            .appendingPathComponent(MOSSConstants.ONNXFiles.tokenizer)
            .path
        let tokenizer = SentencePieceTokenizer(tokenizerPath: tokenizerPath, modelName: variant.modelRepo)
        try await tokenizer.load()
        self.textTokenizer = tokenizer
        self.browserManifest = try MOSSBrowserManifest.find(in: ttsModelDir)
    }
    
    // MARK: - Synthesis
    
    /// 合成语音 (单次调用)
    ///
    /// - Parameters:
    ///   - text: 输入文本
    ///   - speaker: 说话人 (可选，用于语音克隆)
    ///   - options: 生成选项 (覆盖默认选项)
    /// - Returns: 合成结果
    public func speak(
        text: String,
        speaker: MOSSSpeaker? = nil,
        options: MOSSTTSOptions? = nil,
        progressCallback: MOSSProgressCallback = nil
    ) async throws -> TTSResult {
        let opts = options ?? self.options
        
        // 1. 文本预处理
        let processedText = preprocessText(text)
        
        // 2. 文本编码
        let textEncoding = try await textTokenizer.encode(processedText)
        let textTokens = textEncoding.ids.map { Int32($0) }
        
        // 3. 获取参考音频 codes。MOSS-TTS-Nano 使用 prompt acoustic codes 表达音色。
        let promptAudioCodes: [[Int32]]
        if let codes = speaker?.referenceAudioCodes, !codes.isEmpty {
            promptAudioCodes = codes
        } else if let codes = browserManifest?.builtinVoices.first?.promptAudioCodes, !codes.isEmpty {
            promptAudioCodes = codes
        } else {
            throw MOSSTTSError.invalidInput("No speaker reference audio codes are available")
        }
        
        guard let manifest = browserManifest else {
            throw MOSSTTSError.modelNotFound("browser_poc_manifest.json is required for ONNX generation")
        }
        
        // 4. TTS 推理：当前接入真实 ONNX 的 bounded 多帧路径。
        let maxFrames = resolvedMaxFrames(options: opts, manifest: manifest)
        let generationResult = try await engine.generateAudioCodes(
            textTokenIds: textTokens,
            promptAudioCodes: promptAudioCodes,
            manifest: manifest,
            maxFrames: maxFrames,
            assistantRandomU: 0.5,
            audioRandomU: [Float](repeating: 0.5, count: manifest.ttsConfig.nVq),
            progressCallback: progressCallback
        )
        
        // 5. 音频解码 (acoustic codes -> float samples)
        let decodedSamples = try await audioTokenizer.decode(codes: generationResult.audioCodes)
        let allSamples = adaptChannels(decodedSamples, from: audioTokenizer.numChannels, to: opts.channels)
        let outputChannels = opts.channels
        let outputSampleRate = audioTokenizer.sampleRate
        
        return TTSResult(
            audioSamples: allSamples,
            sampleRate: outputSampleRate,
            channels: outputChannels,
            duration: Double(allSamples.count) / Double(outputSampleRate * outputChannels),
            metadata: TTSMetadata(
                text: text,
                processedText: processedText,
                modelVariant: variant
            )
        )
    }
    
    /// 流式合成语音。每生成一帧 acoustic codes，都会立即解码成音频 chunk 并通过 stream 输出。
    public func speakStream(
        text: String,
        speaker: MOSSSpeaker? = nil,
        options: MOSSTTSOptions? = nil
    ) async throws -> AsyncThrowingStream<MOSSTTSStreamChunk, Error> {
        let opts = options ?? self.options
        let processedText = preprocessText(text)
        let textEncoding = try await textTokenizer.encode(processedText)
        let textTokens = textEncoding.ids.map { Int32($0) }
        
        let promptAudioCodes: [[Int32]]
        if let codes = speaker?.referenceAudioCodes, !codes.isEmpty {
            promptAudioCodes = codes
        } else if let codes = browserManifest?.builtinVoices.first?.promptAudioCodes, !codes.isEmpty {
            promptAudioCodes = codes
        } else {
            throw MOSSTTSError.invalidInput("No speaker reference audio codes are available")
        }
        
        guard let manifest = browserManifest else {
            throw MOSSTTSError.modelNotFound("browser_poc_manifest.json is required for ONNX generation")
        }
        
        let maxFrames = resolvedMaxFrames(options: opts, manifest: manifest)
        let sampleRate = audioTokenizer.sampleRate
        let channels = opts.channels
        let codeStream = await engine.streamAudioCodes(
            textTokenIds: textTokens,
            promptAudioCodes: promptAudioCodes,
            manifest: manifest,
            maxFrames: maxFrames,
            assistantRandomU: 0.5,
            audioRandomU: [Float](repeating: 0.5, count: manifest.ttsConfig.nVq)
        )
        
        return AsyncThrowingStream { continuation in
            let task = Task {
                var allSamples: [Float] = []
                do {
                    var step = 0
                    for try await frameCodes in codeStream {
                        step += 1
                        let decodedSamples = try await self.audioTokenizer.decode(codes: [frameCodes])
                        let chunkSamples = self.adaptChannels(
                            decodedSamples,
                            from: self.audioTokenizer.numChannels,
                            to: channels
                        )
                        allSamples.append(contentsOf: chunkSamples)
                        continuation.yield(
                            MOSSTTSStreamChunk(
                                audioSamples: allSamples,
                                newAudioSamples: chunkSamples,
                                currentStep: step,
                                totalSteps: maxFrames,
                                sampleRate: sampleRate,
                                channels: channels,
                                text: text,
                                processedText: processedText,
                                modelVariant: self.variant,
                                isFinal: false
                            )
                        )
                    }
                    continuation.yield(
                        MOSSTTSStreamChunk(
                            audioSamples: allSamples,
                            newAudioSamples: [],
                            currentStep: min(allSamples.isEmpty ? 0 : maxFrames, maxFrames),
                            totalSteps: maxFrames,
                            sampleRate: sampleRate,
                            channels: channels,
                            text: text,
                            processedText: processedText,
                            modelVariant: self.variant,
                            isFinal: true
                        )
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
    
    /// 合成并保存为文件
    ///
    /// - Parameters:
    ///   - text: 输入文本
    ///   - outputURL: 输出文件路径
    ///   - speaker: 说话人
    ///   - options: 生成选项
    public func speakToFile(
        text: String,
        outputURL: URL,
        speaker: MOSSSpeaker? = nil,
        options: MOSSTTSOptions? = nil
    ) async throws {
        let result = try await speak(text: text, speaker: speaker, options: options)
        try saveAsWAV(result, to: outputURL)
    }
    
    /// 合成并保存为文件，可监听生成帧进度；回调返回 `false` 会提前停止生成并保存已生成音频。
    public func speakToFile(
        text: String,
        outputURL: URL,
        speaker: MOSSSpeaker? = nil,
        options: MOSSTTSOptions? = nil,
        progressCallback: MOSSProgressCallback
    ) async throws {
        let result = try await speak(
            text: text,
            speaker: speaker,
            options: options,
            progressCallback: progressCallback
        )
        try saveAsWAV(result, to: outputURL)
    }
    
    // MARK: - Speaker Management
    
    /// 包内置可选音色。
    ///
    /// 这里返回模型 manifest 中声明的全部内置音色，调用方不需要自己读取 manifest 文件。
    /// 若模型目录缺少 `browser_poc_manifest.json`，则返回空数组。
    public var availableSpeakers: [MOSSSpeaker] {
        builtinSpeakers
    }
    
    /// 模型内置音色的稳定 API。
    ///
    /// 与 `availableSpeakers` 相同，保留这个命名是为了让调用方更直观地表达“只要模型内置音色”。
    public var builtinSpeakers: [MOSSSpeaker] {
        browserManifest?.builtinVoices.map(Self.makeBuiltinSpeaker(from:)) ?? []
    }
    
    /// 用本地参考音频创建可传入 `speak` 的克隆音色。
    ///
    /// 当前会完成参考音频读取、重采样和 Audio Tokenizer 编码，并把 acoustic codes 保存在
    /// `MOSSSpeaker.referenceAudioCodes`。完整 TTS ONNX prompt 注入会在推理循环接入真实模型
    /// 输入输出名后继续完成。
    public func makeSpeaker(
        name: String,
        referenceAudioURL: URL
    ) async throws -> MOSSSpeaker {
        let encoding = try await audioTokenizer.encode(audioPath: referenceAudioURL.path)
        return MOSSSpeaker(
            name: name,
            referenceAudioPath: referenceAudioURL.path,
            referenceAudioCodes: encoding.codes
        )
    }
    
    /// 获取说话人 embedding
    private func getSpeakerEmbedding(speaker: MOSSSpeaker?) -> [Float] {
        if let embedding = speaker?.embedding {
            return embedding
        }
        
        if let codes = speaker?.referenceAudioCodes {
            return codes.flatMap { frame in frame.map(Float.init) }
        }
        
        // 返回默认 speaker (zero vector)
        return [Float](repeating: 0, count: 192)
    }
    
    private static func makeBuiltinSpeaker(from voice: MOSSBrowserManifest.BuiltinVoice) -> MOSSSpeaker {
        MOSSSpeaker(
            identifier: voice.voice,
            name: voice.voice,
            displayName: voice.displayName,
            group: voice.group,
            audioFileName: voice.audioFile,
            referenceAudioCodes: voice.promptAudioCodes
        )
    }
    
    // MARK: - Model Management
    
    /// 检查模型是否已缓存
    public static func isModelCached(variant: MOSSModelVariant = .mossTTSNano100M) async -> Bool {
        let downloader = ModelDownloader()
        return await downloader.isModelCached(variant: variant)
    }
    
    /// 预下载模型
    public static func preload(
        variant: MOSSModelVariant = .mossTTSNano100M,
        progressCallback: (@Sendable (ModelDownloader.DownloadProgress) -> Void)? = nil
    ) async throws {
        let downloader = ModelDownloader()
        _ = try await downloader.downloadModels(variant: variant, progressCallback: progressCallback)
    }
    
    /// 获取缓存大小
    public static func cacheSize() async -> String {
        let downloader = ModelDownloader()
        return await downloader.formattedCacheSize
    }
    
    /// 清理模型缓存
    public static func clearCache(for variant: MOSSModelVariant? = nil) async throws {
        let downloader = ModelDownloader()
        try await downloader.clearCache(for: variant)
    }
    
    // MARK: - Private
    
    /// 简单文本预处理
    private func preprocessText(_ text: String) -> String {
        var processed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        processed = processed.replacingOccurrences(of: "\n", with: " ")
        processed = processed.replacingOccurrences(of: "  ", with: " ")
        return processed
    }
    
    private func resolvedMaxFrames(options: MOSSTTSOptions, manifest: MOSSBrowserManifest) -> Int {
        let optionLimit = options.maxGeneratedFrames ?? options.maxLength
        let manifestLimit = manifest.generationDefaults.maxNewFrames ?? optionLimit
        return max(1, min(optionLimit, manifestLimit, options.maxLength))
    }
    
    private func adaptChannels(_ samples: [Float], from sourceChannels: Int, to targetChannels: Int) -> [Float] {
        guard sourceChannels != targetChannels else {
            return samples
        }
        
        if sourceChannels == 2, targetChannels == 1 {
            var mono: [Float] = []
            mono.reserveCapacity(samples.count / 2)
            for index in stride(from: 0, to: samples.count - 1, by: 2) {
                mono.append((samples[index] + samples[index + 1]) * 0.5)
            }
            return mono
        }
        
        if sourceChannels == 1, targetChannels == 2 {
            return samples.flatMap { [$0, $0] }
        }
        
        return samples
    }
    
    private func saveAsWAV(_ result: TTSResult, to url: URL) throws {
        var header = Data(count: 44)
        
        let sampleRate = UInt32(result.sampleRate)
        let numChannels = UInt16(result.channels)
        let bitsPerSample: UInt16 = 16
        let byteRate = sampleRate * UInt32(numChannels) * UInt32(bitsPerSample / 8)
        let blockAlign = numChannels * (bitsPerSample / 8)
        let dataSize = result.audioSamples.count * MemoryLayout<Int16>.size
        let fileSize = UInt32(36 + dataSize)
        
        // Write WAV header using withUnsafeMutableBytes
        header.withUnsafeMutableBytes { buffer in
            guard let ptr = buffer.baseAddress else { return }
            
            // RIFF
            ptr.storeBytes(of: UInt32(0x46464952).littleEndian, toByteOffset: 0, as: UInt32.self)
            // File size
            ptr.storeBytes(of: fileSize.littleEndian, toByteOffset: 4, as: UInt32.self)
            // WAVE
            ptr.storeBytes(of: UInt32(0x45564157).littleEndian, toByteOffset: 8, as: UInt32.self)
            // fmt 
            ptr.storeBytes(of: UInt32(0x20746D66).littleEndian, toByteOffset: 12, as: UInt32.self)
            // fmt chunk size (16)
            ptr.storeBytes(of: UInt32(16).littleEndian, toByteOffset: 16, as: UInt32.self)
            // Audio format (PCM = 1)
            ptr.storeBytes(of: UInt16(1).littleEndian, toByteOffset: 20, as: UInt16.self)
            // Num channels
            ptr.storeBytes(of: numChannels.littleEndian, toByteOffset: 22, as: UInt16.self)
            // Sample rate
            ptr.storeBytes(of: sampleRate.littleEndian, toByteOffset: 24, as: UInt32.self)
            // Byte rate
            ptr.storeBytes(of: byteRate.littleEndian, toByteOffset: 28, as: UInt32.self)
            // Block align
            ptr.storeBytes(of: blockAlign.littleEndian, toByteOffset: 32, as: UInt16.self)
            // Bits per sample
            ptr.storeBytes(of: bitsPerSample.littleEndian, toByteOffset: 34, as: UInt16.self)
            // data
            ptr.storeBytes(of: UInt32(0x61746164).littleEndian, toByteOffset: 36, as: UInt32.self)
            // Data size
            ptr.storeBytes(of: UInt32(dataSize).littleEndian, toByteOffset: 40, as: UInt32.self)
        }
        
        // Convert float samples to Int16
        var audioData = Data()
        for sample in result.audioSamples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * 32767.0)
            audioData.append(contentsOf: withUnsafeBytes(of: int16.littleEndian) { Array($0) })
        }
        
        let fileData = header + audioData
        try fileData.write(to: url)
    }
}

// MARK: - Supporting Types

/// 合成结果
public struct TTSResult: Sendable {
    /// 音频采样 (Float32, -1.0 ~ 1.0)
    public let audioSamples: [Float]
    
    /// 采样率
    public let sampleRate: Int
    
    /// 声道数
    public let channels: Int
    
    /// 时长 (秒)
    public let duration: Double
    
    /// 元数据
    public let metadata: TTSMetadata
    
    public init(
        audioSamples: [Float],
        sampleRate: Int,
        channels: Int,
        duration: Double,
        metadata: TTSMetadata
    ) {
        self.audioSamples = audioSamples
        self.sampleRate = sampleRate
        self.channels = channels
        self.duration = duration
        self.metadata = metadata
    }
}

/// TTS 元数据
public struct TTSMetadata: Sendable {
    public let text: String
    public let processedText: String
    public let modelVariant: MOSSModelVariant
}

/// 流式 TTS 输出 chunk。
public struct MOSSTTSStreamChunk: Sendable {
    public let audioSamples: [Float]
    public let newAudioSamples: [Float]
    public let currentStep: Int
    public let totalSteps: Int
    public let sampleRate: Int
    public let channels: Int
    public let text: String
    public let processedText: String
    public let modelVariant: MOSSModelVariant
    public let isFinal: Bool
    
    public init(
        audioSamples: [Float],
        newAudioSamples: [Float],
        currentStep: Int,
        totalSteps: Int,
        sampleRate: Int,
        channels: Int,
        text: String,
        processedText: String,
        modelVariant: MOSSModelVariant,
        isFinal: Bool
    ) {
        self.audioSamples = audioSamples
        self.newAudioSamples = newAudioSamples
        self.currentStep = currentStep
        self.totalSteps = totalSteps
        self.sampleRate = sampleRate
        self.channels = channels
        self.text = text
        self.processedText = processedText
        self.modelVariant = modelVariant
        self.isFinal = isFinal
    }
}
