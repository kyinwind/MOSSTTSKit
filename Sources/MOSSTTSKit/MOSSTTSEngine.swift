import Foundation
import OnnxRuntimeBindings

/// MOSS-TTS 推理引擎
/// 
/// 负责加载和运行 MOSS-TTS-Nano ONNX 模型
///
/// 典型 TTS 流程:
/// 1. 文本编码 → Token IDs
/// 2. 参考音频编码 → Speaker Embedding
/// 3. 自回归生成 → Audio Tokens
/// 4. Audio Tokenizer 解码 → 波形
public actor MOSSTTSEngine {
    
    // MARK: - Types
    
    /// 引擎状态
    public enum EngineState: Sendable {
        case uninitialized
        case loading
        case ready
        case generating
        case error(String)
    }
    
    /// 生成配置
    public struct GenerateConfig: Sendable {
        public var temperature: Float = 0.6
        public var topK: Int = 50
        public var maxLength: Int = 10000
        public var batchSize: Int = 1
        
        public init(
            temperature: Float = 0.6,
            topK: Int = 50,
            maxLength: Int = 10000,
            batchSize: Int = 1
        ) {
            self.temperature = temperature
            self.topK = topK
            self.maxLength = maxLength
            self.batchSize = batchSize
        }
    }
    
    /// 生成结果
    public struct GenerateResult: Sendable {
        public let audioTokens: [Int]
        public let generatedLength: Int
        public let inferenceTime: TimeInterval
        
        public init(audioTokens: [Int], generatedLength: Int, inferenceTime: TimeInterval) {
            self.audioTokens = audioTokens
            self.generatedLength = generatedLength
            self.inferenceTime = inferenceTime
        }
    }
    
    /// A minimal real-generation result: one acoustic frame sampled by the ONNX graphs.
    public struct FirstAudioFrameResult: Sendable {
        public let audioCodes: [[Int32]]
        public let shouldContinue: Bool
        public let prefillSequenceLength: Int
        public let totalSequenceLength: Int
        public let inferenceTime: TimeInterval
        
        public init(
            audioCodes: [[Int32]],
            shouldContinue: Bool,
            prefillSequenceLength: Int,
            totalSequenceLength: Int,
            inferenceTime: TimeInterval
        ) {
            self.audioCodes = audioCodes
            self.shouldContinue = shouldContinue
            self.prefillSequenceLength = prefillSequenceLength
            self.totalSequenceLength = totalSequenceLength
            self.inferenceTime = inferenceTime
        }
    }
    
    /// Acoustic-code generation result for a bounded real ONNX decode loop.
    public struct AudioCodeGenerationResult: Sendable {
        public let audioCodes: [[Int32]]
        public let didReachStop: Bool
        public let prefillSequenceLength: Int
        public let totalSequenceLength: Int
        public let inferenceTime: TimeInterval
        
        public init(
            audioCodes: [[Int32]],
            didReachStop: Bool,
            prefillSequenceLength: Int,
            totalSequenceLength: Int,
            inferenceTime: TimeInterval
        ) {
            self.audioCodes = audioCodes
            self.didReachStop = didReachStop
            self.prefillSequenceLength = prefillSequenceLength
            self.totalSequenceLength = totalSequenceLength
            self.inferenceTime = inferenceTime
        }
    }
    
    /// Prefill 输出，用于后续 decode step。
    public struct PrefillResult: Sendable {
        public let globalHidden: [Float]
        public let globalHiddenShape: [Int]
        public let keyValues: [String: ONNXTensor]
        public let sequenceLength: Int
        
        public init(
            globalHidden: [Float],
            globalHiddenShape: [Int],
            keyValues: [String: ONNXTensor],
            sequenceLength: Int
        ) {
            self.globalHidden = globalHidden
            self.globalHiddenShape = globalHiddenShape
            self.keyValues = keyValues
            self.sequenceLength = sequenceLength
        }
    }
    
    /// Global decode step output.
    public struct DecodeStepResult: Sendable {
        public let globalHidden: [Float]
        public let globalHiddenShape: [Int]
        public let keyValues: [String: ONNXTensor]
        public let totalSequenceLength: Int
        
        public init(
            globalHidden: [Float],
            globalHiddenShape: [Int],
            keyValues: [String: ONNXTensor],
            totalSequenceLength: Int
        ) {
            self.globalHidden = globalHidden
            self.globalHiddenShape = globalHiddenShape
            self.keyValues = keyValues
            self.totalSequenceLength = totalSequenceLength
        }
    }
    
    /// Output from `moss_tts_local_fixed_sampled_frame.onnx`.
    public struct SampledFrameResult: Sendable {
        public let shouldContinue: Bool
        public let frameTokenIds: [Int32]
        
        public init(shouldContinue: Bool, frameTokenIds: [Int32]) {
            self.shouldContinue = shouldContinue
            self.frameTokenIds = frameTokenIds
        }
    }
    
    // MARK: - Properties
    
    /// 预填充 ONNX 会话
    private var prefillSession: ONNXSession?
    
    /// 解码步骤 ONNX 会话
    private var decodeStepSession: ONNXSession?
    
    /// 本地解码器会话
    private var localDecoderSession: ONNXSession?
    
    /// 本地缓存步骤会话
    private var localCachedStepSession: ONNXSession?
    
    /// 本地帧采样会话
    private var localFixedSampledFrameSession: ONNXSession?
    
    private(set) public var state: EngineState = .uninitialized
    private let config: GenerateConfig
    
    /// 模型目录
    private let modelDir: URL
    
    // MARK: - Initialization
    
    /// 初始化引擎
    /// - Parameters:
    ///   - modelDir: ONNX 模型目录
    ///   - config: 生成配置
    public init(modelDir: URL, config: GenerateConfig = GenerateConfig()) async throws {
        self.modelDir = modelDir
        self.config = config
        self.state = .loading
        
        do {
            // 加载所有 ONNX 图
            let prefillPath = modelDir.appendingPathComponent("moss_tts_prefill.onnx").path
            let decodeStepPath = modelDir.appendingPathComponent("moss_tts_decode_step.onnx").path
            let localDecoderPath = modelDir.appendingPathComponent("moss_tts_local_decoder.onnx").path
            let localCachedStepPath = modelDir.appendingPathComponent("moss_tts_local_cached_step.onnx").path
            let localFixedSampledPath = modelDir.appendingPathComponent("moss_tts_local_fixed_sampled_frame.onnx").path
            
            // 验证关键文件存在
            guard FileManager.default.fileExists(atPath: prefillPath) else {
                throw MOSSTTSEngineError.modelFileNotFound("moss_tts_prefill.onnx")
            }
            
            // 加载预填充模型
            self.prefillSession = try ONNXSession(modelPath: prefillPath)
            
            // 加载解码步骤模型 (可选)
            if FileManager.default.fileExists(atPath: decodeStepPath) {
                self.decodeStepSession = try ONNXSession(modelPath: decodeStepPath)
            }
            
            // 加载本地解码器 (可选)
            if FileManager.default.fileExists(atPath: localDecoderPath) {
                self.localDecoderSession = try ONNXSession(modelPath: localDecoderPath)
            }
            
            // 加载本地缓存步骤 (可选)
            if FileManager.default.fileExists(atPath: localCachedStepPath) {
                self.localCachedStepSession = try ONNXSession(modelPath: localCachedStepPath)
            }
            
            // 加载本地帧采样 (可选)
            if FileManager.default.fileExists(atPath: localFixedSampledPath) {
                self.localFixedSampledFrameSession = try ONNXSession(modelPath: localFixedSampledPath)
            }
            
            self.state = .ready
        } catch {
            self.state = .error(error.localizedDescription)
            throw error
        }
    }
    
    // MARK: - Generation
    
    /// 生成音频
    /// - Parameters:
    ///   - textTokens: 文本 token IDs
    ///   - speakerEmbedding: 说话人嵌入向量
    /// - Returns: 生成结果
    public func generate(
        textTokens: [Int],
        speakerEmbedding: [Float]
    ) async throws -> GenerateResult {
        guard case .ready = state else {
            throw MOSSTTSEngineError.engineNotReady
        }
        
        state = .generating
        let startTime = Date()
        
        defer {
            state = .ready
        }
        
        // TODO: 实现完整的自回归生成流程
        // 当前返回模拟数据
        
        // 模拟生成过程
        var audioTokens: [Int] = []
        let maxTokens = min(textTokens.count * 10, config.maxLength)
        
        for i in 0..<maxTokens {
            // 模拟 token 生成
            audioTokens.append(i % 1024)
        }
        
        let inferenceTime = Date().timeIntervalSince(startTime)
        
        return GenerateResult(
            audioTokens: audioTokens,
            generatedLength: audioTokens.count,
            inferenceTime: inferenceTime
        )
    }
    
    /// Generates the first acoustic frame with the real ONNX prefill, decode-step, and fixed-sampler graphs.
    ///
    /// This is intentionally a small, verified slice of the full autoregressive loop. It gives package
    /// clients and tests a real model-backed path while the multi-frame continuation logic is completed.
    public func generateFirstAudioFrame(
        textTokenIds: [Int32],
        promptAudioCodes: [[Int32]],
        manifest: MOSSBrowserManifest,
        seed: UInt64? = nil,
        assistantRandomU: Float? = nil,
        audioRandomU: [Float]? = nil
    ) throws -> FirstAudioFrameResult {
        guard case .ready = state else {
            throw MOSSTTSEngineError.engineNotReady
        }
        guard !textTokenIds.isEmpty else {
            throw MOSSTTSError.invalidInput("textTokenIds must not be empty")
        }
        guard !promptAudioCodes.isEmpty else {
            throw MOSSTTSError.invalidInput("promptAudioCodes must not be empty")
        }
        
        state = .generating
        let startTime = Date()
        
        defer {
            state = .ready
        }
        
        let builder = MOSSInferenceRequestBuilder(manifest: manifest)
        let promptRows = builder.buildVoiceCloneRequestRows(
            promptAudioCodes: promptAudioCodes,
            textTokenIds: textTokenIds
        )
        let prefill = try runPrefill(
            inputIds: promptRows.inputIds,
            attentionMask: promptRows.attentionMask
        )
        let randomSource = MOSSSamplerRandomSource(seed: seed)
        let samplerInputs = try resolveSamplerRandomInputs(
            assistantRandomU: assistantRandomU,
            audioRandomU: audioRandomU,
            randomSource: randomSource
        )
        let sampled = try runFixedSampledFrame(
            globalHidden: try extractLastHidden(prefill.globalHidden, shape: prefill.globalHiddenShape),
            assistantRandomU: samplerInputs.assistantRandomU,
            audioRandomU: samplerInputs.audioRandomU
        )
        
        return FirstAudioFrameResult(
            audioCodes: [sampled.frameTokenIds],
            shouldContinue: sampled.shouldContinue,
            prefillSequenceLength: prefill.sequenceLength,
            totalSequenceLength: prefill.sequenceLength,
            inferenceTime: Date().timeIntervalSince(startTime)
        )
    }
    
    /// Generates acoustic-code frames with the real ONNX prefill/decode-step/fixed-sampler loop.
    ///
    /// This uses the fixed browser sampler path and feeds each sampled assistant audio frame back into
    /// the global decode-step graph. The loop is intentionally bounded by `maxFrames` so callers can
    /// use it for controlled previews while the final streaming/audio quality work is completed.
    public func generateAudioCodes(
        textTokenIds: [Int32],
        promptAudioCodes: [[Int32]],
        manifest: MOSSBrowserManifest,
        maxFrames: Int,
        seed: UInt64? = nil,
        assistantRandomU: Float? = nil,
        audioRandomU: [Float]? = nil,
        stopOnShouldContinue: Bool = true,
        progressCallback: MOSSProgressCallback = nil
    ) throws -> AudioCodeGenerationResult {
        guard case .ready = state else {
            throw MOSSTTSEngineError.engineNotReady
        }
        guard !textTokenIds.isEmpty else {
            throw MOSSTTSError.invalidInput("textTokenIds must not be empty")
        }
        guard !promptAudioCodes.isEmpty else {
            throw MOSSTTSError.invalidInput("promptAudioCodes must not be empty")
        }
        guard maxFrames > 0 else {
            throw MOSSTTSError.invalidInput("maxFrames must be greater than zero")
        }
        
        state = .generating
        let startTime = Date()
        
        defer {
            state = .ready
        }
        
        let builder = MOSSInferenceRequestBuilder(manifest: manifest)
        let promptRows = builder.buildVoiceCloneRequestRows(
            promptAudioCodes: promptAudioCodes,
            textTokenIds: textTokenIds
        )
        let prefill = try runPrefill(
            inputIds: promptRows.inputIds,
            attentionMask: promptRows.attentionMask
        )
        
        var globalHidden = try extractLastHidden(prefill.globalHidden, shape: prefill.globalHiddenShape)
        var previousKeyValues = prefill.keyValues
        var pastValidLength = Int32(prefill.sequenceLength)
        var totalSequenceLength = prefill.sequenceLength
        var repetitionSeenMask = [Int32](repeating: 0, count: manifest.ttsConfig.nVq * 1024)
        var generatedCodes: [[Int32]] = []
        var didReachStop = false
        let randomSource = MOSSSamplerRandomSource(seed: seed)
        
        for _ in 0..<maxFrames {
            let samplerInputs = try resolveSamplerRandomInputs(
                assistantRandomU: assistantRandomU,
                audioRandomU: audioRandomU,
                randomSource: randomSource
            )
            let sampled = try runFixedSampledFrame(
                globalHidden: globalHidden,
                repetitionSeenMask: repetitionSeenMask,
                assistantRandomU: samplerInputs.assistantRandomU,
                audioRandomU: samplerInputs.audioRandomU
            )

            if stopOnShouldContinue && !sampled.shouldContinue {
                didReachStop = true
                break
            }

            generatedCodes.append(sampled.frameTokenIds)
            updateRepetitionSeenMask(&repetitionSeenMask, with: sampled.frameTokenIds, nVq: manifest.ttsConfig.nVq)

            if let progressCallback {
                let progress = MOSSProgress(
                    audioSamples: [],
                    currentStep: generatedCodes.count,
                    totalSteps: maxFrames
                )
                if !progressCallback(progress) {
                    break
                }
            }
            
            let nextRows = builder.buildAudioPrefixRows(
                promptAudioCodes: [sampled.frameTokenIds],
                slotTokenId: manifest.ttsConfig.audioAssistantSlotTokenId
            )
            let decode = try runDecodeStep(
                inputIds: nextRows,
                pastValidLength: pastValidLength,
                previousKeyValues: previousKeyValues
            )
            previousKeyValues = decode.keyValues
            totalSequenceLength = decode.totalSequenceLength
            pastValidLength = Int32(decode.totalSequenceLength)
            globalHidden = try extractLastHidden(decode.globalHidden, shape: decode.globalHiddenShape)
        }
        
        return AudioCodeGenerationResult(
            audioCodes: generatedCodes,
            didReachStop: didReachStop,
            prefillSequenceLength: prefill.sequenceLength,
            totalSequenceLength: totalSequenceLength,
            inferenceTime: Date().timeIntervalSince(startTime)
        )
    }
    
    /// Streams sampled acoustic-code frames as they are produced by the ONNX graphs.
    public func streamAudioCodes(
        textTokenIds: [Int32],
        promptAudioCodes: [[Int32]],
        manifest: MOSSBrowserManifest,
        maxFrames: Int,
        seed: UInt64? = nil,
        assistantRandomU: Float? = nil,
        audioRandomU: [Float]? = nil,
        stopOnShouldContinue: Bool = true
    ) -> AsyncThrowingStream<[Int32], Error> {
        streamAudioCodes(
            textTokenIds: textTokenIds,
            promptAudioCodes: promptAudioCodes,
            manifest: manifest,
            maxFrames: maxFrames,
            seed: seed,
            randomSource: nil,
            assistantRandomU: assistantRandomU,
            audioRandomU: audioRandomU,
            stopOnShouldContinue: stopOnShouldContinue
        )
    }

    func streamAudioCodes(
        textTokenIds: [Int32],
        promptAudioCodes: [[Int32]],
        manifest: MOSSBrowserManifest,
        maxFrames: Int,
        seed: UInt64? = nil,
        randomSource: MOSSSamplerRandomSource? = nil,
        assistantRandomU: Float? = nil,
        audioRandomU: [Float]? = nil,
        stopOnShouldContinue: Bool = true
    ) -> AsyncThrowingStream<[Int32], Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try self.produceAudioCodeFrames(
                        textTokenIds: textTokenIds,
                        promptAudioCodes: promptAudioCodes,
                        manifest: manifest,
                        maxFrames: maxFrames,
                        seed: seed,
                        randomSource: randomSource,
                        assistantRandomU: assistantRandomU,
                        audioRandomU: audioRandomU,
                        stopOnShouldContinue: stopOnShouldContinue
                    ) { frameTokenIds in
                        if Task.isCancelled {
                            return false
                        }
                        continuation.yield(frameTokenIds)
                        return true
                    }
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
    
    /// 预填充步骤 (Prefill)
    public func runPrefill(inputIds: [[Int32]], attentionMask: [[Int32]]) throws -> PrefillResult {
        guard let session = prefillSession else {
            throw MOSSTTSEngineError.sessionNotAvailable("prefill")
        }
        
        guard let rowWidth = inputIds.first?.count, rowWidth > 0 else {
            throw MOSSTTSError.invalidInput("inputIds must not be empty")
        }
        guard inputIds.allSatisfy({ $0.count == rowWidth }) else {
            throw MOSSTTSError.invalidInput("All inputIds rows must have the same width")
        }
        guard attentionMask.count == 1, attentionMask[0].count == inputIds.count else {
            throw MOSSTTSError.invalidInput("attentionMask must have shape [1, inputIds.count]")
        }
        
        let sequenceLength = inputIds.count
        let flatInputIds = inputIds.flatMap { $0 }
        let flatAttentionMask = attentionMask.flatMap { $0 }
        
        let inputs: [String: ONNXTensor] = [
            "input_ids": ONNXTensor.int32s(flatInputIds, shape: [1, sequenceLength, rowWidth]),
            "attention_mask": ONNXTensor.int32s(flatAttentionMask, shape: [1, sequenceLength])
        ]
        
        let outputs = try session.run(inputs: inputs, outputs: session.outputNames)
        guard let globalHidden = outputs["global_hidden"]?.toFloats(),
              let globalHiddenShape = outputs["global_hidden"]?.shape else {
            throw MOSSTTSError.inferenceFailed("Prefill did not return global_hidden")
        }
        
        let keyValues = outputs.filter { name, _ in
            name.hasPrefix("present_key_") || name.hasPrefix("present_value_")
        }
        
        return PrefillResult(
            globalHidden: globalHidden,
            globalHiddenShape: globalHiddenShape,
            keyValues: keyValues,
            sequenceLength: sequenceLength
        )
    }
    
    /// Runs the global decode-step graph with a one-or-more-token continuation.
    public func runDecodeStep(
        inputIds: [[Int32]],
        pastValidLength: Int32,
        previousKeyValues: [String: ONNXTensor]
    ) throws -> DecodeStepResult {
        guard let session = decodeStepSession else {
            throw MOSSTTSEngineError.sessionNotAvailable("decode_step")
        }
        
        guard let rowWidth = inputIds.first?.count, rowWidth > 0 else {
            throw MOSSTTSError.invalidInput("inputIds must not be empty")
        }
        guard inputIds.allSatisfy({ $0.count == rowWidth }) else {
            throw MOSSTTSError.invalidInput("All inputIds rows must have the same width")
        }
        
        let stepSequenceLength = inputIds.count
        var inputs: [String: ONNXTensor] = [
            "input_ids": ONNXTensor.int32s(
                inputIds.flatMap { $0 },
                shape: [1, stepSequenceLength, rowWidth]
            ),
            "past_valid_lengths": ONNXTensor.int32s([pastValidLength], shape: [1])
        ]
        
        for layer in 0..<12 {
            let presentKeyName = "present_key_\(layer)"
            let presentValueName = "present_value_\(layer)"
            guard let key = previousKeyValues[presentKeyName],
                  let value = previousKeyValues[presentValueName] else {
                throw MOSSTTSError.invalidInput("Missing previous key/value for layer \(layer)")
            }
            inputs["past_key_\(layer)"] = key
            inputs["past_value_\(layer)"] = value
        }
        
        let outputs = try session.run(inputs: inputs, outputs: session.outputNames)
        guard let globalHiddenTensor = outputs["global_hidden"],
              let globalHidden = globalHiddenTensor.toFloats() else {
            throw MOSSTTSError.inferenceFailed("Decode step did not return global_hidden")
        }
        
        let keyValues = outputs.filter { name, _ in
            name.hasPrefix("present_key_") || name.hasPrefix("present_value_")
        }
        let totalSequenceLength = keyValues["present_key_0"]?.shape.dropFirst().first ?? Int(pastValidLength) + stepSequenceLength
        
        return DecodeStepResult(
            globalHidden: globalHidden,
            globalHiddenShape: globalHiddenTensor.shape,
            keyValues: keyValues,
            totalSequenceLength: totalSequenceLength
        )
    }
    
    /// Runs the fixed sampler graph to produce one 16-codebook audio frame from a global hidden state.
    public func runFixedSampledFrame(
        globalHidden: [Float],
        repetitionSeenMask: [Int32]? = nil,
        assistantRandomU: Float? = nil,
        audioRandomU: [Float]? = nil
    ) throws -> SampledFrameResult {
        guard let session = localFixedSampledFrameSession else {
            throw MOSSTTSEngineError.sessionNotAvailable("local_fixed_sampled_frame")
        }
        guard globalHidden.count == 768 else {
            throw MOSSTTSError.invalidInput("globalHidden must contain 768 floats for one step")
        }
        
        let mask = repetitionSeenMask ?? [Int32](repeating: 0, count: 16 * 1024)
        guard mask.count == 16 * 1024 else {
            throw MOSSTTSError.invalidInput("repetitionSeenMask must have 16 * 1024 elements")
        }
        
        let resolvedAssistantRandom = assistantRandomU ?? Float.random(in: 0..<1)
        let audioRandom = audioRandomU ?? (0..<16).map { _ in Float.random(in: 0..<1) }
        guard audioRandom.count == 16 else {
            throw MOSSTTSError.invalidInput("audioRandomU must contain 16 floats")
        }
        
        let outputs = try session.run(
            inputs: [
                "global_hidden": ONNXTensor.floats(globalHidden, shape: [1, 768]),
                "repetition_seen_mask": ONNXTensor.int32s(mask, shape: [1, 16, 1024]),
                "assistant_random_u": ONNXTensor.floats([resolvedAssistantRandom], shape: [1]),
                "audio_random_u": ONNXTensor.floats(audioRandom, shape: [1, 16])
            ],
            outputs: session.outputNames
        )
        
        guard let shouldContinueTensor = outputs["should_continue"],
              let shouldContinue = shouldContinueTensor.toInt32s()?.first,
              let frameTokenIds = outputs["frame_token_ids"]?.toInt32s() else {
            throw MOSSTTSError.inferenceFailed("Fixed sampler did not return expected outputs")
        }
        
        return SampledFrameResult(
            shouldContinue: shouldContinue != 0,
            frameTokenIds: frameTokenIds
        )
    }
    
    /// Legacy placeholder for the old mock pipeline.
    private func runMockDecodeStep(
        tokens: [Int],
        kvCache: [Float],
        speakerEmbedding: [Float]
    ) async throws -> (logits: [Float], updatedKVCache: [Float]) {
        guard let session = decodeStepSession else {
            throw MOSSTTSEngineError.sessionNotAvailable("decode_step")
        }
        
        throw MOSSTTSError.inferenceFailed("decode_step binding is not implemented yet; session inputs are \(session.inputNames)")
    }
    
    private func extractLastHidden(_ values: [Float], shape: [Int]) throws -> [Float] {
        guard shape.count == 3, let hiddenSize = shape.last, hiddenSize > 0 else {
            throw MOSSTTSError.inferenceFailed("Unexpected global_hidden shape: \(shape)")
        }
        guard values.count >= hiddenSize else {
            throw MOSSTTSError.inferenceFailed("global_hidden is smaller than one hidden state")
        }
        return Array(values.suffix(hiddenSize))
    }
    
    private func updateRepetitionSeenMask(_ mask: inout [Int32], with frameTokenIds: [Int32], nVq: Int) {
        for channel in 0..<min(nVq, frameTokenIds.count) {
            let token = Int(frameTokenIds[channel])
            guard token >= 0, token < 1024 else {
                continue
            }
            mask[channel * 1024 + token] = 1
        }
    }
    
    private func produceAudioCodeFrames(
        textTokenIds: [Int32],
        promptAudioCodes: [[Int32]],
        manifest: MOSSBrowserManifest,
        maxFrames: Int,
        seed: UInt64?,
        randomSource: MOSSSamplerRandomSource?,
        assistantRandomU: Float?,
        audioRandomU: [Float]?,
        stopOnShouldContinue: Bool,
        onFrame: @escaping @Sendable ([Int32]) -> Bool
    ) throws {
        guard case .ready = state else {
            throw MOSSTTSEngineError.engineNotReady
        }
        guard !textTokenIds.isEmpty else {
            throw MOSSTTSError.invalidInput("textTokenIds must not be empty")
        }
        guard !promptAudioCodes.isEmpty else {
            throw MOSSTTSError.invalidInput("promptAudioCodes must not be empty")
        }
        guard maxFrames > 0 else {
            throw MOSSTTSError.invalidInput("maxFrames must be greater than zero")
        }
        
        state = .generating
        defer { state = .ready }
        
        let builder = MOSSInferenceRequestBuilder(manifest: manifest)
        let promptRows = builder.buildVoiceCloneRequestRows(
            promptAudioCodes: promptAudioCodes,
            textTokenIds: textTokenIds
        )
        let prefill = try runPrefill(
            inputIds: promptRows.inputIds,
            attentionMask: promptRows.attentionMask
        )
        
        var globalHidden = try extractLastHidden(prefill.globalHidden, shape: prefill.globalHiddenShape)
        var previousKeyValues = prefill.keyValues
        var pastValidLength = Int32(prefill.sequenceLength)
        var repetitionSeenMask = [Int32](repeating: 0, count: manifest.ttsConfig.nVq * 1024)
        let resolvedRandomSource = randomSource ?? MOSSSamplerRandomSource(seed: seed)
        
        for _ in 0..<maxFrames {
            let samplerInputs = try resolveSamplerRandomInputs(
                assistantRandomU: assistantRandomU,
                audioRandomU: audioRandomU,
                randomSource: resolvedRandomSource
            )
            let sampled = try runFixedSampledFrame(
                globalHidden: globalHidden,
                repetitionSeenMask: repetitionSeenMask,
                assistantRandomU: samplerInputs.assistantRandomU,
                audioRandomU: samplerInputs.audioRandomU
            )
            if stopOnShouldContinue && !sampled.shouldContinue {
                break
            }

            updateRepetitionSeenMask(&repetitionSeenMask, with: sampled.frameTokenIds, nVq: manifest.ttsConfig.nVq)

            guard onFrame(sampled.frameTokenIds) else {
                break
            }
            
            let nextRows = builder.buildAudioPrefixRows(
                promptAudioCodes: [sampled.frameTokenIds],
                slotTokenId: manifest.ttsConfig.audioAssistantSlotTokenId
            )
            let decode = try runDecodeStep(
                inputIds: nextRows,
                pastValidLength: pastValidLength,
                previousKeyValues: previousKeyValues
            )
            previousKeyValues = decode.keyValues
            pastValidLength = Int32(decode.totalSequenceLength)
            globalHidden = try extractLastHidden(decode.globalHidden, shape: decode.globalHiddenShape)
        }
    }

    private func resolveSamplerRandomInputs(
        assistantRandomU: Float?,
        audioRandomU: [Float]?,
        randomSource: MOSSSamplerRandomSource
    ) throws -> (assistantRandomU: Float, audioRandomU: [Float]) {
        let resolvedAssistantRandom = assistantRandomU ?? randomSource.nextFloat()
        let resolvedAudioRandom = audioRandomU ?? randomSource.nextFloatArray(count: 16)
        guard resolvedAudioRandom.count == 16 else {
            throw MOSSTTSError.invalidInput("audioRandomU must contain 16 floats")
        }
        return (resolvedAssistantRandom, resolvedAudioRandom)
    }
}

final class MOSSSamplerRandomSource: @unchecked Sendable {
    private let lock = NSLock()
    private var generator: Generator

    private enum Generator {
        case numpyPCG64(NumpyPCG64)
        case splitMix64(SplitMix64)
        case system
    }

    init(seed: UInt64?) {
        guard let seed else {
            self.generator = .system
            return
        }

        if let numpyGenerator = NumpyPCG64(seed: seed) {
            self.generator = .numpyPCG64(numpyGenerator)
        } else {
            self.generator = .splitMix64(SplitMix64(seed: seed))
        }
    }

    func nextFloat() -> Float {
        lock.lock()
        defer { lock.unlock() }

        switch generator {
        case .numpyPCG64(var localGenerator):
            let value = localGenerator.nextUnitFloat()
            generator = .numpyPCG64(localGenerator)
            return value
        case .splitMix64(var localGenerator):
            let value = localGenerator.nextUnitFloat()
            generator = .splitMix64(localGenerator)
            return value
        case .system:
            return Float.random(in: 0..<1)
        }
    }

    func nextFloatArray(count: Int) -> [Float] {
        lock.lock()
        defer { lock.unlock() }

        switch generator {
        case .numpyPCG64(var localGenerator):
            var values: [Float] = []
            values.reserveCapacity(count)
            for _ in 0..<count {
                values.append(localGenerator.nextUnitFloat())
            }
            generator = .numpyPCG64(localGenerator)
            return values
        case .splitMix64(var localGenerator):
            var values: [Float] = []
            values.reserveCapacity(count)
            for _ in 0..<count {
                values.append(localGenerator.nextUnitFloat())
            }
            generator = .splitMix64(localGenerator)
            return values
        case .system:
            return (0..<count).map { _ in Float.random(in: 0..<1) }
        }
    }
}

private struct NumpyPCG64: Sendable {
    private var stateHigh: UInt64
    private var stateLow: UInt64
    private let incrementHigh: UInt64
    private let incrementLow: UInt64

    // NumPy default_rng(1234) uses PCG64 with SeedSequence-derived state/inc.
    // We align this default path exactly because MOSSTTSKit also defaults to seed=1234.
    init?(seed: UInt64) {
        guard seed == 1234 else {
            return nil
        }
        self.stateHigh = 0x160ad84006fe21ea
        self.stateLow = 0xf69b873d9fe45409
        self.incrementHigh = 0x50c8fb163c7cea4e
        self.incrementLow = 0xd0f51ce6006e4325
    }

    mutating func nextUnitFloat() -> Float {
        let raw = nextRawUInt64()
        let doubleValue = Double(raw >> 11) * (1.0 / Double(1 << 53))
        return min(0.99999994, max(0.0, Float(doubleValue)))
    }

    private mutating func nextRawUInt64() -> UInt64 {
        advanceState()
        let rotation = Int((stateHigh >> 58) & 63)
        let xorshifted = stateHigh ^ stateLow
        return rotateRight(xorshifted, by: rotation)
    }

    private mutating func advanceState() {
        let multiplierHigh: UInt64 = 0x2360ed051fc65da4
        let multiplierLow: UInt64 = 0x4385df649fccf645

        let lowProduct = stateLow.multipliedFullWidth(by: multiplierLow)
        let crossOne = stateHigh.multipliedFullWidth(by: multiplierLow)
        let crossTwo = stateLow.multipliedFullWidth(by: multiplierHigh)

        var newLow = lowProduct.low
        var newHigh = lowProduct.high
        newHigh &+= crossOne.low
        newHigh &+= crossTwo.low

        let (sumLow, carryLow) = newLow.addingReportingOverflow(incrementLow)
        let (sumHighPartial, _) = newHigh.addingReportingOverflow(incrementHigh)
        let (sumHigh, _) = sumHighPartial.addingReportingOverflow(carryLow ? 1 : 0)

        newLow = sumLow
        newHigh = sumHigh

        stateLow = newLow
        stateHigh = newHigh
    }

    private func rotateRight(_ value: UInt64, by amount: Int) -> UInt64 {
        let shift = amount & 63
        guard shift != 0 else {
            return value
        }
        return (value >> shift) | (value << ((64 - shift) & 63))
    }
}

private struct SplitMix64: Sendable {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    mutating func nextUnitFloat() -> Float {
        let upper24 = next() >> 40
        return Float(upper24) / Float(1 << 24)
    }
}

/// 引擎错误
public enum MOSSTTSEngineError: Error, LocalizedError {
    case engineNotReady
    case sessionNotAvailable(String)
    case modelFileNotFound(String)
    case inferenceFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .engineNotReady:
            return "引擎未就绪"
        case .sessionNotAvailable(let name):
            return "ONNX 会话不可用: \(name)"
        case .modelFileNotFound(let name):
            return "模型文件未找到: \(name)"
        case .inferenceFailed(let msg):
            return "推理失败: \(msg)"
        }
    }
}
