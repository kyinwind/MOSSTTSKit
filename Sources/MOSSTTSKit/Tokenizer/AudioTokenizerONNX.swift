/// AudioTokenizerONNX.swift
/// 
/// MOSS Audio Tokenizer 的 ONNX 实现
/// 
/// 模型文件：
/// - moss_audio_tokenizer_encode.onnx: 编码器
/// - moss_audio_tokenizer_decode_full.onnx: 完整解码器
/// - moss_audio_tokenizer_decode_step.onnx: 流式解码器

import Foundation
import AVFoundation

/// ONNX 音频 tokenizer 实现
public final class AudioTokenizerONNX: @unchecked Sendable, MOSSAudioTokenizer {
    private var encodeSession: ONNXSession?
    private var decodeSession: ONNXSession?
    private var decodeStepSession: ONNXSession?
    
    /// 编码器模型路径
    public let encodeModelPath: String
    
    /// 解码器模型路径
    public let decodeModelPath: String

    /// 流式解码器模型路径
    public let decodeStepModelPath: String?
    
    /// 配置
    public let config: AudioTokenizerConfig
    
    /// 采样率
    public var sampleRate: Int { config.sampleRate }
    
    /// 通道数
    public var numChannels: Int { config.numChannels }
    
    /// 初始化
    /// - Parameters:
    ///   - encodeModelPath: 编码器 ONNX 模型路径
    ///   - decodeModelPath: 解码器 ONNX 模型路径
    ///   - config: 配置
    public init(
        encodeModelPath: String,
        decodeModelPath: String,
        decodeStepModelPath: String? = nil,
        config: AudioTokenizerConfig = AudioTokenizerConfig()
    ) {
        self.encodeModelPath = encodeModelPath
        self.decodeModelPath = decodeModelPath
        self.decodeStepModelPath = decodeStepModelPath
        self.config = config
    }
    
    /// 从模型目录加载
    /// - Parameter modelDir: 模型目录路径
    /// - Returns: 音频 tokenizer 实例
    public static func fromDirectory(
        _ modelDir: String
    ) async throws -> AudioTokenizerONNX {
        let encodePath = (modelDir as NSString).appendingPathComponent("moss_audio_tokenizer_encode.onnx")
        let decodePath = (modelDir as NSString).appendingPathComponent("moss_audio_tokenizer_decode_full.onnx")
        let decodeStepPath = (modelDir as NSString).appendingPathComponent("moss_audio_tokenizer_decode_step.onnx")
        
        guard FileManager.default.fileExists(atPath: encodePath) else {
            throw MOSSTTSError.modelNotFound("Encode model not found: \(encodePath)")
        }
        
        guard FileManager.default.fileExists(atPath: decodePath) else {
            throw MOSSTTSError.modelNotFound("Decode model not found: \(decodePath)")
        }
        
        let tokenizer = AudioTokenizerONNX(
            encodeModelPath: encodePath,
            decodeModelPath: decodePath,
            decodeStepModelPath: FileManager.default.fileExists(atPath: decodeStepPath) ? decodeStepPath : nil
        )
        
        try await tokenizer.load()
        return tokenizer
    }
    
    /// 加载模型
    public func load() async throws {
        encodeSession = try ONNXSession(modelPath: encodeModelPath)
        decodeSession = try ONNXSession(modelPath: decodeModelPath)
        if let decodeStepModelPath, FileManager.default.fileExists(atPath: decodeStepModelPath) {
            decodeStepSession = try ONNXSession(modelPath: decodeStepModelPath)
        }
    }
    
    public func encode(audioPath: String) async throws -> AudioEncodingResult {
        // 1. 加载并预处理音频
        let (samples, originalSR) = try await loadAudio(from: audioPath)
        
        // 2. 重采样到目标采样率
        let resampledSamples = resample(
            samples: samples,
            from: originalSR,
            to: config.sampleRate
        )
        
        // 3. 编码
        return try await encode(samples: resampledSamples, sampleRate: config.sampleRate)
    }
    
    public func encode(samples: [Float], sampleRate: Int) async throws -> AudioEncodingResult {
        guard let session = encodeSession else {
            throw MOSSTTSError.inferenceFailed("Encoder not loaded")
        }
        
        let inputSamples = samples.map { Float($0) }
        let frameShift = config.downsampleRate
        let numFrames = inputSamples.count / frameShift
        
        // 填充到完整帧
        let paddedLength = numFrames * frameShift
        let paddedSamples = Array(inputSamples.prefix(paddedLength))
        guard !paddedSamples.isEmpty else {
            throw MOSSTTSError.invalidInput("Audio is too short for one tokenizer frame")
        }
        
        let waveform = makeStereoChannelMajorSamples(fromMono: paddedSamples)
        
        let waveformTensor = ONNXTensor.floats(
            waveform,
            shape: [1, config.numChannels, paddedLength]
        )
        let inputLengthsTensor = ONNXTensor.int32s(
            [Int32(paddedLength)],
            shape: [1]
        )
        
        let waveformName = session.inputNames.first ?? "waveform"
        let inputLengthsName = session.inputNames.dropFirst().first ?? "input_lengths"
        let audioCodesName = session.outputNames.first ?? "audio_codes"
        
        // 执行推理
        let outputs = try session.run(
            inputs: [
                waveformName: waveformTensor,
                inputLengthsName: inputLengthsTensor
            ],
            outputs: session.outputNames
        )
        
        // 解析输出
        guard let outputTensor = outputs[audioCodesName],
              let outputData = outputTensor.toInt32s() else {
            throw MOSSTTSError.inferenceFailed("Failed to get encoder output")
        }
        
        // 转换输出为 RVQ codes
        // 输出格式: [num_frames, num_codebooks]
        var codes: [[Int32]] = []
        for frameIdx in 0..<numFrames {
            var frameCodes: [Int32] = []
            for codebookIdx in 0..<config.numCodebooks {
                let offset = frameIdx * config.numCodebooks + codebookIdx
                if offset < outputData.count {
                    frameCodes.append(outputData[offset])
                } else {
                    frameCodes.append(0)
                }
            }
            codes.append(frameCodes)
        }
        
        let duration = Double(samples.count) / Double(sampleRate)
        return AudioEncodingResult(
            codes: codes,
            sampleRate: sampleRate,
            duration: duration
        )
    }
    
    public func decode(codes: [[Int32]]) async throws -> [Float] {
        guard let session = decodeSession else {
            throw MOSSTTSError.inferenceFailed("Decoder not loaded")
        }
        
        let numFrames = codes.count
        guard numFrames > 0 else {
            return []
        }
        
        // 展平 codes 为 [num_frames * num_codebooks]
        var flatCodes: [Int32] = []
        for frame in codes {
            flatCodes.append(contentsOf: frame)
        }
        
        let inputTensor = ONNXTensor.int32s(
            flatCodes,
            shape: [1, numFrames, config.numCodebooks]
        )
        let lengthsTensor = ONNXTensor.int32s(
            [Int32(numFrames)],
            shape: [1]
        )
        
        let audioCodesName = session.inputNames.first ?? "audio_codes"
        let audioCodeLengthsName = session.inputNames.dropFirst().first ?? "audio_code_lengths"
        let audioOutputName = session.outputNames.first ?? "audio"
        
        // 执行推理
        let outputs = try session.run(
            inputs: [
                audioCodesName: inputTensor,
                audioCodeLengthsName: lengthsTensor
            ],
            outputs: session.outputNames
        )
        
        // 解析输出
        guard let outputTensor = outputs[audioOutputName],
              let outputData = outputTensor.toFloats() else {
            throw MOSSTTSError.inferenceFailed("Failed to get decoder output")
        }

        let audioLengthsName = session.outputNames.dropFirst().first ?? "audio_lengths"
        let validFrameCount = outputs[audioLengthsName]?.toInt32s()?.first.map(Int.init)

        return interleaveChannelMajorAudioSamples(
            Array(outputData),
            shape: outputTensor.shape,
            validFrames: validFrameCount
        )
    }
    
    public func decode(codes: [[Int32]], to outputPath: String) async throws {
        let samples = try await decode(codes: codes)
        try await saveAudio(samples: samples, to: outputPath)
    }

    func decodeIncrementally(
        codes: [[Int32]],
        framesPerChunk: Int = 8
    ) async throws -> [[Float]] {
        guard !codes.isEmpty else { return [] }

        if let decodeStepSession {
            let streamingSession = CodecStreamingDecodeSession(session: decodeStepSession, config: config)
            defer { streamingSession.reset() }

            var decodedChunks: [[Float]] = []
            var startIndex = 0
            while startIndex < codes.count {
                let endIndex = min(startIndex + max(1, framesPerChunk), codes.count)
                let frameChunk = Array(codes[startIndex..<endIndex])
                if let samples = try streamingSession.runFrames(frameChunk), !samples.isEmpty {
                    decodedChunks.append(samples)
                }
                startIndex = endIndex
            }
            return decodedChunks
        }

        let samples = try await decode(codes: codes)
        return [samples]
    }

    func makeIncrementalDecoder(framesPerChunk: Int = 8) -> IncrementalAudioDecoder {
        IncrementalAudioDecoder(
            tokenizer: self,
            decodeStepSession: decodeStepSession,
            config: config,
            framesPerChunk: framesPerChunk
        )
    }
    
    // MARK: - 私有方法
    
    /// 加载音频文件
    private func loadAudio(from path: String) async throws -> ([Float], Int) {
        let url = URL(fileURLWithPath: path)
        let audioFile = try AVAudioFile(forReading: url)
        
        let format = audioFile.processingFormat
        let frameCount = UInt32(audioFile.length)
        
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw MOSSTTSError.audioProcessingFailed("Failed to create audio buffer")
        }
        
        try audioFile.read(into: buffer)
        
        // 提取 Float 数据
        var samples: [Float] = []
        if let channelData = buffer.floatChannelData {
            let channels = Int(format.channelCount)
            let frames = Int(frameCount)
            
            // 混音为单声道
            for frame in 0..<frames {
                var sum: Float = 0
                for channel in 0..<channels {
                    sum += channelData[channel][frame]
                }
                samples.append(sum / Float(channels))
            }
        }
        
        return (samples, Int(format.sampleRate))
    }
    
    /// 重采样
    private func resample(samples: [Float], from oldSR: Int, to newSR: Int) -> [Float] {
        if oldSR == newSR {
            return samples
        }
        
        let ratio = Double(newSR) / Double(oldSR)
        let newLength = Int(Double(samples.count) * ratio)
        
        // 简单的线性插值重采样
        var resampled: [Float] = []
        resampled.reserveCapacity(newLength)
        
        for i in 0..<newLength {
            let srcIndex = Double(i) / ratio
            let srcIndexInt = Int(srcIndex)
            let fraction = Float(srcIndex - Double(srcIndexInt))
            
            if srcIndexInt + 1 < samples.count {
                let sample = samples[srcIndexInt] * (1 - fraction) + samples[srcIndexInt + 1] * fraction
                resampled.append(sample)
            } else if srcIndexInt < samples.count {
                resampled.append(samples[srcIndexInt])
            }
        }
        
        return resampled
    }
    
    private func makeStereoChannelMajorSamples(fromMono samples: [Float]) -> [Float] {
        guard config.numChannels == 2 else { return samples }
        return samples + samples
    }
    
    fileprivate func interleaveChannelMajorAudio(
        _ samples: [Float],
        shape: [Int],
        validFrames: Int? = nil
    ) -> [Float] {
        interleaveChannelMajorAudioSamples(samples, shape: shape, validFrames: validFrames)
    }
    
    /// 保存音频文件
    private func saveAudio(samples: [Float], to path: String) async throws {
        let url = URL(fileURLWithPath: path)
        
        // 创建音频格式
        let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(config.sampleRate),
            channels: AVAudioChannelCount(config.numChannels)
        )!
        
        // 创建音频文件
        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        
        // 创建缓冲区
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(samples.count / config.numChannels)
        ) else {
            throw MOSSTTSError.audioProcessingFailed("Failed to create audio buffer")
        }
        
        // 填充数据
        buffer.frameLength = buffer.frameCapacity
        
        if config.numChannels == 2, let channelData = buffer.floatChannelData {
            // 立体声：交替写入左右声道
            let samplesPerChannel = samples.count / 2
            for i in 0..<samplesPerChannel {
                channelData[0][i] = samples[i * 2]
                channelData[1][i] = samples[i * 2 + 1]
            }
        } else if let channelData = buffer.floatChannelData {
            // 单声道
            for i in 0..<samples.count {
                channelData[0][i] = samples[i]
            }
        }
        
        // 写入文件
        try audioFile.write(from: buffer)
    }
}

final class IncrementalAudioDecoder {
    private let tokenizer: AudioTokenizerONNX
    private let config: AudioTokenizerConfig
    private let framesPerChunk: Int
    private let streamingSession: CodecStreamingDecodeSession?
    private var pendingFrames: [[Int32]] = []
    private var fallbackBufferedFrames: [[Int32]] = []

    init(
        tokenizer: AudioTokenizerONNX,
        decodeStepSession: ONNXSession?,
        config: AudioTokenizerConfig,
        framesPerChunk: Int
    ) {
        self.tokenizer = tokenizer
        self.config = config
        self.framesPerChunk = max(1, framesPerChunk)
        self.streamingSession = decodeStepSession.map { CodecStreamingDecodeSession(session: $0, config: config) }
    }

    func append(frameCodes: [Int32]) throws -> [Float]? {
        if let streamingSession {
            pendingFrames.append(frameCodes)
            guard pendingFrames.count >= framesPerChunk else {
                return nil
            }
            let frames = pendingFrames
            pendingFrames.removeAll(keepingCapacity: true)
            return try streamingSession.runFrames(frames)
        }

        fallbackBufferedFrames.append(frameCodes)
        return nil
    }

    func finish() async throws -> [Float]? {
        if let streamingSession {
            defer { streamingSession.reset() }
            guard !pendingFrames.isEmpty else {
                return nil
            }
            let frames = pendingFrames
            pendingFrames.removeAll(keepingCapacity: true)
            return try streamingSession.runFrames(frames)
        }

        guard !fallbackBufferedFrames.isEmpty else {
            return nil
        }
        let frames = fallbackBufferedFrames
        fallbackBufferedFrames.removeAll(keepingCapacity: true)
        return try await tokenizer.decode(codes: frames)
    }
}

private final class CodecStreamingDecodeSession {
    private let session: ONNXSession
    private let config: AudioTokenizerConfig
    private var stateFeeds: [String: ONNXTensor] = [:]

    private static let transformerLayerCount = 4
    private static let attentionCapacities = [500, 500, 500, 500, 800, 800, 1200, 1200, 1600, 1600, 1600, 1600]
    private static let attentionHeads = 4
    private static let attentionHeadDim = 64

    init(session: ONNXSession, config: AudioTokenizerConfig) {
        self.session = session
        self.config = config
        reset()
    }

    func reset() {
        stateFeeds.removeAll(keepingCapacity: true)

        for index in 0..<Self.transformerLayerCount {
            stateFeeds["transformer_offset_\(index)"] = .int32s([0], shape: [1])
        }

        for (index, capacity) in Self.attentionCapacities.enumerated() {
            stateFeeds["attn_offset_\(index)"] = .int32s([0], shape: [1])
            stateFeeds["attn_cached_keys_\(index)"] = .floats(
                [Float](repeating: 0, count: Self.attentionHeads * capacity * Self.attentionHeadDim),
                shape: [1, Self.attentionHeads, capacity, Self.attentionHeadDim]
            )
            stateFeeds["attn_cached_values_\(index)"] = .floats(
                [Float](repeating: 0, count: Self.attentionHeads * capacity * Self.attentionHeadDim),
                shape: [1, Self.attentionHeads, capacity, Self.attentionHeadDim]
            )
            stateFeeds["attn_cached_positions_\(index)"] = .int32s(
                [Int32](repeating: -1, count: capacity),
                shape: [1, capacity]
            )
        }
    }

    func runFrames(_ frameRows: [[Int32]]) throws -> [Float]? {
        guard !frameRows.isEmpty else { return nil }

        var flatCodes: [Int32] = []
        flatCodes.reserveCapacity(frameRows.count * config.numCodebooks)
        for frame in frameRows {
            if frame.count >= config.numCodebooks {
                flatCodes.append(contentsOf: frame.prefix(config.numCodebooks))
            } else {
                flatCodes.append(contentsOf: frame)
                flatCodes.append(contentsOf: repeatElement(0, count: config.numCodebooks - frame.count))
            }
        }

        var inputs: [String: ONNXTensor] = [
            "audio_codes": .int32s(flatCodes, shape: [1, frameRows.count, config.numCodebooks]),
            "audio_code_lengths": .int32s([Int32(frameRows.count)], shape: [1]),
        ]
        for (name, tensor) in stateFeeds {
            inputs[name] = tensor
        }

        let outputs = try session.run(inputs: inputs, outputs: session.outputNames)

        for index in 0..<Self.transformerLayerCount {
            if let output = outputs["transformer_offset_out_\(index)"] {
                stateFeeds["transformer_offset_\(index)"] = output
            }
        }

        for index in 0..<Self.attentionCapacities.count {
            if let output = outputs["attn_offset_out_\(index)"] {
                stateFeeds["attn_offset_\(index)"] = output
            }
            if let output = outputs["attn_cached_keys_out_\(index)"] {
                stateFeeds["attn_cached_keys_\(index)"] = output
            }
            if let output = outputs["attn_cached_values_out_\(index)"] {
                stateFeeds["attn_cached_values_\(index)"] = output
            }
            if let output = outputs["attn_cached_positions_out_\(index)"] {
                stateFeeds["attn_cached_positions_\(index)"] = output
            }
        }

        guard let outputTensor = outputs["audio"],
              let outputData = outputTensor.toFloats() else {
            throw MOSSTTSError.inferenceFailed("Failed to get streaming decoder audio output")
        }

        let validFrameCount = outputs["audio_lengths"]?.toInt32s()?.first.map(Int.init)
        let samples = interleaveChannelMajorAudioSamples(
            Array(outputData),
            shape: outputTensor.shape,
            validFrames: validFrameCount
        )
        return samples.isEmpty ? nil : samples
    }
}

private func interleaveChannelMajorAudioSamples(
    _ samples: [Float],
    shape: [Int],
    validFrames: Int? = nil
) -> [Float] {
    guard shape.count >= 3 else { return samples }
    let channels = shape[shape.count - 2]
    let frames = shape[shape.count - 1]
    let trimmedFrames = max(0, min(validFrames ?? frames, frames))
    guard channels == 2, samples.count >= channels * frames else {
        if let validFrames = validFrames, validFrames < samples.count {
            return Array(samples.prefix(validFrames))
        }
        return samples
    }

    var interleaved: [Float] = []
    interleaved.reserveCapacity(channels * trimmedFrames)
    let leftOffset = 0
    let rightOffset = frames

    for index in 0..<trimmedFrames {
        interleaved.append(samples[leftOffset + index])
        interleaved.append(samples[rightOffset + index])
    }

    return interleaved
}
