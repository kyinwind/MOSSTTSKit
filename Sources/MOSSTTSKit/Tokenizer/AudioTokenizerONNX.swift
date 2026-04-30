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
    
    /// 编码器模型路径
    public let encodeModelPath: String
    
    /// 解码器模型路径
    public let decodeModelPath: String
    
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
        config: AudioTokenizerConfig = AudioTokenizerConfig()
    ) {
        self.encodeModelPath = encodeModelPath
        self.decodeModelPath = decodeModelPath
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
        
        guard FileManager.default.fileExists(atPath: encodePath) else {
            throw MOSSTTSError.modelNotFound("Encode model not found: \(encodePath)")
        }
        
        guard FileManager.default.fileExists(atPath: decodePath) else {
            throw MOSSTTSError.modelNotFound("Decode model not found: \(decodePath)")
        }
        
        let tokenizer = AudioTokenizerONNX(
            encodeModelPath: encodePath,
            decodeModelPath: decodePath
        )
        
        try await tokenizer.load()
        return tokenizer
    }
    
    /// 加载模型
    public func load() async throws {
        encodeSession = try ONNXSession(modelPath: encodeModelPath)
        decodeSession = try ONNXSession(modelPath: decodeModelPath)
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

        return interleaveChannelMajorAudio(
            Array(outputData),
            shape: outputTensor.shape,
            validFrames: validFrameCount
        )
    }
    
    public func decode(codes: [[Int32]], to outputPath: String) async throws {
        let samples = try await decode(codes: codes)
        try await saveAudio(samples: samples, to: outputPath)
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
    
    private func interleaveChannelMajorAudio(
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
