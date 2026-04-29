/// MOSSAudioPlayer.swift
/// 
/// 音频播放和导出功能
/// 使用 AVFoundation 实现

import Foundation
import AVFoundation

/// 音频播放器
public final class MOSSAudioPlayer: @unchecked Sendable {
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var audioFile: AVAudioFile?
    
    /// 初始化
    public init() {}
    
    /// 播放音频数据
    /// - Parameters:
    ///   - samples: 音频采样数据
    ///   - sampleRate: 采样率
    public func play(samples: [Float], sampleRate: Int) throws {
        let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: 2
        )!
        
        // 创建音频缓冲区
        let frameCount = AVAudioFrameCount(samples.count / 2)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw MOSSTTSError.audioProcessingFailed("Failed to create audio buffer")
        }
        
        buffer.frameLength = frameCount
        
        // 填充数据（立体声）
        if let channelData = buffer.floatChannelData {
            let samplesPerChannel = samples.count / 2
            for i in 0..<samplesPerChannel {
                channelData[0][i] = samples[i * 2]
                channelData[1][i] = samples[i * 2 + 1]
            }
        }
        
        // 设置音频引擎
        audioEngine = AVAudioEngine()
        guard let engine = audioEngine else {
            throw MOSSTTSError.audioProcessingFailed("Failed to create audio engine")
        }
        
        playerNode = AVAudioPlayerNode()
        guard let player = playerNode else {
            throw MOSSTTSError.audioProcessingFailed("Failed to create player node")
        }
        
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        
        do {
            try engine.start()
        } catch {
            throw MOSSTTSError.audioProcessingFailed("Failed to start audio engine: \(error)")
        }
        
        // 播放
        player.scheduleBuffer(buffer, at: nil, options: []) {
            // 播放完成
        }
        player.play()
    }
    
    /// 停止播放
    public func stop() {
        playerNode?.stop()
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil
    }
    
    deinit {
        stop()
    }
}

/// 音频导出器
public enum MOSSAudioExporter {
    /// 导出音频数据为 WAV 文件
    /// - Parameters:
    ///   - samples: 音频采样数据（Float，范围 [-1, 1]）
    ///   - sampleRate: 采样率
    ///   - channels: 声道数
    ///   - to: 输出文件路径
    public static func exportWAV(
        samples: [Float],
        sampleRate: Int,
        channels: Int = 2,
        to path: String
    ) throws {
        let url = URL(fileURLWithPath: path)
        
        // 创建音频格式
        let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(sampleRate),
            channels: AVAudioChannelCount(channels)
        )!
        
        // 创建音频文件
        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        
        // 创建缓冲区
        let frameCount = AVAudioFrameCount(samples.count / channels)
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        ) else {
            throw MOSSTTSError.audioProcessingFailed("Failed to create audio buffer")
        }
        
        buffer.frameLength = frameCount
        
        // 填充数据
        if channels == 2, let channelData = buffer.floatChannelData {
            let samplesPerChannel = samples.count / 2
            for i in 0..<samplesPerChannel {
                channelData[0][i] = samples[i * 2]
                channelData[1][i] = samples[i * 2 + 1]
            }
        } else if let channelData = buffer.floatChannelData {
            for i in 0..<samples.count {
                channelData[0][i] = samples[i]
            }
        }
        
        // 写入文件
        try audioFile.write(from: buffer)
    }
    
    /// 导出 MOSSSpeechResult 为 WAV 文件
    /// - Parameters:
    ///   - result: 语音结果
    ///   - to: 输出文件路径
    public static func export(
        result: MOSSSpeechResult,
        to path: String
    ) throws {
        try exportWAV(
            samples: result.audioSamples,
            sampleRate: result.sampleRate,
            channels: 2,
            to: path
        )
    }
    
    /// 导出为 WAV Data
    /// - Parameters:
    ///   - samples: 音频采样数据
    ///   - sampleRate: 采样率
    ///   - channels: 声道数
    /// - Returns: WAV 格式的 Data
    public static func exportToData(
        samples: [Float],
        sampleRate: Int,
        channels: Int = 2
    ) throws -> Data {
        // 创建临时文件
        let tempDir = FileManager.default.temporaryDirectory
        let tempURL = tempDir.appendingPathComponent(UUID().uuidString + ".wav")
        
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }
        
        try exportWAV(samples: samples, sampleRate: sampleRate, channels: channels, to: tempURL.path)
        
        return try Data(contentsOf: tempURL)
    }
}
