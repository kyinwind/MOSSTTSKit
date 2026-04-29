/// AudioTokenizer.swift
/// 
/// 音频 tokenizer 实现
/// MOSS-Audio-Tokenizer-Nano 用于音频编码/解码
/// 
/// 文档：
/// - https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX
/// - 48kHz 采样率，立体声输出
/// - 12.5 Hz 令牌速率，16 RVQ 代码本

import Foundation

/// 音频编码结果
public struct AudioEncodingResult: Sendable {
    /// 音频编码 (RVQ codes)
    /// Shape: [num_frames, num_codebooks] = [T, 16]
    public let codes: [[Int32]]
    
    /// 帧数
    public var numFrames: Int { codes.count }
    
    /// 代码本数量
    public var numCodebooks: Int { codes.first?.count ?? 0 }
    
    /// 原始采样率
    public let sampleRate: Int
    
    /// 原始音频时长（秒）
    public let duration: Double
}

/// 音频 tokenizer 接口
public protocol MOSSAudioTokenizer: Sendable {
    /// 编码音频文件
    /// - Parameter audioPath: 音频文件路径 (WAV, MP3, FLAC 等)
    /// - Returns: 音频编码结果
    func encode(audioPath: String) async throws -> AudioEncodingResult
    
    /// 编码原始音频数据
    /// - Parameters:
    ///   - samples: 音频采样数据 (Float, 范围 [-1, 1])
    ///   - sampleRate: 采样率
    /// - Returns: 音频编码结果
    func encode(samples: [Float], sampleRate: Int) async throws -> AudioEncodingResult
    
    /// 解码音频编码
    /// - Parameter codes: 音频编码
    /// - Returns: 音频采样数据 (Float, 范围 [-1, 1])
    func decode(codes: [[Int32]]) async throws -> [Float]
    
    /// 解码并保存为文件
    /// - Parameters:
    ///   - codes: 音频编码
    ///   - outputPath: 输出文件路径
    func decode(codes: [[Int32]], to outputPath: String) async throws
    
    /// 获取采样率
    var sampleRate: Int { get }
    
    /// 获取通道数
    var numChannels: Int { get }
}

/// 音频 tokenizer 配置
public struct AudioTokenizerConfig: Sendable {
    /// 采样率
    public let sampleRate: Int = 48000
    
    /// 通道数
    public let numChannels: Int = 2
    
    /// 令牌速率 (Hz)
    public let frameRate: Int = 12
    
    /// 每个 codec frame 对应的采样点数。
    public let downsampleRate: Int = 3840
    
    /// RVQ 代码本数量
    public let numCodebooks: Int = 16
    
    /// 期望的音频格式
    public var expectedFormat: AudioFormat {
        AudioFormat(sampleRate: sampleRate, channels: numChannels)
    }
    
    public init() {}
}

/// 音频格式
public struct AudioFormat: Sendable, Equatable {
    public let sampleRate: Int
    public let channels: Int
    
    public init(sampleRate: Int, channels: Int) {
        self.sampleRate = sampleRate
        self.channels = channels
    }
    
    /// 检查音频格式是否匹配
    public func matches(sampleRate: Int, channels: Int) -> Bool {
        self.sampleRate == sampleRate && self.channels == channels
    }
}

/// 音频文件工具
public enum AudioFileUtils {
    /// 检查音频文件是否存在
    public static func exists(at path: String) -> Bool {
        FileManager.default.fileExists(atPath: path)
    }
    
    /// 获取音频文件信息
    public static func getInfo(at path: String) throws -> AudioFileInfo {
        // TODO: 使用 AVFoundation 或 AudioToolbox 获取音频信息
        // 暂时返回模拟数据
        return AudioFileInfo(
            path: path,
            sampleRate: 48000,
            channels: 2,
            duration: 0,
            format: "wav"
        )
    }
}

/// 音频文件信息
public struct AudioFileInfo: Sendable {
    public let path: String
    public let sampleRate: Int
    public let channels: Int
    public let duration: Double
    public let format: String
}
