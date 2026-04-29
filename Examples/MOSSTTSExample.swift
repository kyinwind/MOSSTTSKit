/// MOSSTTSExample.swift
/// 
/// MOSSTTSKit 使用示例
/// 
/// 这个文件展示如何使用 MOSSTTSKit 进行语音合成

import Foundation
import MOSSTTSKit

/// 示例：基本用法
func basicUsage() async throws {
    // 1. 创建 TTS 实例
    let tts = try await MOSSTTSKit()
    
    // 2. 生成语音
    let result = try await tts.speak(text: "你好，世界！这是 MOSS-TTS 的语音合成演示。")
    
    // 3. 保存为 WAV 文件
    try MOSSAudioExporter.export(result: result, to: "output.wav")
    
    print("生成了 \(result.audioSamples.count) 个采样点")
    print("采样率: \(result.sampleRate) Hz")
    print("生成时间: \(result.timings?.fullPipeline ?? 0) 秒")
}

/// 示例：带进度回调
func withProgressCallback() async throws {
    let tts = try await MOSSTTSKit()
    
    let result = try await tts.speak(
        text: "这是一段带进度回调的语音合成示例。",
        options: MOSSTTSOptions()
    ) { progress in
        let percent = Double(progress.currentStep) / Double(progress.totalSteps) * 100
        print("进度: \(Int(percent))%")
        return true // 返回 false 可取消
    }
    
    print("生成了 \(result.audioSamples.count) 个采样点")
}

/// 示例：自定义配置
func customConfig() async throws {
    // 使用自定义配置
    let config = MOSSTTSConfig(
        modelVariant: .mossTTSNano100M,
        download: true,
        loadImmediately: true,
        verbose: true,
        seed: 42
    )
    
    let tts = try await MOSSTTSKit(config: config)
    
    // 使用自定义生成选项
    let options = MOSSTTSOptions(
        temperature: 0.8,
        topK: 40,
        maxNewTokens: 1024,
        outputSampleRate: 48000
    )
    
    let result = try await tts.speak(
        text: "使用自定义配置和生成选项。",
        options: options
    )
    
    try MOSSAudioExporter.export(result: result, to: "custom_output.wav")
}

/// 示例：从本地模型初始化
func fromLocalModels() async throws {
    // 指定本地模型文件夹
    let modelFolder = URL(fileURLWithPath: "/path/to/models")
    
    if MOSSTTSDownloader.isModelDownloaded(at: modelFolder) {
        let tts = try await MOSSTTSKit.fromLocalModels(at: modelFolder)
        
        let result = try await tts.speak(text: "从本地模型生成语音。")
        try MOSSAudioExporter.export(result: result, to: "local_output.wav")
    } else {
        print("模型未找到，请先下载模型。")
    }
}

/// 示例：从 HuggingFace 下载并初始化
func fromHuggingFace() async throws {
    // 从 HuggingFace 下载并初始化
    let tts = try await MOSSTTSKit.fromHuggingFace(
        variant: .mossTTSNano100M,
        token: nil // 如果需要访问私有模型，传入 HuggingFace token
    )
    
    let result = try await tts.speak(text: "从 HuggingFace 下载模型并生成语音。")
    try MOSSAudioExporter.export(result: result, to: "hf_output.wav")
}

/// 示例：音频播放
func audioPlayback() async throws {
    let tts = try await MOSSTTSKit()
    
    let result = try await tts.speak(text: "这段语音将被播放。")
    
    let player = MOSSAudioPlayer()
    try player.play(samples: result.audioSamples, sampleRate: result.sampleRate)
    
    // 等待播放完成（实际应用中应使用适当的方式）
    try await Task.sleep(nanoseconds: 5_000_000_000)
    
    player.stop()
}

/// 示例：流式生成
func streamingGeneration() async throws {
    let tts = try await MOSSTTSKit()
    
    var allSamples: [Float] = []
    
    let result = try await tts.speak(
        text: "这是一段流式生成的语音。",
        options: .streaming
    ) { progress in
        // 实时处理音频块
        allSamples.append(contentsOf: progress.audioSamples)
        print("收到 \(progress.audioSamples.count) 个采样点")
        return true
    }
    
    // 最终音频
    let finalAudio = allSamples.isEmpty ? result.audioSamples : allSamples
    try MOSSAudioExporter.exportWAV(
        samples: finalAudio,
        sampleRate: result.sampleRate,
        to: "streaming_output.wav"
    )
}

/// 主函数示例
@main
struct MOSSTTSExampleMain {
    static func main() async {
        do {
            print("=== MOSS-TTS-Nano 使用示例 ===\n")
            
            // 基本用法
            print("1. 基本用法")
            try await basicUsage()
            print("✓ 基本用法完成\n")
            
            // 带进度回调
            print("2. 带进度回调")
            try await withProgressCallback()
            print("✓ 带进度回调完成\n")
            
            // 自定义配置
            print("3. 自定义配置")
            try await customConfig()
            print("✓ 自定义配置完成\n")
            
            print("所有示例执行完成！")
            
        } catch {
            print("错误: \(error)")
        }
    }
}
