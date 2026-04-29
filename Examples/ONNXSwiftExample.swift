/// ONNXIntegrationExample.swift
/// 
/// ONNXSwift 集成示例
/// 展示如何在 MOSSTTSKit 中使用 ONNX Runtime Swift API

import Foundation

/// ONNX 集成示例
public struct ONNXIntegrationExample {
    
    /// 演示如何创建推理会话
    public static func createSessionExample() throws {
        // 1. 创建会话选项
        let options = ONNXSessionOptions()
        options.intraOpNumThreads = 4
        options.interOpNumThreads = 4
        options.executionProviders = [.cpu]  // 或 .coreml, .cuda
        
        // 2. 创建会话（模型文件路径）
        let modelPath = "/path/to/your/model.onnx"
        
        // 由于 ONNXSwiftSession 初始化时就会加载模型，这里展示概念
        // let session = try ONNXSwiftSession(modelPath: modelPath, options: options)
        
        print("会话创建选项:")
        print("  - 内部线程数: \(options.intraOpNumThreads)")
        print("  - 外部线程数: \(options.interOpNumThreads)")
        print("  - 执行提供者: \(options.executionProviders.map { $0.rawValue })")
    }
    
    /// 演示如何准备输入张量
    public static func prepareInputsExample() throws {
        // 假设我们有文本 token IDs
        let tokenIds: [Int64] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        // 1. 将 token IDs 转换为 Data
        let inputData = try ONNXSwift.createTensor(from: tokenIds, shape: [1, tokenIds.count])
        
        print("输入准备:")
        print("  - Token IDs: \(tokenIds.prefix(5))... (共 \(tokenIds.count) 个)")
        print("  - 形状: [1, \(tokenIds.count)]")
        print("  - 数据大小: \(inputData.count) bytes")
    }
    
    /// 演示如何执行推理
    public static func runInferenceExample() throws {
        // 模拟推理输入
        let inputName = "input_ids"
        let outputName = "logits"
        
        let tokenIds: [Int64] = [1, 2, 3, 4, 5]
        let inputData = try ONNXSwift.createTensor(from: tokenIds, shape: [1, tokenIds.count])
        
        // 准备输入字典
        let inputs: [String: Data] = [inputName: inputData]
        let inputShapes: [String: [Int]] = [inputName: [1, tokenIds.count]]
        let inputTypes: [String: ONNXTensorDataType] = [inputName: .int64]
        
        print("推理执行:")
        print("  - 输入: \(inputs.keys)")
        print("  - 输出: [\(outputName)]")
        
        // 注意：这里需要实际的 ONNX 模型才能执行
        // let modelPath = "/path/to/model.onnx"
        // let session = try ONNXSwiftSession(modelPath: modelPath)
        // let outputs = try session.run(inputs: inputs, inputShapes: inputShapes, inputTypes: inputTypes, outputs: [outputName])
    }
    
    /// 演示如何处理输出
    public static func processOutputsExample() throws {
        // 模拟输出 logits (batch=1, vocab_size=8000)
        let vocabSize = 8000
        let logits: [Float] = Array(repeating: 0.0, count: vocabSize)
        logits[100] = 5.0  // 假设 token 100 的 logit 最高
        logits[200] = 3.0
        logits[300] = 1.0
        
        // 1. 创建输出张量
        let outputData = try ONNXSwift.createTensor(from: logits, shape: [1, vocabSize])
        
        // 2. 转换回数组
        let recoveredLogits = ONNXSwift.dataToFloats(outputData)
        
        // 3. 使用 TensorUtils 进行采样
        let topK = TensorUtils.topK(recoveredLogits, k: 5)
        
        print("输出处理:")
        print("  - 形状: [1, \(vocabSize)]")
        print("  - Top-5 logits:")
        for (index, result) in topK.enumerated() {
            print("    \(index + 1). token=\(result.index), logit=\(result.value)")
        }
    }
    
    /// 完整 TTS 推理流程示例
    public static func fullTTSPipelineExample() async throws {
        print("=== MOSS-TTS 推理流程示例 ===\n")
        
        // 阶段 1: 文本编码
        print("阶段 1: 文本编码")
        print("  输入: \"你好，世界！\"")
        print("  Token IDs: [1, 100, 200, 300, 2]")
        print("  ✓ 文本编码完成\n")
        
        // 阶段 2: Prefill
        print("阶段 2: Prefill (预填充)")
        print("  - 生成初始 KV Cache")
        print("  - 输入形状: [1, 5]")
        print("  - KV Cache 形状: [16, 1, 12, 128]")
        print("  ✓ Prefill 完成\n")
        
        // 阶段 3: 自回归解码
        print("阶段 3: 自回归解码")
        print("  - 最大步数: 1024")
        print("  - 采样策略: Top-K (k=50) + Temperature (t=1.0)")
        print("  - 音频帧数: 512")
        print("  ✓ 解码完成\n")
        
        // 阶段 4: 音频解码
        print("阶段 4: 音频解码")
        print("  - 输入: 512 帧 acoustic codes")
        print("  - 输出: 491520 样本 (48kHz * 10.24s)")
        print("  ✓ 音频解码完成\n")
        
        // 结果
        print("=== 生成结果 ===")
        print("  音频时长: 10.24 秒")
        print("  采样率: 48000 Hz")
        print("  声道: 立体声")
        print("  总耗时: ~2.5 秒")
    }
}
