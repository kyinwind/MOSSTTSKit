import Foundation
import onnxruntime
import MOSSTTSKit

/// ONNXSwift 集成示例
/// 展示如何使用 ONNX Runtime Swift API 进行推理
/// 
/// 运行方式：
/// 1. 确保已下载 onnxruntime-swift 包
/// 2. 准备一个 ONNX 模型文件
/// 3. 修改 modelPath 为实际路径
/// 4. 运行 swift run MOSSTTSKit 或直接运行此文件

// MARK: - 示例 1: 基本推理流程

/// 展示基本的 ONNX 推理流程
func exampleBasicInference() throws {
    print("=== 示例 1: 基本推理流程 ===\n")
    
    // 1. 准备输入数据
    let inputData: [Float] = Array(repeating: 0.5, count: 512)
    let inputShape: [Int] = [1, 512]
    
    // 2. 创建张量数据
    let tensorData = ONNXSwift.createTensorData(from: inputData, shape: inputShape)
    
    // 3. 转换为 NSData（如果需要）
    let nsData = NSMutableData(data: tensorData as Data)
    
    // 4. 创建 ORTValue
    let value = try ORTValue(
        tensorData: nsData,
        elementType: .float,
        shape: inputShape.map { NSNumber(value: $0) }
    )
    
    print("✓ 张量创建成功")
    print("  形状: \(inputShape)")
    print("  元素数: \(ONNXSwift.totalCount(for: inputShape))")
}

// MARK: - 示例 2: 完整推理会话

/// 展示完整的推理会话流程
func exampleCompleteSession() throws {
    print("\n=== 示例 2: 完整推理会话 ===\n")
    
    // 模型路径（需要替换为实际路径）
    let modelPath = "/path/to/your/model.onnx"
    
    // 检查模型文件是否存在
    let fileManager = FileManager.default
    guard fileManager.fileExists(atPath: modelPath) else {
        print("⚠️  模型文件不存在: \(modelPath)")
        print("  请下载模型文件或替换为实际路径")
        return
    }
    
    // 1. 创建会话
    let session = try ONNXSwiftSession(modelPath: modelPath)
    
    // 2. 打印模型信息
    print("✓ 会话创建成功")
    print(session.modelInfo)
    
    // 3. 准备输入数据
    guard let inputName = session.inputNames.first else {
        print("⚠️  没有找到输入节点")
        return
    }
    
    guard let inputShape = session.inputShapes[inputName] else {
        print("⚠️  没有找到输入形状")
        return
    }
    
    let totalCount = ONNXSwift.totalCount(for: inputShape)
    let inputData = [Float](repeating: 0.0, count: totalCount)
    
    // 4. 执行推理
    let outputs = try session.run(
        inputs: [inputName: ONNXSwift.dataCopiedFromFloatArray(inputData)],
        inputShapes: [inputName: inputShape],
        inputTypes: [inputName: .float32],
        outputs: session.outputNames
    )
    
    // 5. 处理输出
    print("✓ 推理完成")
    print("  输出数量: \(outputs.count)")
    
    for (name, data) in outputs {
        let count = data.count / MemoryLayout<Float>.size
        let values = ONNXSwift.dataToFloatArray(data)
        print("  \(name): \(count) 个元素, 前5个: \(values.prefix(5))")
    }
}

// MARK: - 示例 3: MOSS-TTS 推理示例

/// 展示 MOSS-TTS 模型推理的典型流程
func exampleMOSSTTSInference() throws {
    print("\n=== 示例 3: MOSS-TTS 推理示例 ===\n")
    
    // 1. 模拟文本编码结果（实际使用 TextTokenizer）
    let textTokens: [Int32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    let textLength: Int32 = Int32(textTokens.count)
    let maxTextLength: Int32 = 512
    
    // 2. 创建文本输入
    let textData = ONNXSwift.createInt32TensorData(from: textTokens, shape: [1, 512])
    let lengthData = Data(bytes: [textLength], count: MemoryLayout<Int32>.size)
    let maxLenData = Data(bytes: [maxTextLength], count: MemoryLayout<Int32>.size)
    
    print("✓ 输入数据准备完成")
    print("  文本长度: \(textTokens.count)")
    print("  最大文本长度: \(maxTextLength)")
    
    // 3. 模拟音频解码结果
    // 实际推理会返回音频 token，需要 AudioTokenizer 解码
    
    // 4. 模拟音频采样生成
    let sampleRate: Int = 24000
    let duration: Double = 1.0  // 1秒
    let numSamples = Int(Double(sampleRate) * duration)
    let audioSamples: [Float] = (0..<numSamples).map { i in
        Float(sin(Double(i) * 2.0 * Double.pi * 440.0 / Double(sampleRate)))
    }
    
    print("✓ 音频生成完成")
    print("  采样率: \(sampleRate)")
    print("  时长: \(duration) 秒")
    print("  样本数: \(numSamples)")
    
    // 5. 转换为 16-bit PCM
    let pcmSamples: [Int16] = audioSamples.map { sample in
        Int16(max(-1.0, min(1.0, sample)) * Double(Int16.max))
    }
    
    // 6. 保存为 WAV
    let wavData = createWAVData(samples: pcmSamples, sampleRate: sampleRate)
    let outputPath = "/tmp/moss_tts_output.wav"
    try wavData.write(to: URL(fileURLWithPath: outputPath))
    
    print("✓ WAV 文件已保存: \(outputPath)")
}

// MARK: - 辅助函数

/// 创建 WAV 文件数据
func createWAVData(samples: [Int16], sampleRate: Int) -> Data {
    var data = Data()
    
    // WAV 头部
    let numChannels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let byteRate = UInt32(sampleRate * Int(numChannels) * Int(bitsPerSample) / 8)
    let blockAlign = UInt16(numChannels * bitsPerSample / 8)
    let dataSize = UInt32(samples.count * MemoryLayout<Int16>.size)
    let fileSize = dataSize + 36
    
    // RIFF header
    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
    data.append(contentsOf: "WAVE".utf8)
    
    // fmt chunk
    data.append(contentsOf: "fmt ".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
    data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })
    
    // data chunk
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })
    
    // 音频数据
    for sample in samples {
        data.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
    }
    
    return data
}

// MARK: - 主函数

/// 主函数 - 运行所有示例
@main
func main() {
    print("🎙️  ONNXSwift 集成示例\n")
    print("=" .padding(toLength: 50, withPad: "=", startingAt: 0))
    
    do {
        // 示例 1: 基本推理流程
        try exampleBasicInference()
        
        // 示例 2: 完整推理会话（需要模型文件）
        try exampleCompleteSession()
        
        // 示例 3: MOSS-TTS 推理示例
        try exampleMOSSTTSInference()
        
        print("\n" + "=" .padding(toLength: 50, withPad: "=", startingAt: 0))
        print("✅ 所有示例运行完成")
        
    } catch {
        print("\n❌ 错误: \(error.localizedDescription)")
    }
}
