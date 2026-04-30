/// MOSSTTSKitTests.swift
/// 
/// MOSSTTSKit 单元测试

import XCTest
@testable import MOSSTTSKit

final class MOSSTTSKitTests: XCTestCase {
    
    // MARK: - TensorUtils Tests
    
    func testFloatArrayToDataConversion() throws {
        let floats: [Float] = [1.0, 2.0, 3.0, 4.0]
        let data = TensorUtils.floatsToData(floats)
        
        // 验证数据大小
        XCTAssertEqual(data.count, floats.count * MemoryLayout<Float>.size)
        
        // 验证转换回来
        let converted = TensorUtils.dataToFloats(data)
        XCTAssertEqual(converted, floats)
    }
    
    func testInt32ArrayToDataConversion() throws {
        let ints: [Int32] = [1, 2, 3, 4]
        let data = TensorUtils.int32sToData(ints)
        
        // 验证数据大小
        XCTAssertEqual(data.count, ints.count * MemoryLayout<Int32>.size)
        
        // 验证转换回来
        let converted = TensorUtils.dataToInt32s(data)
        XCTAssertEqual(converted, ints)
    }
    
    func testONNXTensorToInt32s() throws {
        let values: [Int32] = [10, 20, 30]
        let tensor = ONNXTensor.int32s(values, shape: [1, 3])
        
        XCTAssertEqual(tensor.toInt32s(), values)
        XCTAssertNil(tensor.toInt64s())
        XCTAssertNil(tensor.toFloats())
    }
    
    func testTopKSelection() throws {
        // 创建 values: [0.1, 0.5, 0.3, 0.7, 0.2]
        let values: [Float] = [0.1, 0.5, 0.3, 0.7, 0.2]
        let k = 2
        
        let result = TensorUtils.topK(values, k: k)
        
        // 验证返回数量
        XCTAssertEqual(result.count, k)
        
        // 验证排序（降序）
        if result.count == 2 {
            XCTAssertGreaterThan(result[0].value, result[1].value)
        }
    }
    
    func testTemperatureSampling() throws {
        // 测试温度采样
        let logits: [Float] = [1.0, 2.0, 3.0]
        let temperature: Float = 1.0
        
        let result = TensorUtils.temperatureSample(logits, temperature: temperature)
        
        // 验证返回概率分布
        XCTAssertEqual(result.count, logits.count)
        XCTAssertEqual(result.reduce(0, +), 1.0, accuracy: 0.001)
    }
    
    func testApplyRepetitionPenalty() throws {
        // 创建 logits
        let logits: [Float] = [1.0, 2.0, 3.0, 2.0, 1.0]
        let previousTokens: [Int] = [1, 2] // 索引 1 和 2
        let penalty: Float = 1.5
        
        let result = TensorUtils.applyRepetitionPenalty(logits, previousTokens: previousTokens, penalty: penalty)
        
        // 验证被惩罚的 token 的 logit 变小了
        XCTAssertLessThan(result[1], logits[1]) // 原来 2.0
        XCTAssertLessThan(result[2], logits[2]) // 原来 3.0
    }
    
    // MARK: - Model Variant Tests
    
    func testModelVariantProperties() throws {
        let nano = MOSSModelVariant.mossTTSNano100M
        
        XCTAssertEqual(nano.modelRepo, "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX")
        XCTAssertEqual(nano.tokenizerRepo, "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX")
    }
    
    // MARK: - Speaker Tests
    
    func testSpeakerCreation() throws {
        let speaker = MOSSSpeaker(
            name: "Test Speaker",
            referenceAudioPath: "/path/to/audio.wav"
        )
        
        XCTAssertEqual(speaker.name, "Test Speaker")
        XCTAssertNil(speaker.identifier)
        XCTAssertNil(speaker.displayName)
        XCTAssertNil(speaker.group)
        XCTAssertNil(speaker.audioFileName)
        XCTAssertEqual(speaker.referenceAudioPath, "/path/to/audio.wav")
        XCTAssertNil(speaker.referenceAudioData)
        XCTAssertNil(speaker.embedding)
        XCTAssertNil(speaker.referenceAudioCodes)
        XCTAssertFalse(speaker.hasEmbedding)
    }
    
    func testSpeakerStoresCloningData() throws {
        let embedding: [Float] = [0.1, 0.2, 0.3]
        let codes: [[Int32]] = [[1, 2, 3], [4, 5, 6]]
        let speaker = MOSSSpeaker(
            name: "Cloned Speaker",
            embedding: embedding,
            referenceAudioCodes: codes
        )
        
        XCTAssertTrue(speaker.hasEmbedding)
        XCTAssertEqual(speaker.getEmbedding(), embedding)
        XCTAssertEqual(speaker.referenceAudioCodes, codes)
    }
    
    func testSpeakerStoresBuiltinMetadata() throws {
        let speaker = MOSSSpeaker(
            identifier: "Junhao",
            name: "Junhao",
            displayName: "CN 欢迎关注模思智能",
            group: "Chinese Male",
            audioFileName: "zh_1.wav",
            referenceAudioCodes: [[1, 2, 3]]
        )
        
        XCTAssertEqual(speaker.identifier, "Junhao")
        XCTAssertEqual(speaker.name, "Junhao")
        XCTAssertEqual(speaker.displayName, "CN 欢迎关注模思智能")
        XCTAssertEqual(speaker.group, "Chinese Male")
        XCTAssertEqual(speaker.audioFileName, "zh_1.wav")
    }
    
    // MARK: - Speech Result Tests
    
    func testSpeechResultCreation() throws {
        let audioSamples: [Float] = [0.1, 0.2, 0.3, 0.4]
        let timings = MOSSSpeechTimings()
        
        let result = MOSSSpeechResult(
            audioSamples: audioSamples,
            sampleRate: 48000,
            timings: timings
        )
        
        XCTAssertEqual(result.audioSamples.count, 4)
        XCTAssertEqual(result.sampleRate, 48000)
        XCTAssertEqual(result.audioDuration, 4.0 / 48000.0, accuracy: 0.0001)
    }
    
    // MARK: - AudioFrame Tests
    
    func testAudioFrameCreation() throws {
        let samples: [Float] = [0.1, 0.2, 0.3]
        let frame = AudioFrame(samples: samples, index: 0)
        
        XCTAssertEqual(frame.samples.count, 3)
        XCTAssertEqual(frame.index, 0)
    }
    
    func testAudioFrameTimestamp() throws {
        let samples: [Float] = [0.1, 0.2, 0.3]
        let frame = AudioFrame(samples: samples, index: 1, timestamp: 0.5)
        
        XCTAssertEqual(frame.timestamp, 0.5, accuracy: 0.001)
    }
}
