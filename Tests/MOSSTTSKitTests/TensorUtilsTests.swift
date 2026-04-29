/// TensorUtilsTests.swift
/// 
/// TensorUtils 单元测试

import XCTest
@testable import MOSSTTSKit

final class TensorUtilsTests: XCTestCase {
    
    // MARK: - Data Conversion Tests
    
    func testFloatsToDataAndBack() throws {
        let original: [Float] = [
            0.0, 1.0, -1.0, 0.5, -0.5,
            Float.leastNormalMagnitude, Float.greatestFiniteMagnitude
        ]
        
        let data = TensorUtils.floatsToData(original)
        let recovered = TensorUtils.dataToFloats(data)
        
        XCTAssertEqual(original.count, recovered.count)
        
        for (original, recovered) in zip(original, recovered) {
            XCTAssertEqual(original, recovered)
        }
    }
    
    func testInt32sToDataAndBack() throws {
        let original: [Int32] = [
            0, 1, -1, Int32.max, Int32.min
        ]
        
        let data = TensorUtils.int32sToData(original)
        let recovered = TensorUtils.dataToInt32s(data)
        
        XCTAssertEqual(original.count, recovered.count)
        
        for (original, recovered) in zip(original, recovered) {
            XCTAssertEqual(original, recovered)
        }
    }
    
    func testInt64sToDataAndBack() throws {
        let original: [Int64] = [
            0, 1, -1, Int64.max, Int64.min
        ]
        
        let data = TensorUtils.int64sToData(original)
        let recovered = TensorUtils.dataToInt64s(data)
        
        XCTAssertEqual(original.count, recovered.count)
        
        for (original, recovered) in zip(original, recovered) {
            XCTAssertEqual(original, recovered)
        }
    }
    
    func testEmptyArrayConversion() throws {
        let emptyFloats: [Float] = []
        let emptyData = TensorUtils.floatsToData(emptyFloats)
        let recovered = TensorUtils.dataToFloats(emptyData)
        
        XCTAssertTrue(recovered.isEmpty)
    }
    
    func testLargeArrayConversion() throws {
        let largeArray = [Float](repeating: 1.0, count: 10000)
        let data = TensorUtils.floatsToData(largeArray)
        let recovered = TensorUtils.dataToFloats(data)
        
        XCTAssertEqual(largeArray.count, recovered.count)
        XCTAssertTrue(recovered.allSatisfy { $0 == 1.0 })
    }
    
    // MARK: - Shape Operations Tests
    
    func testFlattenedSize() throws {
        XCTAssertEqual(TensorUtils.flattenedSize(of: [2, 3, 4]), 24)
        XCTAssertEqual(TensorUtils.flattenedSize(of: [1, 10]), 10)
        XCTAssertEqual(TensorUtils.flattenedSize(of: [100]), 100)
    }
    
    func testReshape() throws {
        let array = [1, 2, 3, 4, 5, 6]
        let reshaped = TensorUtils.reshape(array, to: [2, 3])
        
        XCTAssertNotNil(reshaped)
        XCTAssertEqual(reshaped?.count, 2)
        XCTAssertEqual(reshaped?[0], [1, 2, 3])
        XCTAssertEqual(reshaped?[1], [4, 5, 6])
    }
    
    func testReshapeInvalidSize() throws {
        let array = [1, 2, 3]
        let reshaped = TensorUtils.reshape(array, to: [2, 3])
        
        XCTAssertNil(reshaped)
    }
    
    // MARK: - Top-K Selection Tests
    
    func testTopKWithSmallK() throws {
        let values: [Float] = [0.1, 0.5, 0.3, 0.7, 0.2]
        let k = 1
        
        let result = TensorUtils.topK(values, k: k)
        
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0].index, 3) // 最大值 0.7 在索引 3
        XCTAssertEqual(result[0].value, 0.7)
    }
    
    func testTopKWithLargeK() throws {
        let values: [Float] = [0.1, 0.5, 0.3, 0.7, 0.2]
        let k = 10 // 超过数组长度
        
        let result = TensorUtils.topK(values, k: k)
        
        XCTAssertEqual(result.count, 0)  // k > count 时返回空
    }
    
    func testTopKWithEqualValues() throws {
        let values: [Float] = [1.0, 1.0, 1.0, 1.0, 1.0]
        let k = 3
        
        let result = TensorUtils.topK(values, k: k)
        
        XCTAssertEqual(result.count, 3)
    }
    
    func testTopKWithNegativeValues() throws {
        let values: [Float] = [-5.0, -1.0, -3.0, -2.0, -4.0]
        let k = 2
        
        let result = TensorUtils.topK(values, k: k)
        
        XCTAssertEqual(result.count, 2)
        // -1.0 是最大值
        XCTAssertEqual(result[0].index, 1)
        XCTAssertEqual(result[0].value, -1.0)
    }
    
    // MARK: - Temperature Sampling Tests
    
    func testTemperatureSample() throws {
        let logits: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let temperature: Float = 1.0
        
        let result = TensorUtils.temperatureSample(logits, temperature: temperature)
        
        // 验证返回概率分布
        XCTAssertEqual(result.count, logits.count)
        
        // 验证和为 1
        let sum = result.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 0.001)
    }
    
    func testTemperatureSampleWithZeroTemperature() throws {
        let logits: [Float] = [1.0, 5.0, 2.0, 0.5, 1.5]
        let temperature: Float = 0.0
        
        // 零温度返回原始 logits
        let result = TensorUtils.temperatureSample(logits, temperature: temperature)
        
        XCTAssertEqual(result, logits)
    }
    
    func testTemperatureSampleWithHighTemperature() throws {
        let logits: [Float] = [1.0, 2.0, 3.0]
        let temperature: Float = 10.0
        
        let result = TensorUtils.temperatureSample(logits, temperature: temperature)
        
        // 高温度会使分布更均匀
        XCTAssertEqual(result.count, logits.count)
        XCTAssertEqual(result.reduce(0, +), 1.0, accuracy: 0.001)
    }
    
    func testSampleFromDistribution() throws {
        let probabilities: [Float] = [0.1, 0.2, 0.3, 0.4]
        let result = TensorUtils.sampleFromDistribution(probabilities)
        
        XCTAssertGreaterThanOrEqual(result, 0)
        XCTAssertLessThan(result, probabilities.count)
    }
    
    // MARK: - Repetition Penalty Tests
    
    func testApplyRepetitionPenaltyWithPositiveLogits() throws {
        let logits: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let previousTokens: [Int] = [1, 2]
        let penalty: Float = 2.0
        
        let result = TensorUtils.applyRepetitionPenalty(logits, previousTokens: previousTokens, penalty: penalty)
        
        // 正数 logit 会被除以 penalty（变小）
        XCTAssertLessThan(result[1], logits[1])
        XCTAssertLessThan(result[2], logits[2])
        // 未出现的 token 不变
        XCTAssertEqual(result[0], 1.0)
        XCTAssertEqual(result[3], 4.0)
        XCTAssertEqual(result[4], 5.0)
    }
    
    func testApplyRepetitionPenaltyWithPenaltyOfOne() throws {
        let logits: [Float] = [1.0, 2.0, 3.0]
        let previousTokens: [Int] = [1, 2]
        let penalty: Float = 1.0
        
        let result = TensorUtils.applyRepetitionPenalty(logits, previousTokens: previousTokens, penalty: penalty)
        
        // penalty 为 1.0 时，logits 不变
        XCTAssertEqual(result, logits)
    }
    
    // MARK: - Audio Processing Tests
    
    func testNormalizeAudio() throws {
        let audio: [Float] = [-1.0, 0.0, 1.0]
        let normalized = TensorUtils.normalizeAudio(audio)
        
        // 归一化后应该在 [-1, 1] 范围内
        XCTAssertLessThanOrEqual(normalized.max() ?? 0, 1.0)
        XCTAssertGreaterThanOrEqual(normalized.min() ?? 0, -1.0)
    }
    
    func testNormalizeAudioWithSmallValues() throws {
        let audio: [Float] = [0.1, 0.2, 0.3]
        let normalized = TensorUtils.normalizeAudio(audio)
        
        // 小值应该保持不变
        XCTAssertEqual(audio, normalized)
    }
    
    func testApplyVolumeGain() throws {
        let audio: [Float] = [0.1, 0.2, 0.3]
        let amplified = TensorUtils.applyVolumeGain(audio, gain: 2.0)
        
        XCTAssertEqual(amplified, [0.2, 0.4, 0.6])
    }
    
    func testFadeIn() throws {
        let audio: [Float] = [1.0, 1.0, 1.0, 1.0, 1.0]
        let faded = TensorUtils.fadeIn(audio, fadeLength: 3)
        
        // 第一个样本应该为 0（0/3 * 1.0）
        XCTAssertEqual(faded[0], 0.0)
        // 第二个样本应该是 1/3
        XCTAssertEqual(faded[1], 1.0 / 3.0, accuracy: 0.001)
        // 最后一个样本应该是原始值
        XCTAssertEqual(faded[4], 1.0)
    }
    
    func testMelToLinear() throws {
        let mel: [Float] = [1.0, 2.0, 3.0, 4.0]
        let linear = TensorUtils.melToLinear(mel)
        
        // 占位实现返回原始值
        XCTAssertEqual(mel, linear)
    }
}
