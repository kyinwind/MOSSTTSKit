// TensorUtils.swift
// 张量工具函数

import Foundation
import Accelerate

/// 张量工具
public enum TensorUtils {
    // MARK: - 类型转换
    
    /// Float 数组转 Data
    public static func floatsToData(_ values: [Float]) -> Data {
        return values.withUnsafeBytes { Data($0) }
    }
    
    /// Int32 数组转 Data
    public static func int32sToData(_ values: [Int32]) -> Data {
        return values.withUnsafeBytes { Data($0) }
    }
    
    /// Int64 数组转 Data
    public static func int64sToData(_ values: [Int64]) -> Data {
        return values.withUnsafeBytes { Data($0) }
    }
    
    /// Data 转 Float 数组
    public static func dataToFloats(_ data: Data) -> [Float] {
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }
    
    /// Data 转 Int32 数组
    public static func dataToInt32s(_ data: Data) -> [Int32] {
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int32.self))
        }
    }
    
    /// Data 转 Int64 数组
    public static func dataToInt64s(_ data: Data) -> [Int64] {
        return data.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int64.self))
        }
    }
    
    // MARK: - 形状操作
    
    /// 计算扁平化数组的大小
    public static func flattenedSize(of shape: [Int]) -> Int {
        return shape.reduce(1, *)
    }
    
    /// 重塑数组
    public static func reshape<T>(_ array: [T], to shape: [Int]) -> [[T]]? {
        let flatSize = flattenedSize(of: shape)
        guard flatSize == array.count else { return nil }
        
        var index = 0
        
        func reshapeRecursive(_ remainingShape: [Int]) -> [[T]] {
            if remainingShape.count == 1 {
                let count = remainingShape[0]
                let slice = Array(array[index..<index + count])
                index += count
                return [slice]
            } else {
                let count = remainingShape[0]
                let subShape = Array(remainingShape.dropFirst())
                var slices: [[T]] = []
                for _ in 0..<count {
                    slices.append(contentsOf: reshapeRecursive(subShape))
                }
                return slices
            }
        }
        
        return reshapeRecursive(shape)
    }
    
    // MARK: - 数值操作
    
    /// Top-K 采样
    public static func topK(_ values: [Float], k: Int) -> [(value: Float, index: Int)] {
        guard k > 0 && k <= values.count else { return [] }
        
        let indexed = values.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }
        
        return sorted.prefix(k).map { (value: $0.1, index: $0.0) }
    }
    
    /// 温度采样
    public static func temperatureSample(_ logits: [Float], temperature: Float) -> [Float] {
        guard temperature > 0 else { return logits }
        
        // 归一化
        let maxLogit = logits.max() ?? 0
        let expValues = logits.map { exp(Float($0 - maxLogit) / temperature) }
        let sum = expValues.reduce(0, +)
        
        return expValues.map { $0 / sum }
    }
    
    /// 从概率分布采样
    public static func sampleFromDistribution(_ probabilities: [Float]) -> Int {
        let r = Float.random(in: 0..<1)
        var cumulative: Float = 0
        
        for (index, prob) in probabilities.enumerated() {
            cumulative += prob
            if r <= cumulative {
                return index
            }
        }
        
        return probabilities.count - 1
    }
    
    /// 应用重复惩罚
    public static func applyRepetitionPenalty(_ logits: [Float], previousTokens: [Int], penalty: Float) -> [Float] {
        guard penalty != 1.0 else { return logits }
        
        var penalizedLogits = logits
        for token in previousTokens.suffix(100) {  // 只看最近的 tokens
            if token < penalizedLogits.count {
                if penalizedLogits[token] > 0 {
                    penalizedLogits[token] /= penalty
                } else {
                    penalizedLogits[token] *= penalty
                }
            }
        }
        
        return penalizedLogits
    }
    
    // MARK: - 音频处理
    
    /// 归一化音频（-1.0 ~ 1.0）
    public static func normalizeAudio(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return samples }
        
        let maxAbs = samples.map { abs($0) }.max() ?? 1.0
        if maxAbs > 1.0 {
            return samples.map { $0 / maxAbs }
        }
        return samples
    }
    
    /// 应用音量增益
    public static func applyVolumeGain(_ samples: [Float], gain: Float) -> [Float] {
        return samples.map { $0 * gain }
    }
    
    /// 淡入淡出
    public static func fadeIn(_ samples: [Float], fadeLength: Int) -> [Float] {
        guard fadeLength > 0 && fadeLength < samples.count else { return samples }
        
        var result = samples
        for i in 0..<fadeLength {
            let factor = Float(i) / Float(fadeLength)
            result[i] *= factor
        }
        return result
    }
    
    /// 交叉淡入淡出
    public static func crossfade(_ samples1: [Float], _ samples2: [Float], fadeLength: Int) -> [Float] {
        guard fadeLength > 0 else {
            return samples1 + samples2
        }
        
        var result = samples1
        
        // 淡出第一个音频
        for i in 0..<fadeLength {
            let index = samples1.count - fadeLength + i
            if index < result.count {
                let factor = Float(fadeLength - i) / Float(fadeLength)
                result[index] *= factor
            }
        }
        
        // 淡入第二个音频并拼接
        var fadeInSamples: [Float] = []
        for i in 0..<fadeLength {
            let factor = Float(i) / Float(fadeLength)
            if i < samples2.count {
                fadeInSamples.append(samples2[i] * factor)
            }
        }
        
        return result + fadeInSamples + Array(samples2.dropFirst(fadeLength))
    }
    
    // MARK: - Mel 频谱处理
    
    /// 简单的 mel 到线性频谱转换（占位）
    public static func melToLinear(_ mel: [Float]) -> [Float] {
        // TODO: 实现真正的 mel 逆变换
        // 这里返回原始值作为占位
        return mel
    }
}
