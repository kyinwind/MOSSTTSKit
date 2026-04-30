import Foundation

// MARK: - Speaker Extensions
// 
// MOSSSpeaker 已定义在 MOSSTTSModels.swift 中
// 此文件提供扩展方法

extension MOSSSpeaker {
    
    /// 获取 speaker embedding
    /// 如果有缓存的 embedding 直接返回，否则返回 nil
    public func getEmbedding() -> [Float]? {
        embedding
    }
    
    /// 检查是否有缓存的 embedding
    public var hasEmbedding: Bool {
        embedding != nil
    }
}
