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

// MARK: - Preset Speakers

extension MOSSSpeaker {
    
    /// 预设音色集合
    public static let presets: [MOSSSpeaker] = [
        MOSSSpeaker(name: "默认中文"),
        MOSSSpeaker(name: "默认英文"),
    ]
    
    /// 默认中文音色
    public static let defaultChinese = presets[0]
    
    /// 默认英文音色
    public static let defaultEnglish = presets[1]
}
