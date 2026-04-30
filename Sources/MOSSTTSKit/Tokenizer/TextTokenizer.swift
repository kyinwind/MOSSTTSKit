/// TextTokenizer.swift
/// 
/// 文本 tokenizer 实现
/// MOSS-TTS-Nano 使用 SentencePiece tokenizer
/// 
/// 文档：
/// - https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX
/// - tokenizer.model: SentencePiece tokenizer

import Foundation
import Tokenizers

/// 文本编码结果
public struct TextEncodingResult: Sendable {
    /// Token IDs
    public let ids: [Int32]
    
    /// Token 数量
    public var length: Int { ids.count }
    
    /// 原始文本
    public let text: String
}

/// 文本 tokenizer 接口
public protocol MOSSTextTokenizer: Sendable {
    /// 对文本进行编码
    /// - Parameter text: 输入文本
    /// - Returns: 编码结果
    func encode(_ text: String) async throws -> TextEncodingResult
    
    /// 对 token IDs 进行解码
    /// - Parameter ids: Token IDs
    /// - Returns: 解码后的文本
    func decode(_ ids: [Int32]) async throws -> String
    
    /// 获取词汇表大小
    var vocabularySize: Int { get }
    
    /// 获取特殊 token
    var bosTokenId: Int32? { get }
    var eosTokenId: Int32? { get }
    var padTokenId: Int32? { get }
}

/// HuggingFace AutoTokenizer 包装器
public final class HuggingFaceTextTokenizer: MOSSTextTokenizer {
    private let tokenizer: any Tokenizer
    
    /// 词汇表大小
    /// 注意：swift-transformers 的 Tokenizer 协议没有 vocabularySize 属性
    /// 我们通过尝试解码一个大的 ID 来估计
    public var vocabularySize: Int {
        // 临时实现：返回常见大小
        // TODO: 通过 tokenizer.convertIdToToken() 探测
        32000
    }
    
    /// 特殊 token IDs
    public var bosTokenId: Int32? {
        guard let id = tokenizer.bosTokenId else { return nil }
        return Int32(id)
    }
    
    public var eosTokenId: Int32? {
        guard let id = tokenizer.eosTokenId else { return nil }
        return Int32(id)
    }
    
    public var padTokenId: Int32? {
        // swift-transformers Tokenizer 协议没有 padTokenId
        // 尝试从配置获取
        nil
    }
    
    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }
    
    /// 从 HuggingFace Hub 加载
    public static func fromPretrained(
        modelName: String = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
    ) async throws -> HuggingFaceTextTokenizer {
        let tokenizer = try await AutoTokenizer.from(pretrained: modelName)
        return HuggingFaceTextTokenizer(tokenizer: tokenizer)
    }
    
    /// 从本地文件加载
    public static func fromLocal(
        tokenizerPath: String
    ) async throws -> HuggingFaceTextTokenizer {
        guard FileManager.default.fileExists(atPath: tokenizerPath) else {
            throw MOSSTTSError.tokenizerNotFound("Tokenizer not found at: \(tokenizerPath)")
        }
        
        let url = URL(fileURLWithPath: tokenizerPath)
        let modelFolder: URL
        if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
            modelFolder = url
        } else {
            modelFolder = url.deletingLastPathComponent()
        }
        
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder)
        return HuggingFaceTextTokenizer(tokenizer: tokenizer)
    }
    
    public func encode(_ text: String) async throws -> TextEncodingResult {
        // swift-transformers 的 encode 返回 [Int]
        let ids = tokenizer.encode(text: text)
        return TextEncodingResult(
            ids: ids.map { Int32($0) },
            text: text
        )
    }
    
    public func decode(_ ids: [Int32]) async throws -> String {
        // swift-transformers 的 decode 需要 [Int]
        let stringResult = tokenizer.decode(tokens: ids.map { Int($0) })
        return stringResult
    }
}

/// SentencePiece tokenizer 实现
/// 
/// 注意：swift-transformers 目前直接支持 SentencePiece 文件
/// 如果遇到兼容性问题，可考虑：
/// 1. 使用 HuggingFace tokenizer.json
/// 2. 或通过 Python 子进程调用
public final class SentencePieceTokenizer: @unchecked Sendable, MOSSTextTokenizer {
    private var spmTokenizer: HuggingFaceTextTokenizer?
    private var byteFallbackTokenizer: ByteFallbackTextTokenizer?
    private let tokenizerPath: String?
    private let modelName: String
    
    public var vocabularySize: Int {
        spmTokenizer?.vocabularySize ?? byteFallbackTokenizer?.vocabularySize ?? 0
    }
    
    public var bosTokenId: Int32? {
        spmTokenizer?.bosTokenId ?? byteFallbackTokenizer?.bosTokenId
    }
    
    public var eosTokenId: Int32? {
        spmTokenizer?.eosTokenId ?? byteFallbackTokenizer?.eosTokenId
    }
    
    public var padTokenId: Int32? {
        spmTokenizer?.padTokenId ?? byteFallbackTokenizer?.padTokenId
    }
    
    /// 初始化
    /// - Parameters:
    ///   - tokenizerPath: 本地 tokenizer.model 路径
    ///   - modelName: HuggingFace 模型名（用于下载）
    public init(tokenizerPath: String? = nil, modelName: String = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX") {
        self.tokenizerPath = tokenizerPath
        self.modelName = modelName
    }
    
    /// 加载 tokenizer
    public func load() async throws {
        if let path = tokenizerPath, FileManager.default.fileExists(atPath: path) {
            do {
                spmTokenizer = try await HuggingFaceTextTokenizer.fromLocal(tokenizerPath: path)
                byteFallbackTokenizer = nil
            } catch {
                if let bundledTokenizer = try await Self.loadBundledTokenizer() {
                    spmTokenizer = bundledTokenizer
                    byteFallbackTokenizer = nil
                } else {
                    // The ONNX release currently ships a SentencePiece `tokenizer.model` without
                    // tokenizer.json/tokenizer_config.json. If the packaged fallback tokenizer
                    // resources are unavailable, use the model's byte-token range as a last resort.
                    spmTokenizer = nil
                    byteFallbackTokenizer = ByteFallbackTextTokenizer()
                }
            }
        } else {
            spmTokenizer = try await HuggingFaceTextTokenizer.fromPretrained(modelName: modelName)
            byteFallbackTokenizer = nil
        }
    }
    
    public func encode(_ text: String) async throws -> TextEncodingResult {
        if let tokenizer = spmTokenizer {
            return try await tokenizer.encode(text)
        }
        if let tokenizer = byteFallbackTokenizer {
            return try await tokenizer.encode(text)
        }
        
        throw MOSSTTSError.tokenizerNotFound("Tokenizer not loaded")
    }
    
    public func decode(_ ids: [Int32]) async throws -> String {
        if let tokenizer = spmTokenizer {
            return try await tokenizer.decode(ids)
        }
        if let tokenizer = byteFallbackTokenizer {
            return try await tokenizer.decode(ids)
        }
        
        throw MOSSTTSError.tokenizerNotFound("Tokenizer not loaded")
    }

    private static func loadBundledTokenizer() async throws -> HuggingFaceTextTokenizer? {
        guard let resourceURL = Bundle.module.resourceURL else {
            return nil
        }
        let tokenizerJSONPath = resourceURL.appendingPathComponent("tokenizer.json").path
        let tokenizerConfigPath = resourceURL.appendingPathComponent("tokenizer_config.json").path
        guard FileManager.default.fileExists(atPath: tokenizerJSONPath),
              FileManager.default.fileExists(atPath: tokenizerConfigPath) else {
            return nil
        }
        return try await HuggingFaceTextTokenizer.fromLocal(tokenizerPath: resourceURL.path)
    }
}

private final class ByteFallbackTextTokenizer: @unchecked Sendable, MOSSTextTokenizer {
    private let byteTokenBase: Int32 = 14
    
    var vocabularySize: Int { 16_384 }
    var bosTokenId: Int32? { nil }
    var eosTokenId: Int32? { 1 }
    var padTokenId: Int32? { 2 }
    
    func encode(_ text: String) async throws -> TextEncodingResult {
        let ids = text.utf8.map { byteTokenBase + Int32($0) }
        return TextEncodingResult(ids: ids, text: text)
    }
    
    func decode(_ ids: [Int32]) async throws -> String {
        let bytes = ids.compactMap { id -> UInt8? in
            let byte = id - byteTokenBase
            guard byte >= 0, byte <= 255 else {
                return nil
            }
            return UInt8(byte)
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
