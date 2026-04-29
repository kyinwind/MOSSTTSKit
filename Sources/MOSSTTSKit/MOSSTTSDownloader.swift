/// MOSSTTSDownloader.swift
/// 
/// 模型下载器
/// 使用简单的 HTTP 请求下载模型

import Foundation

/// 模型下载器
public final class MOSSTTSDownloader: @unchecked Sendable {
    /// 配置
    private let config: MOSSTTSConfig
    
    /// 下载会话
    private var downloadSession: URLSession?
    
    /// 初始化
    public init(config: MOSSTTSConfig) {
        self.config = config
        self.downloadSession = URLSession(configuration: .default)
    }
    
    /// 下载模型
    /// - Parameter progressCallback: 进度回调
    /// - Returns: 下载的模型文件夹 URL
    public func download(
        progressCallback: ((MOSSProgress) -> Void)? = nil
    ) async throws -> URL {
        // 确定下载目录
        let destinationFolder: URL
        if let customBase = config.downloadBase {
            destinationFolder = customBase
        } else {
            // 使用缓存目录
            let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            destinationFolder = cacheDir.appendingPathComponent("MOSSTTSKit")
        }
        
        // 创建目录
        try FileManager.default.createDirectory(
            at: destinationFolder,
            withIntermediateDirectories: true
        )
        
        // 下载 TTS 模型
        let ttsFolder = destinationFolder.appendingPathComponent("TTS")
        try await downloadTTSModel(to: ttsFolder, progressCallback: progressCallback)
        
        // 下载 Audio Tokenizer 模型
        let audioTokenizerFolder = destinationFolder.appendingPathComponent("AudioTokenizer")
        try await downloadAudioTokenizerModel(to: audioTokenizerFolder, progressCallback: progressCallback)
        
        return destinationFolder
    }
    
    /// TTS 模型文件列表
    private var ttsModelFiles: [String] {
        [
            "moss_tts_prefill.onnx",
            "moss_tts_decode_step.onnx",
            "moss_tts_local_decoder.onnx",
            "moss_tts_local_cached_step.onnx",
            "moss_tts_local_fixed_sampled_frame.onnx",
            "moss_tts_global_shared.data",
            "moss_tts_local_shared.data",
            "tokenizer.model"
        ]
    }
    
    /// Audio Tokenizer 模型文件列表
    private var audioTokenizerFiles: [String] {
        [
            "moss_audio_tokenizer_encode.onnx",
            "moss_audio_tokenizer_encode.data",
            "moss_audio_tokenizer_decode_full.onnx",
            "moss_audio_tokenizer_decode_step.onnx",
            "moss_audio_tokenizer_decode_shared.data"
        ]
    }
    
    /// 下载 TTS 模型
    private func downloadTTSModel(
        to folder: URL,
        progressCallback: ((MOSSProgress) -> Void)?
    ) async throws {
        let repo = config.modelRepo
        let revision = config.downloadRevision ?? "main"
        
        // 创建目录
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        
        // 下载每个文件
        for fileName in ttsModelFiles {
            let fileURL = folder.appendingPathComponent(fileName)
            
            // 检查是否已存在
            if FileManager.default.fileExists(atPath: fileURL.path) {
                continue
            }
            
            let downloadURL = "https://huggingface.co/\(repo)/resolve/\(revision)/\(fileName)"
            
            progressCallback?(MOSSProgress(
                audioSamples: [],
                currentStep: ttsModelFiles.firstIndex(of: fileName) ?? 0,
                totalSteps: ttsModelFiles.count
            ))
            
            try await downloadFile(from: downloadURL, to: fileURL)
        }
    }
    
    /// 下载 Audio Tokenizer 模型
    private func downloadAudioTokenizerModel(
        to folder: URL,
        progressCallback: ((MOSSProgress) -> Void)?
    ) async throws {
        let repo = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
        
        // 创建目录
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        
        // 下载每个文件
        for fileName in audioTokenizerFiles {
            let fileURL = folder.appendingPathComponent(fileName)
            
            // 检查是否已存在
            if FileManager.default.fileExists(atPath: fileURL.path) {
                continue
            }
            
            let downloadURL = "https://huggingface.co/\(repo)/resolve/main/\(fileName)"
            
            progressCallback?(MOSSProgress(
                audioSamples: [],
                currentStep: audioTokenizerFiles.firstIndex(of: fileName) ?? 0,
                totalSteps: audioTokenizerFiles.count
            ))
            
            try await downloadFile(from: downloadURL, to: fileURL)
        }
    }
    
    /// 下载单个文件
    private func downloadFile(from urlString: String, to destination: URL) async throws {
        guard let url = URL(string: urlString) else {
            throw MOSSTTSError.downloadFailed("Invalid URL: \(urlString)")
        }
        
        // 创建下载任务
        let session = downloadSession ?? URLSession(configuration: .default)
        let (tempURL, response) = try await session.download(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw MOSSTTSError.downloadFailed("Invalid response")
        }
        
        guard (200...299).contains(httpResponse.statusCode) else {
            throw MOSSTTSError.downloadFailed("HTTP \(httpResponse.statusCode): \(urlString)")
        }
        
        // 移动到目标位置
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }
    
    /// 取消下载
    public func cancel() {
        downloadSession?.invalidateAndCancel()
        downloadSession = nil
    }
    
    /// 检查模型是否已下载
    public static func isModelDownloaded(at folder: URL) -> Bool {
        let prefillPath = folder
            .appendingPathComponent("TTS")
            .appendingPathComponent("moss_tts_prefill.onnx")
        return FileManager.default.fileExists(atPath: prefillPath.path)
    }
    
    /// 获取默认模型缓存路径
    public static var defaultCachePath: URL {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cacheDir.appendingPathComponent("MOSSTTSKit")
    }
}
