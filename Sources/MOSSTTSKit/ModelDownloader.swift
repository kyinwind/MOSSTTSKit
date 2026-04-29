import Foundation

/// MOSS-TTS 模型下载器
///
/// 使用 HuggingFace 下载模型，自动管理缓存
public actor ModelDownloader {
    
    // MARK: - Constants
    
    /// 默认缓存目录
    public static let defaultCacheDir: URL = {
        let cache = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cache.appendingPathComponent("MOSSTTSKit/Models", isDirectory: true)
    }()
    
    /// HuggingFace 基础 URL
    private static let hfBaseURL = "https://huggingface.co"
    
    // MARK: - Types
    
    /// 下载进度
    public struct DownloadProgress: Sendable {
        public let bytesDownloaded: Int64
        public let totalBytes: Int64
        public let stage: Stage
        public let fileName: String
        
        public var progress: Double {
            guard totalBytes > 0 else { return 0 }
            return Double(bytesDownloaded) / Double(totalBytes)
        }
        
        public var description: String {
            let downloaded = formatBytes(bytesDownloaded)
            let total = formatBytes(totalBytes)
            let percent = Int(progress * 100)
            return "[\(stage.description)] \(fileName): \(downloaded)/\(total) (\(percent)%)"
        }
        
        private func formatBytes(_ bytes: Int64) -> String {
            let formatter = ByteCountFormatter()
            formatter.countStyle = .file
            return formatter.string(fromByteCount: bytes)
        }
        
        public enum Stage: Sendable {
            case connecting
            case downloading
            case verifying
            case done
            case failed(String)
            
            public var description: String {
                switch self {
                case .connecting: return "连接中"
                case .downloading: return "下载中"
                case .verifying: return "验证中"
                case .done: return "完成"
                case .failed(let msg): return "失败: \(msg)"
                }
            }
        }
    }
    
    /// 下载进度回调
    public typealias ProgressCallback = @Sendable (DownloadProgress) -> Void
    
    /// 错误类型
    public enum DownloadError: Error, LocalizedError {
        case modelNotFound(String)
        case networkError(String)
        case insufficientSpace
        case downloadFailed(String)
        case fileSystemError(String)
        case verificationFailed
        
        public var errorDescription: String? {
            switch self {
            case .modelNotFound(let repo):
                return "模型仓库不存在: \(repo)"
            case .networkError(let msg):
                return "网络错误: \(msg)"
            case .insufficientSpace:
                return "磁盘空间不足"
            case .downloadFailed(let msg):
                return "下载失败: \(msg)"
            case .fileSystemError(let msg):
                return "文件系统错误: \(msg)"
            case .verificationFailed:
                return "文件验证失败"
            }
        }
    }
    
    // MARK: - Properties
    
    /// 缓存目录
    private let cacheDir: URL
    
    /// 正在下载的任务
    private var currentDownloads: [String: Task<URL, Error>] = [:]
    
    // MARK: - MOSS-TTS-100M-ONNX 必需文件
    
    /// TTS 模型文件列表
    public static let ttsModelFiles = [
        "moss_tts_prefill.onnx",
        "moss_tts_decode_step.onnx",
        "moss_tts_local_decoder.onnx",
        "moss_tts_local_cached_step.onnx",
        "moss_tts_local_fixed_sampled_frame.onnx",
        "moss_tts_global_shared.data",
        "moss_tts_local_shared.data",
        "tokenizer.model",
        "tts_browser_onnx_meta.json",
        "browser_poc_manifest.json"
    ]
    
    /// Audio Tokenizer 文件列表
    public static let audioTokenizerFiles = [
        "moss_audio_tokenizer_encode.onnx",
        "moss_audio_tokenizer_encode.data",
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_step.onnx",
        "moss_audio_tokenizer_decode_shared.data",
        "codec_browser_onnx_meta.json"
    ]
    
    // MARK: - Initialization
    
    public init(cacheDir: URL? = nil) {
        self.cacheDir = cacheDir ?? Self.defaultCacheDir
    }
    
    // MARK: - Public API
    
    /// 下载所有必需模型（透明下载入口）
    /// - Parameters:
    ///   - variant: 模型变体
    ///   - progressCallback: 进度回调
    /// - Returns: 模型路径信息
    public func downloadModels(
        variant: MOSSModelVariant = .mossTTSNano100M,
        progressCallback: ProgressCallback? = nil
    ) async throws -> ModelPaths {
        try ensureCacheDirectoryExists()
        
        let ttsDir = ttsModelDir(for: variant)
        let tokenizerDir = tokenizerModelDir(for: variant)
        
        // 下载 TTS 模型
        for file in Self.ttsModelFiles {
            let localPath = ttsDir.appendingPathComponent(file)
            if !FileManager.default.fileExists(atPath: localPath.path) {
                progressCallback?(DownloadProgress(
                    bytesDownloaded: 0,
                    totalBytes: 0,
                    stage: .connecting,
                    fileName: file
                ))
                
                try await downloadFile(
                    repo: variant.modelRepo,
                    fileName: file,
                    to: localPath,
                    progressCallback: progressCallback
                )
            } else {
                progressCallback?(DownloadProgress(
                    bytesDownloaded: 1,
                    totalBytes: 1,
                    stage: .done,
                    fileName: file
                ))
            }
        }
        
        // 下载 Audio Tokenizer
        for file in Self.audioTokenizerFiles {
            let localPath = tokenizerDir.appendingPathComponent(file)
            if !FileManager.default.fileExists(atPath: localPath.path) {
                progressCallback?(DownloadProgress(
                    bytesDownloaded: 0,
                    totalBytes: 0,
                    stage: .connecting,
                    fileName: file
                ))
                
                try await downloadFile(
                    repo: variant.tokenizerRepo,
                    fileName: file,
                    to: localPath,
                    progressCallback: progressCallback
                )
            } else {
                progressCallback?(DownloadProgress(
                    bytesDownloaded: 1,
                    totalBytes: 1,
                    stage: .done,
                    fileName: file
                ))
            }
        }
        
        return ModelPaths(
            ttsModelDir: ttsDir,
            audioTokenizerDir: tokenizerDir
        )
    }
    
    /// 检查模型是否已缓存
    public func isModelCached(variant: MOSSModelVariant = .mossTTSNano100M) -> Bool {
        modelAvailability(for: variant).isComplete
    }
    
    /// 检查当前缓存目录中的模型完整性。
    public func modelAvailability(variant: MOSSModelVariant = .mossTTSNano100M) -> ModelAvailability {
        modelAvailability(for: variant)
    }
    
    /// 获取 TTS 模型目录
    public func ttsModelDir(for variant: MOSSModelVariant) -> URL {
        cacheDir.appendingPathComponent("MOSS-TTS-Nano-100M-ONNX")
    }
    
    /// 获取 Audio Tokenizer 目录
    public func tokenizerModelDir(for variant: MOSSModelVariant) -> URL {
        cacheDir.appendingPathComponent("MOSS-Audio-Tokenizer-Nano-ONNX")
    }
    
    /// 清理缓存
    public func clearCache(for variant: MOSSModelVariant? = nil) throws {
        if let variant = variant {
            let ttsDir = ttsModelDir(for: variant)
            let tokenizerDir = tokenizerModelDir(for: variant)
            if FileManager.default.fileExists(atPath: ttsDir.path) {
                try FileManager.default.removeItem(at: ttsDir)
            }
            if FileManager.default.fileExists(atPath: tokenizerDir.path) {
                try FileManager.default.removeItem(at: tokenizerDir)
            }
        } else {
            if FileManager.default.fileExists(atPath: cacheDir.path) {
                try FileManager.default.removeItem(at: cacheDir)
            }
        }
    }
    
    /// 获取缓存大小
    public func cacheSize() -> Int64 {
        guard FileManager.default.fileExists(atPath: cacheDir.path),
              let enumerator = FileManager.default.enumerator(
                at: cacheDir,
                includingPropertiesForKeys: [.fileSizeKey],
                options: [.skipsHiddenFiles]
              ) else { return 0 }
        
        var totalSize: Int64 = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                totalSize += Int64(size)
            }
        }
        return totalSize
    }
    
    /// 格式化缓存大小
    public var formattedCacheSize: String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: cacheSize())
    }
    
    /// 取消所有下载
    public func cancelAllDownloads() {
        for (_, task) in currentDownloads {
            task.cancel()
        }
        currentDownloads.removeAll()
    }
    
    // MARK: - Private
    
    private func ensureCacheDirectoryExists() throws {
        if !FileManager.default.fileExists(atPath: cacheDir.path) {
            try FileManager.default.createDirectory(
                at: cacheDir,
                withIntermediateDirectories: true,
                attributes: nil
            )
        }
    }
    
    private func modelAvailability(for variant: MOSSModelVariant) -> ModelAvailability {
        ModelPaths(
            ttsModelDir: ttsModelDir(for: variant),
            audioTokenizerDir: tokenizerModelDir(for: variant)
        )
        .availability()
    }
    
    private func downloadFile(
        repo: String,
        fileName: String,
        to destination: URL,
        progressCallback: ProgressCallback?
    ) async throws {
        let urlString = "\(Self.hfBaseURL)/\(repo)/resolve/main/\(fileName)"
        guard let url = URL(string: urlString) else {
            throw DownloadError.modelNotFound(repo)
        }
        
        let downloadTask = Task<URL, Error> {
            try await downloadFileWithResume(
                from: url,
                repo: repo,
                fileName: fileName,
                to: destination,
                progressCallback: progressCallback
            )
        }
        
        currentDownloads[fileName] = downloadTask
        
        do {
            _ = try await downloadTask.value
            currentDownloads.removeValue(forKey: fileName)
            
            progressCallback?(DownloadProgress(
                bytesDownloaded: 1,
                totalBytes: 1,
                stage: .done,
                fileName: fileName
            ))
        } catch {
            currentDownloads.removeValue(forKey: fileName)
            throw error
        }
    }
    
    private func downloadFileWithResume(
        from url: URL,
        repo: String,
        fileName: String,
        to destination: URL,
        progressCallback: ProgressCallback?
    ) async throws -> URL {
        let fileManager = FileManager.default
        let parentDir = destination.deletingLastPathComponent()
        if !fileManager.fileExists(atPath: parentDir.path) {
            try fileManager.createDirectory(at: parentDir, withIntermediateDirectories: true)
        }
        
        let partialURL = destination.appendingPathExtension("partial")
        var resumedBytes = fileSize(at: partialURL)
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        if resumedBytes > 0 {
            request.setValue("bytes=\(resumedBytes)-", forHTTPHeaderField: "Range")
        }
        
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw DownloadError.networkError("无效的响应")
        }
        
        if httpResponse.statusCode == 404 {
            throw DownloadError.modelNotFound("\(repo)/\(fileName)")
        }
        
        guard (200...299).contains(httpResponse.statusCode) else {
            throw DownloadError.downloadFailed("HTTP \(httpResponse.statusCode)")
        }
        
        if resumedBytes > 0 && httpResponse.statusCode != 206 {
            try? fileManager.removeItem(at: partialURL)
            resumedBytes = 0
        }
        
        if !fileManager.fileExists(atPath: partialURL.path) {
            fileManager.createFile(atPath: partialURL.path, contents: nil)
        }
        
        let contentLength = httpResponse.value(forHTTPHeaderField: "Content-Length").flatMap(Int64.init) ?? -1
        let totalBytes = contentLength > 0 ? contentLength + resumedBytes : 0
        var downloadedBytes = resumedBytes
        var lastReportedBytes = downloadedBytes
        var buffer: [UInt8] = []
        buffer.reserveCapacity(64 * 1024)
        
        let handle = try FileHandle(forWritingTo: partialURL)
        defer { try? handle.close() }
        try handle.seekToEnd()
        
        progressCallback?(DownloadProgress(
            bytesDownloaded: downloadedBytes,
            totalBytes: totalBytes,
            stage: .downloading,
            fileName: fileName
        ))
        
        do {
            for try await byte in bytes {
                try Task.checkCancellation()
                buffer.append(byte)
                
                if buffer.count >= 64 * 1024 {
                    try handle.write(contentsOf: Data(buffer))
                    downloadedBytes += Int64(buffer.count)
                    buffer.removeAll(keepingCapacity: true)
                    
                    if downloadedBytes - lastReportedBytes >= 512 * 1024 {
                        lastReportedBytes = downloadedBytes
                        progressCallback?(DownloadProgress(
                            bytesDownloaded: downloadedBytes,
                            totalBytes: totalBytes,
                            stage: .downloading,
                            fileName: fileName
                        ))
                    }
                }
            }
            
            if !buffer.isEmpty {
                try handle.write(contentsOf: Data(buffer))
                downloadedBytes += Int64(buffer.count)
            }
        } catch {
            progressCallback?(DownloadProgress(
                bytesDownloaded: downloadedBytes,
                totalBytes: totalBytes,
                stage: .failed(error.localizedDescription),
                fileName: fileName
            ))
            throw error
        }
        
        progressCallback?(DownloadProgress(
            bytesDownloaded: downloadedBytes,
            totalBytes: totalBytes,
            stage: .verifying,
            fileName: fileName
        ))
        
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: partialURL, to: destination)
        
        guard fileManager.fileExists(atPath: destination.path), fileSize(at: destination) > 0 else {
            throw DownloadError.verificationFailed
        }
        
        return destination
    }
    
    private func fileSize(at url: URL) -> Int64 {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize else {
            return 0
        }
        return Int64(size)
    }
}

/// 模型路径信息
public struct ModelPaths: Sendable {
    public let ttsModelDir: URL
    public let audioTokenizerDir: URL
    
    public init(ttsModelDir: URL, audioTokenizerDir: URL) {
        self.ttsModelDir = ttsModelDir
        self.audioTokenizerDir = audioTokenizerDir
    }
    
    /// 获取 TTS 模型特定文件路径
    public func ttsFile(_ name: String) -> URL {
        ttsModelDir.appendingPathComponent(name)
    }
    
    /// 获取 Audio Tokenizer 特定文件路径
    public func tokenizerFile(_ name: String) -> URL {
        audioTokenizerDir.appendingPathComponent(name)
    }
    
    /// 检查当前模型目录是否包含 MOSSTTSKit 需要的全部文件。
    public func availability(fileManager: FileManager = .default) -> ModelAvailability {
        let missingTTSFiles = ModelDownloader.ttsModelFiles.filter { file in
            !fileManager.fileExists(atPath: ttsFile(file).path)
        }
        
        let missingAudioTokenizerFiles = ModelDownloader.audioTokenizerFiles.filter { file in
            !fileManager.fileExists(atPath: tokenizerFile(file).path)
        }
        
        return ModelAvailability(
            ttsModelDir: ttsModelDir,
            audioTokenizerDir: audioTokenizerDir,
            missingTTSFiles: missingTTSFiles,
            missingAudioTokenizerFiles: missingAudioTokenizerFiles
        )
    }
    
    /// 如果模型目录不完整则抛出带缺失文件列表的错误。
    public func validate(fileManager: FileManager = .default) throws {
        let availability = availability(fileManager: fileManager)
        guard availability.isComplete else {
            throw MOSSTTSError.modelNotFound(availability.missingFilesDescription)
        }
    }
}

/// 模型目录完整性检查结果。
public struct ModelAvailability: Sendable, Equatable {
    public let ttsModelDir: URL
    public let audioTokenizerDir: URL
    public let missingTTSFiles: [String]
    public let missingAudioTokenizerFiles: [String]
    
    public var isComplete: Bool {
        missingTTSFiles.isEmpty && missingAudioTokenizerFiles.isEmpty
    }
    
    public var missingFilesDescription: String {
        var lines: [String] = []
        
        if missingTTSFiles.isEmpty {
            lines.append("TTS model files: complete")
        } else {
            lines.append("Missing TTS model files in \(ttsModelDir.path): \(missingTTSFiles.joined(separator: ", "))")
        }
        
        if missingAudioTokenizerFiles.isEmpty {
            lines.append("Audio tokenizer files: complete")
        } else {
            lines.append("Missing audio tokenizer files in \(audioTokenizerDir.path): \(missingAudioTokenizerFiles.joined(separator: ", "))")
        }
        
        return lines.joined(separator: "\n")
    }
    
    public init(
        ttsModelDir: URL,
        audioTokenizerDir: URL,
        missingTTSFiles: [String],
        missingAudioTokenizerFiles: [String]
    ) {
        self.ttsModelDir = ttsModelDir
        self.audioTokenizerDir = audioTokenizerDir
        self.missingTTSFiles = missingTTSFiles
        self.missingAudioTokenizerFiles = missingAudioTokenizerFiles
    }
}
