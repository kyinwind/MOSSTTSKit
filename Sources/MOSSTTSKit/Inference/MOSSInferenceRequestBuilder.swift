import Foundation

/// Parsed `browser_poc_manifest.json` used by the ONNX CPU runtime.
public struct MOSSBrowserManifest: Sendable, Decodable {
    public let modelFiles: ModelFiles
    public let ttsConfig: TTSConfig
    public let promptTemplates: PromptTemplates
    public let generationDefaults: GenerationDefaults
    public let builtinVoices: [BuiltinVoice]
    
    public static func load(from url: URL) throws -> MOSSBrowserManifest {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let data = try Data(contentsOf: url)
        return try decoder.decode(MOSSBrowserManifest.self, from: data)
    }
    
    public static func find(in ttsModelDir: URL) throws -> MOSSBrowserManifest? {
        let candidates = [
            ttsModelDir.appendingPathComponent("browser_poc_manifest.json"),
            ttsModelDir.deletingLastPathComponent()
                .appendingPathComponent("browser_poc_manifest.json"),
            ttsModelDir.deletingLastPathComponent()
                .appendingPathComponent("MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json"),
        ]
        
        for candidate in candidates where FileManager.default.fileExists(atPath: candidate.path) {
            return try load(from: candidate)
        }
        
        return nil
    }
    
    public struct ModelFiles: Sendable, Decodable {
        public let ttsMeta: String
        public let codecMeta: String
        public let tokenizerModel: String?
    }
    
    public struct TTSConfig: Sendable, Decodable {
        public let nVq: Int
        public let audioPadTokenId: Int
        public let audioStartTokenId: Int
        public let audioEndTokenId: Int
        public let audioUserSlotTokenId: Int
        public let audioAssistantSlotTokenId: Int
    }
    
    public struct PromptTemplates: Sendable, Decodable {
        public let userPromptPrefixTokenIds: [Int]
        public let userPromptAfterReferenceTokenIds: [Int]
        public let assistantPromptPrefixTokenIds: [Int]
    }
    
    public struct GenerationDefaults: Sendable, Decodable {
        public let maxNewFrames: Int?
        public let doSample: Bool?
        public let sampleMode: String?
        public let textTemperature: Float?
        public let textTopK: Int?
        public let textTopP: Float?
        public let audioTemperature: Float?
        public let audioTopK: Int?
        public let audioTopP: Float?
        public let audioRepetitionPenalty: Float?
    }
    
    public struct BuiltinVoice: Sendable, Decodable {
        public let voice: String
        public let displayName: String?
        public let group: String?
        public let audioFile: String?
        public let promptAudioCodes: [[Int32]]
        
        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.voice = try container.decode(String.self, forKey: .voice)
            self.displayName = try container.decodeIfPresent(String.self, forKey: .displayName)
            self.group = try container.decodeIfPresent(String.self, forKey: .group)
            self.audioFile = try container.decodeIfPresent(String.self, forKey: .audioFile)
            self.promptAudioCodes = try container.decodeIfPresent([[Int32]].self, forKey: .promptAudioCodes) ?? []
        }
        
        private enum CodingKeys: String, CodingKey {
            case voice
            case displayName
            case group
            case audioFile
            case promptAudioCodes
        }
    }
}

/// Builds the row-major TTS request expected by the MOSS-TTS-Nano ONNX prefill graph.
public struct MOSSInferenceRequestBuilder: Sendable {
    public struct RequestRows: Sendable, Equatable {
        public let inputIds: [[Int32]]
        public let attentionMask: [[Int32]]
        
        public init(inputIds: [[Int32]], attentionMask: [[Int32]]) {
            self.inputIds = inputIds
            self.attentionMask = attentionMask
        }
    }
    
    public let manifest: MOSSBrowserManifest
    
    public init(manifest: MOSSBrowserManifest) {
        self.manifest = manifest
    }
    
    public func buildVoiceCloneRequestRows(
        promptAudioCodes: [[Int32]],
        textTokenIds: [Int32]
    ) -> RequestRows {
        let config = manifest.ttsConfig
        
        let prefixTextTokenIds =
            manifest.promptTemplates.userPromptPrefixTokenIds
            + [config.audioStartTokenId]
        
        let suffixTextTokenIds =
            [config.audioEndTokenId]
            + manifest.promptTemplates.userPromptAfterReferenceTokenIds
            + textTokenIds.map(Int.init)
            + manifest.promptTemplates.assistantPromptPrefixTokenIds
            + [config.audioStartTokenId]
        
        var rows: [[Int32]] = []
        rows.reserveCapacity(prefixTextTokenIds.count + promptAudioCodes.count + suffixTextTokenIds.count)
        rows.append(contentsOf: buildTextRows(tokenIds: prefixTextTokenIds))
        rows.append(contentsOf: buildAudioPrefixRows(promptAudioCodes: promptAudioCodes))
        rows.append(contentsOf: buildTextRows(tokenIds: suffixTextTokenIds))
        
        return RequestRows(
            inputIds: rows,
            attentionMask: [[Int32](repeating: 1, count: rows.count)]
        )
    }
    
    public func buildTextRows(tokenIds: [Int]) -> [[Int32]] {
        let rowWidth = manifest.ttsConfig.nVq + 1
        let audioPad = Int32(manifest.ttsConfig.audioPadTokenId)
        
        return tokenIds.map { tokenId in
            var row = [Int32](repeating: audioPad, count: rowWidth)
            row[0] = Int32(tokenId)
            return row
        }
    }
    
    public func buildAudioPrefixRows(
        promptAudioCodes: [[Int32]],
        slotTokenId: Int? = nil
    ) -> [[Int32]] {
        let config = manifest.ttsConfig
        let rowWidth = config.nVq + 1
        let audioPad = Int32(config.audioPadTokenId)
        let resolvedSlotTokenId = Int32(slotTokenId ?? config.audioUserSlotTokenId)
        
        return promptAudioCodes.map { codeRow in
            var row = [Int32](repeating: audioPad, count: rowWidth)
            row[0] = resolvedSlotTokenId
            
            for index in 0..<min(codeRow.count, config.nVq) {
                row[index + 1] = codeRow[index]
            }
            
            return row
        }
    }
}
