/// ConfigTests.swift
/// 
/// 配置类单元测试

import XCTest
@testable import MOSSTTSKit

final class ConfigTests: XCTestCase {
    
    // MARK: - MOSSTTSConfig Tests
    
    func testDefaultConfig() throws {
        let config = MOSSTTSConfig()
        
        XCTAssertEqual(config.modelVariant, .mossTTSNano100M)
        XCTAssertEqual(config.modelRepo, "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX")
        XCTAssertEqual(config.download, true)
        XCTAssertEqual(config.verbose, true)
    }
    
    func testConfigWithCustomSettings() throws {
        let config = MOSSTTSConfig(
            modelVariant: .mossTTSNano100M,
            modelFolder: URL(fileURLWithPath: "/path/to/models"),
            modelToken: "hf_test_token"
        )
        
        XCTAssertEqual(config.modelVariant, .mossTTSNano100M)
        XCTAssertNotNil(config.modelFolder)
        XCTAssertEqual(config.modelToken, "hf_test_token")
    }
    
    func testConfigModelDirectories() throws {
        let config = MOSSTTSConfig(
            modelFolder: URL(fileURLWithPath: "/path/to/models")
        )
        
        XCTAssertNotNil(config.ttsModelFolder)
        XCTAssertNotNil(config.tokenizerModelFolder)
        XCTAssertEqual(config.ttsModelFolder?.lastPathComponent, "tts")
        XCTAssertEqual(config.tokenizerModelFolder?.lastPathComponent, "audio_tokenizer")
    }
    
    func testConfigDownloadSettings() throws {
        let config = MOSSTTSConfig(
            download: false,
            prewarm: true,
            loadImmediately: false
        )
        
        XCTAssertFalse(config.download)
        XCTAssertTrue(config.prewarm)
        XCTAssertFalse(config.loadImmediately)
    }
    
    func testOptionsMaxGeneratedFrames() throws {
        let options = MOSSTTSOptions(maxLength: 100, maxGeneratedFrames: 12)
        
        XCTAssertEqual(options.maxLength, 100)
        XCTAssertEqual(options.maxGeneratedFrames, 12)
        XCTAssertTrue(options.validate().isEmpty)
    }

    func testOptionsDefaultUsesManifestFrameLimitWhenUnset() throws {
        let options = MOSSTTSOptions()

        XCTAssertNil(options.maxGeneratedFrames)
        XCTAssertTrue(options.validate().isEmpty)
    }
    
    func testOptionsRejectInvalidMaxGeneratedFrames() throws {
        let options = MOSSTTSOptions(maxGeneratedFrames: 0)
        
        XCTAssertEqual(options.validate().count, 1)
    }
    
    // MARK: - MOSSModelVariant Tests
    
    func testModelVariantProperties() throws {
        let variant = MOSSModelVariant.mossTTSNano100M
        
        XCTAssertEqual(variant.modelRepo, "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX")
        XCTAssertEqual(variant.tokenizerRepo, "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX")
        XCTAssertEqual(variant.displayName, "MOSS-TTS-Nano 100M")
        XCTAssertEqual(variant.rawValue, "100M")
    }
    
    func testModelPathsAvailabilityReportsMissingFiles() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        let paths = ModelPaths(
            ttsModelDir: root.appendingPathComponent("tts"),
            audioTokenizerDir: root.appendingPathComponent("audio_tokenizer")
        )
        
        let availability = paths.availability()
        
        XCTAssertFalse(availability.isComplete)
        XCTAssertEqual(availability.missingTTSFiles, ModelDownloader.ttsModelFiles)
        XCTAssertEqual(availability.missingAudioTokenizerFiles, ModelDownloader.audioTokenizerFiles)
        XCTAssertTrue(availability.missingFilesDescription.contains("moss_tts_prefill.onnx"))
        XCTAssertTrue(availability.missingFilesDescription.contains("moss_audio_tokenizer_encode.onnx"))
    }
    
    func testModelPathsAvailabilityForCompleteDirectories() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent(UUID().uuidString)
        let paths = ModelPaths(
            ttsModelDir: root.appendingPathComponent("tts"),
            audioTokenizerDir: root.appendingPathComponent("audio_tokenizer")
        )
        
        try FileManager.default.createDirectory(at: paths.ttsModelDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: paths.audioTokenizerDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        
        for file in ModelDownloader.ttsModelFiles {
            FileManager.default.createFile(atPath: paths.ttsFile(file).path, contents: Data())
        }
        
        for file in ModelDownloader.audioTokenizerFiles {
            FileManager.default.createFile(atPath: paths.tokenizerFile(file).path, contents: Data())
        }
        
        let availability = paths.availability()
        
        XCTAssertTrue(availability.isComplete)
        XCTAssertNoThrow(try paths.validate())
    }
    
    // MARK: - ONNXExecutionProvider Tests
    
    func testExecutionProviderCases() throws {
        XCTAssertEqual(ONNXExecutionProvider.cpu.rawValue, "CPU")
        XCTAssertEqual(ONNXExecutionProvider.coreml.rawValue, "CoreML")
        XCTAssertEqual(ONNXExecutionProvider.cuda.rawValue, "CUDA")
    }
    
    // MARK: - Audio Format Tests
    
    func testDefaultAudioFormat() throws {
        let format = MOSSAudioFormat.defaultFormat
        
        XCTAssertEqual(format.sampleRate, 48000)
        XCTAssertEqual(format.channels, 2)
        XCTAssertEqual(format.samplesPerFrame, 960)
    }
}
