/// TokenizerTests.swift
/// 
/// Tokenizer 单元测试

import XCTest
@testable import MOSSTTSKit

final class TokenizerTests: XCTestCase {
    
    // MARK: - Model Info Tests
    
    func testMOSSModelVariantRepos() throws {
        let variant = MOSSModelVariant.mossTTSNano100M
        
        XCTAssertEqual(variant.modelRepo, "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX")
        XCTAssertEqual(variant.tokenizerRepo, "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX")
    }
    
    // MARK: - Audio Tokenizer Tests
    
    func testAudioTokenizerONNXCreation() throws {
        let audioTokenizer = AudioTokenizerONNX(
            encodeModelPath: "/path/to/encode.onnx",
            decodeModelPath: "/path/to/decode.onnx"
        )
        
        XCTAssertNotNil(audioTokenizer)
        XCTAssertEqual(audioTokenizer.encodeModelPath, "/path/to/encode.onnx")
        XCTAssertEqual(audioTokenizer.decodeModelPath, "/path/to/decode.onnx")
    }
    
    func testAudioTokenizerONNXLoad() async throws {
        let audioTokenizer = AudioTokenizerONNX(
            encodeModelPath: "/path/to/encode.onnx",
            decodeModelPath: "/path/to/decode.onnx"
        )
        
        // 由于模型不存在，这里应该抛出错误
        do {
            try await audioTokenizer.load()
            XCTFail("Expected error when loading non-existent model")
        } catch {
            // 预期行为
            XCTAssertTrue(true)
        }
    }
}
