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
        XCTAssertNil(audioTokenizer.decodeStepModelPath)
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

    func testSentencePieceTokenizerUsesBundledTokenizerResources() async throws {
        let tokenizer = SentencePieceTokenizer(
            tokenizerPath: "/Users/yangxuehui/Library/Caches/MOSSTTSKit/Models/MOSS-TTS-Nano-100M-ONNX/tokenizer.model",
            modelName: "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
        )
        try await tokenizer.load()

        let text = "欢迎关注模思智能、上海创智学院与复旦大学自然语言处理实验室。"
        let encoding = try await tokenizer.encode(text)

        XCTAssertEqual(
            encoding.ids,
            [8651, 2691, 11099, 10670, 7669, 10508, 4627, 11074, 11315, 6439, 10617, 10859, 11643, 2957, 1531, 4139, 2305, 4146, 11255, 10382]
        )
    }

    func testSentencePieceTokenizerRegressionSamplesMatchOfficialModel() async throws {
        let tokenizer = SentencePieceTokenizer(
            tokenizerPath: "/Users/yangxuehui/Library/Caches/MOSSTTSKit/Models/MOSS-TTS-Nano-100M-ONNX/tokenizer.model",
            modelName: "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
        )
        try await tokenizer.load()

        let samples: [(String, [Int32])] = [
            (
                "你好，这是一个包内测试音频。",
                [3985, 10445, 10364, 4960, 10779, 10608, 7306, 10892, 11577, 10382]
            ),
            (
                "今天是2026年5月1日，天气不错。",
                [10356, 1054, 10387, 7224, 10752, 10492, 10604, 10672, 10385, 10656, 10364, 10126, 3513, 10382]
            ),
            (
                "请在3分钟后提醒我开会。",
                [10356, 10926, 10405, 10386, 3134, 10428, 5600, 10398, 10506, 10434, 10382]
            ),
            (
                "OpenAI 的 GPT 模型现在支持中英混排。",
                [543, 7317, 7958, 4690, 558, 9304, 10356, 9016, 643, 3047, 10439, 11048, 11332, 11103, 10382]
            ),
            (
                "价格是 12.5 元，不是 15 元。",
                [10356, 3219, 10387, 3433, 10380, 10604, 10356, 10819, 10364, 552, 3015, 10356, 10819, 10382]
            ),
            (
                "第一段说中文。Second sentence mixes English.",
                [10356, 1023, 10853, 10416, 10439, 10646, 10382, 10412, 10357, 10373, 805, 2923, 851, 6843, 302, 5556, 10380]
            )
        ]

        for (text, expectedIDs) in samples {
            let encoding = try await tokenizer.encode(text)
            XCTAssertEqual(encoding.ids, expectedIDs, "Unexpected token ids for text: \(text)")
        }
    }
}
