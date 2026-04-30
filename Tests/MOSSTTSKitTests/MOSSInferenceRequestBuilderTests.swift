import Foundation
import XCTest
@testable import MOSSTTSKit

final class MOSSInferenceRequestBuilderTests: XCTestCase {
    func testManifestLoadsAndBuildsVoiceCloneRows() throws {
        let manifestURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        try manifestJSON.write(to: manifestURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: manifestURL) }
        
        let manifest = try MOSSBrowserManifest.load(from: manifestURL)
        let builder = MOSSInferenceRequestBuilder(manifest: manifest)
        let request = builder.buildVoiceCloneRequestRows(
            promptAudioCodes: [[10, 11, 12], [20, 21, 22]],
            textTokenIds: [900, 901]
        )
        
        XCTAssertEqual(manifest.builtinVoices.first?.voice, "demo")
        XCTAssertEqual(manifest.builtinVoices.first?.displayName, "Demo Voice")
        XCTAssertEqual(manifest.builtinVoices.first?.group, "Demo")
        XCTAssertEqual(manifest.builtinVoices.first?.audioFile, "demo.wav")
        XCTAssertEqual(request.attentionMask, [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        XCTAssertEqual(request.inputIds[0], [100, -1, -1, -1])
        XCTAssertEqual(request.inputIds[1], [201, -1, -1, -1])
        XCTAssertEqual(request.inputIds[2], [301, 10, 11, 12])
        XCTAssertEqual(request.inputIds[3], [301, 20, 21, 22])
        XCTAssertEqual(request.inputIds[4], [202, -1, -1, -1])
        XCTAssertEqual(request.inputIds[5], [110, -1, -1, -1])
        XCTAssertEqual(request.inputIds[6], [900, -1, -1, -1])
        XCTAssertEqual(request.inputIds[7], [901, -1, -1, -1])
        XCTAssertEqual(request.inputIds[8], [120, -1, -1, -1])
        XCTAssertEqual(request.inputIds[9], [201, -1, -1, -1])
    }
    
    private var manifestJSON: String {
        """
        {
          "model_files": {
            "tts_meta": "tts_browser_onnx_meta.json",
            "codec_meta": "codec_browser_onnx_meta.json",
            "tokenizer_model": "tokenizer.model"
          },
          "tts_config": {
            "n_vq": 3,
            "audio_pad_token_id": -1,
            "audio_start_token_id": 201,
            "audio_end_token_id": 202,
            "audio_user_slot_token_id": 301,
            "audio_assistant_slot_token_id": 302
          },
          "prompt_templates": {
            "user_prompt_prefix_token_ids": [100],
            "user_prompt_after_reference_token_ids": [110],
            "assistant_prompt_prefix_token_ids": [120]
          },
          "generation_defaults": {
            "max_new_frames": 128,
            "do_sample": true,
            "sample_mode": "topk",
            "text_temperature": 1.0,
            "text_top_k": 50,
            "text_top_p": 0.95,
            "audio_temperature": 0.8,
            "audio_top_k": 30,
            "audio_top_p": 0.9,
            "audio_repetition_penalty": 1.1
          },
          "builtin_voices": [
            {
              "voice": "demo",
              "display_name": "Demo Voice",
              "group": "Demo",
              "audio_file": "demo.wav",
              "prompt_audio_codes": [[1, 2, 3]]
            }
          ]
        }
        """
    }
}
