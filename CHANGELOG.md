# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.7]

### Added

- Automatic long-text chunking for `speak(...)`, `speakStream(...)`, and `speakToFile(...)`
- `MOSSTTSOptions.maxTextTokensPerChunk` for controlling per-chunk token budget
- `MOSSTTSOptions.maxReferenceAudioDuration`, `MOSSTTSOptions.maxReferenceAudioPromptFrames`, and `makeSpeaker(..., maxDuration:)` for bounded voice-clone reference audio encoding and prompt prefill
- Inter-chunk pause insertion for more natural long-form playback
- Internal regression sample generator coverage for listening-based verification

### Fixed

- Ellipsis normalization for Chinese text so `……` / `...` are treated as sentence-level pauses instead of model input tokens that can cause skipped speech
- Tokenizer normalization alignment with the upstream SentencePiece behavior
- Chinese punctuation tokenization issues that could introduce incorrect spoken syllables
- Audio tokenizer decode trimming using `audio_lengths`
- Long regression sample truncation caused by an overly small frame cap
- Long-text chunk truncation caused by an overly aggressive default `maxGeneratedFrames` limit at the package and app call-site layers

### Improved

- Long-text synthesis usability for app integrations such as TTSMate
- Default `MOSSTTSOptions()` behavior now falls back to the model manifest frame limit when `maxGeneratedFrames` is unset
- Voice cloning now reads, stores, and feeds a bounded reference segment by default, reducing memory spikes both during `makeSpeaker(...)` and later `speak(...)` prefill
- Sample and integration guidance now distinguishes short preview frame caps from full-text synthesis defaults
- Tokenizer regression coverage for Chinese punctuation, dates, numbers, mixed Chinese/English text, and multi-sentence inputs
- Repository documentation for model download, source attribution, license details, and long-text support
