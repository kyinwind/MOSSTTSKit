# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.5]

### Added

- Automatic long-text chunking for `speak(...)`, `speakStream(...)`, and `speakToFile(...)`
- `MOSSTTSOptions.maxTextTokensPerChunk` for controlling per-chunk token budget
- Inter-chunk pause insertion for more natural long-form playback
- Internal regression sample generator coverage for listening-based verification

### Fixed

- Tokenizer normalization alignment with the upstream SentencePiece behavior
- Chinese punctuation tokenization issues that could introduce incorrect spoken syllables
- Audio tokenizer decode trimming using `audio_lengths`
- Long regression sample truncation caused by an overly small frame cap

### Improved

- Long-text synthesis usability for app integrations such as TTSMate
- Tokenizer regression coverage for Chinese punctuation, dates, numbers, mixed Chinese/English text, and multi-sentence inputs
- Repository documentation for model download, source attribution, license details, and long-text support
