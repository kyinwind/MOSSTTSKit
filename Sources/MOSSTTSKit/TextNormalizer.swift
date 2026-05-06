import Foundation

/// Normalizes user-facing text into a safer TTS prompt.
///
/// This layer intentionally handles punctuation that is meaningful for reading
/// rhythm but risky when passed to the acoustic model as raw text.
public struct TextNormalizer: Sendable {
    public static let sentenceEndPunctuation: Set<Character> = ["。", "！", "？", ".", "!", "?"]
    public static let clauseSplitPunctuation: Set<Character> = ["，", "、", "；", "：", ",", ";", ":"]
    public static let closingPunctuation: Set<Character> = ["”", "’", "\"", "'", "）", "】", "》", "」", "』", "〉", ")", "]"]

    public init() {}

    public func normalize(_ text: String) -> String {
        var processed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        processed = normalizeEllipses(in: processed)
        processed = normalizeLineBreakBoundaries(in: processed)
        processed = normalizeDanglingTerminalPunctuation(in: processed)
        while processed.contains("  ") {
            processed = processed.replacingOccurrences(of: "  ", with: " ")
        }
        return processed
    }

    public static func containsCJK(_ text: String) -> Bool {
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x4E00...0x9FFF, 0x3400...0x4DBF, 0x3040...0x30FF, 0xAC00...0xD7AF:
                return true
            default:
                continue
            }
        }
        return false
    }

    private func normalizeEllipses(in text: String) -> String {
        guard text.contains("…") || text.contains("...") else { return text }

        let sentenceTerminator: Character = Self.containsCJK(text) ? "。" : "."
        let characters = Array(text)
        var result: [Character] = []
        result.reserveCapacity(characters.count)

        var index = 0
        while index < characters.count {
            let character = characters[index]

            if character == "…" {
                while index < characters.count, characters[index] == "…" {
                    index += 1
                }
                appendSentenceTerminator(sentenceTerminator, to: &result)
                continue
            }

            if character == "." {
                var runEnd = index
                while runEnd < characters.count, characters[runEnd] == "." {
                    runEnd += 1
                }

                if runEnd - index >= 3 {
                    appendSentenceTerminator(sentenceTerminator, to: &result)
                    index = runEnd
                    continue
                }
            }

            result.append(character)
            index += 1
        }

        return String(result)
    }

    private func normalizeLineBreakBoundaries(in text: String) -> String {
        let normalizedNewlines = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")

        let parts = normalizedNewlines
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !parts.isEmpty else { return "" }

        var result = ""
        result.reserveCapacity(normalizedNewlines.count + parts.count * 2)

        for (index, part) in parts.enumerated() {
            if index > 0 {
                let separator = Self.containsCJK(part) || Self.containsCJK(result) ? "。 " : ". "
                if let last = result.last, !Self.sentenceEndPunctuation.contains(last) {
                    result.append(contentsOf: separator)
                } else {
                    result.append(" ")
                }
            }
            result.append(contentsOf: part)
        }

        return result
    }

    private func normalizeDanglingTerminalPunctuation(in text: String) -> String {
        var characters = Array(text.trimmingCharacters(in: .whitespacesAndNewlines))
        guard let last = characters.last, Self.clauseSplitPunctuation.contains(last) else {
            return text
        }

        while let last = characters.last, Self.clauseSplitPunctuation.contains(last) || last.isWhitespace {
            characters.removeLast()
        }

        guard !characters.isEmpty else {
            return ""
        }

        let terminator: Character = Self.containsCJK(String(characters)) ? "。" : "."
        appendSentenceTerminator(terminator, to: &characters)
        return String(characters)
    }

    private func appendSentenceTerminator(_ terminator: Character, to result: inout [Character]) {
        while let last = result.last, last.isWhitespace {
            result.removeLast()
        }
        if let last = result.last, Self.sentenceEndPunctuation.contains(last) {
            return
        }
        result.append(terminator)
    }
}
