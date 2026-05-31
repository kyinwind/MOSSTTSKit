import Foundation

/// 将用户输入文本规范化为更适合 TTS 模型的 prompt。
///
/// 这里处理的是“人读起来合理，但直接喂给声学模型可能不稳定”的文本形态，
/// 例如省略号、连续破折号、中文引号、换行和结尾悬空标点。规则要尽量保守，
/// 只处理已经在真实合成中证明容易导致漏读、重复或异常延长的场景。
public struct TextNormalizer: Sendable {
    /// 句末标点。长文本切分和换行归一化都会把这些符号视为完整句子边界。
    public static let sentenceEndPunctuation: Set<Character> = ["。", "！", "？", ".", "!", "?"]

    /// 从句级标点。它们适合用来切分长句，但如果出现在文本末尾，容易形成未完成 prompt。
    public static let clauseSplitPunctuation: Set<Character> = ["，", "、", "；", "：", ",", ";", ":"]

    /// 右侧闭合标点。句子切分时需要把它们保留在前一句里，例如 `他说：“你好。”`。
    public static let closingPunctuation: Set<Character> = ["”", "’", "\"", "'", "）", "】", "》", "」", "』", "〉", ")", "]"]

    public init() {}

    /// 执行文本规范化。
    ///
    /// 注意：这里不是通用自然语言清洗器，不应该随意改写用户文本。
    /// 新规则必须足够确定，并配套 `TextNormalizerTests` 回归测试。
    public func normalize(_ text: String) -> String {
        var processed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        processed = normalizeQuotationMarks(in: processed)
        processed = normalizeEllipses(in: processed)
        processed = normalizeDashSeparators(in: processed)
        processed = normalizeLineBreakBoundaries(in: processed)
        processed = normalizeDanglingTerminalPunctuation(in: processed)
        processed = normalizeChineseOpeningMarkSpacing(in: processed)
        while processed.contains("  ") {
            processed = processed.replacingOccurrences(of: "  ", with: " ")
        }
        return processed.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// 判断文本是否包含中日韩字符，用于选择中文句号还是英文句点作为补全边界。
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

    /// 移除中文排版引号。
    ///
    /// 引号通常只表达书面排版和对话边界，并不需要被 TTS 读出。真实长文本中，
    /// 以 `“` 开头的 chunk 容易让模型进入不稳定 prompt，表现为开头文字漏读、
    /// 长停顿或句尾重复。这里仅移除中文引号，暂不处理英文 `'`，避免误伤
    /// `don't` 这类英文缩写。
    private func normalizeQuotationMarks(in text: String) -> String {
        let quotationMarks: Set<Character> = ["“", "”", "‘", "’", "「", "」", "『", "』"]
        guard text.contains(where: { quotationMarks.contains($0) }) else { return text }
        return String(text.filter { !quotationMarks.contains($0) })
    }

    /// 将省略号归一化为句子边界。
    ///
    /// MOSS-TTS-Nano 对 `……` / `...` 这类原始输入比较敏感，可能导致后续文本漏读。
    /// 对中文文本使用 `。`，对非中文文本使用 `.`。
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

    /// 将连续横线或中文破折号归一化为句子边界。
    ///
    /// `---` / `——` 在书籍文本里常表示停顿或插入语，但直接进入模型时可能被当成
    /// 异常续写提示，出现重复上一句末尾几个字的情况。单个 `-` 会保留，避免误伤
    /// `MOSS-TTS-Nano` 这类英文连字符。
    private func normalizeDashSeparators(in text: String) -> String {
        guard text.contains("--") || text.contains("—") else { return text }

        let sentenceTerminator: Character = Self.containsCJK(text) ? "。" : "."
        let characters = Array(text)
        var result: [Character] = []
        result.reserveCapacity(characters.count)

        var index = 0
        while index < characters.count {
            let character = characters[index]

            if character == "-" {
                var runEnd = index
                while runEnd < characters.count, characters[runEnd] == "-" {
                    runEnd += 1
                }

                if runEnd - index >= 2 {
                    appendSentenceBoundary(sentenceTerminator, to: &result)
                    index = runEnd
                    continue
                }
            }

            if character == "—" {
                while index < characters.count, characters[index] == "—" {
                    index += 1
                }
                appendSentenceBoundary(sentenceTerminator, to: &result)
                continue
            }

            result.append(character)
            index += 1
        }

        return String(result)
    }

    /// 将非空换行归一化为句子/段落边界。
    ///
    /// 真实文本里换行通常表示段落停顿。如果直接删除换行，模型听感上会像没有停顿。
    /// 如果前一段已经有句末标点，则只插入空格；否则补一个句末边界。
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

    /// 清理句末标点和中文开括号/书名号之间的空格。
    ///
    /// `---《圣经》` 会先被归一化成 `。 《圣经》`。这个空格对中文朗读没有必要，
    /// 在极短引用文本前还可能被模型放大成长停顿，所以这里把它收回为 `。《圣经》`。
    private func normalizeChineseOpeningMarkSpacing(in text: String) -> String {
        var result = text
        let openingMarks = ["《", "“", "‘", "「", "『", "（", "【"]
        let sentenceTerminators = ["。", "！", "？", ".", "!", "?"]

        for terminator in sentenceTerminators {
            for mark in openingMarks {
                result = result.replacingOccurrences(of: "\(terminator) \(mark)", with: "\(terminator)\(mark)")
            }
        }

        return result
    }

    /// 修正结尾悬空的从句标点。
    ///
    /// 例如 `Taiguanglin：`、`旁白：` 或以逗号结尾的短文本，对模型来说像是
    /// “还没说完的 prompt”，容易生成异常长音频或不稳定尾音。这里会去掉末尾
    /// 从句标点并补成完整句子。
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

    /// 在结果尾部追加句末标点；如果已经有句末标点则不重复追加。
    private func appendSentenceTerminator(_ terminator: Character, to result: inout [Character]) {
        while let last = result.last, last.isWhitespace {
            result.removeLast()
        }
        if let last = result.last, Self.sentenceEndPunctuation.contains(last) {
            return
        }
        result.append(terminator)
    }

    /// 追加一个句子边界，并清理边界前不适合直接接句号的空白和从句标点。
    ///
    /// 例如 `非常值得一读，---揭示...` 会变成 `非常值得一读。 揭示...`，
    /// 而不是 `非常值得一读，。 揭示...`。
    private func appendSentenceBoundary(_ terminator: Character, to result: inout [Character]) {
        while let last = result.last, last.isWhitespace || Self.clauseSplitPunctuation.contains(last) {
            result.removeLast()
        }

        guard !result.isEmpty else { return }

        appendSentenceTerminator(terminator, to: &result)

        if let last = result.last, !last.isWhitespace {
            result.append(" ")
        }
    }
}
