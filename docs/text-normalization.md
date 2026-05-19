# Text Normalization

MOSSTTSKit normalizes user text before tokenization and chunking. The goal is to
turn app-facing text into a safer TTS prompt while preserving the intended
reading content.

## Why This Exists

MOSS-TTS-Nano is sensitive to some punctuation patterns. Raw book text, copied
web text, and speaker labels can contain punctuation that looks natural to a
reader but behaves like an unfinished or unusual prompt to the model.

The normalization layer handles those cases in one place instead of scattering
small fixes across synthesis, chunking, and app integrations.

## Current Rules

- Leading and trailing whitespace is removed.
- Chinese and ASCII ellipses are converted to sentence boundaries.
- Repeated dash separators such as `---`, `--`, and `——` are converted to
  sentence boundaries. Single hyphens in words such as `MOSS-TTS-Nano` are kept.
- Non-empty line breaks are treated as phrase or sentence boundaries.
- A dangling final clause punctuation mark, such as `：`, `:`, `，`, or `,`, is
  converted into a sentence terminator.
- Repeated spaces introduced by normalization are collapsed.

Examples:

```text
利娜正睡在我身边……

我一点都不想再睡了。
```

becomes:

```text
利娜正睡在我身边。 我一点都不想再睡了。
```

```text
Taiguanglin：
```

becomes:

```text
Taiguanglin.
```

```text
非常值得一读，---揭示地球史前文明。
```

becomes:

```text
非常值得一读。 揭示地球史前文明。
```

## Engineering Rule

New text preprocessing behavior should be added through `TextNormalizer`, with a
focused regression test in `TextNormalizerTests`. If the behavior affects real
model output, also add or update an integration test or listening sample that
uses the original problematic text.
