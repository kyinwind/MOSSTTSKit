# 从 ASR/TTS 到 MOSS-TTS-Nano：语音大模型工程入门

这本小书面向想把语音模型真正接进产品的人。它不会只讲论文名词，也不会只讲怎么运行 demo，而是希望把语音 AI 的技术路线、模型结构、推理流程、工程坑点和 MOSSTTSKit 的实现经验连起来。

本书会随着 MOSSTTSKit 的开发和讨论持续补充。

## 目录草案

1. 语音 AI 全景图
2. ASR 主流技术路线
3. TTS 主流技术路线
4. 语音模型中的共同问题
5. MOSS-TTS-Nano 深入
6. Qwen3-TTS 深入
7. MOSS-TTS-Nano 与 Qwen3-TTS 对比
8. 从官方 runtime 到 Swift Package
9. 长文本、流式、内存与产品化
10. 测试、试听样本与质量评估

---

# 第一章：语音 AI 全景图

## 1.1 为什么先看全景图

如果只从 MOSS-TTS-Nano 的代码开始看，很容易被一堆文件名淹没：

- `tokenizer.model`
- `moss_tts_prefill.onnx`
- `moss_tts_decode_step.onnx`
- `moss_tts_local_decoder.onnx`
- `moss_audio_tokenizer_encode.onnx`
- `moss_audio_tokenizer_decode_step.onnx`
- `browser_poc_manifest.json`

这些东西看起来像一堆工程零件，但它们背后其实对应的是语音 AI 的几个核心问题：

- 文字如何变成模型能理解的 token？
- 连续的声音如何变成离散的 token？
- 模型是在生成 Mel 频谱，还是在生成 audio code？
- 为什么需要 prefill？
- 为什么需要 decode step？
- 为什么有些模型支持流式，有些模型不支持？
- 为什么长文本会吞字、爆内存、或者停顿不自然？

这一章先不急着解释每个 ONNX 输入输出，而是先建立一张地图。地图清楚了，后面看 MOSS-TTS-Nano、Qwen3-TTS、Whisper、Paraformer、CosyVoice、F5-TTS 时，就不会觉得它们是完全不同的魔法。

## 1.2 语音 AI 的四个基本方向

语音 AI 可以先粗略分成四类任务。

### ASR：Automatic Speech Recognition

ASR 是语音识别，也就是：

```text
音频 -> 文字
```

例子：

```text
输入：一段 wav 音频
输出：你好，欢迎使用 TTSMate。
```

ASR 的难点是，音频是连续信号，文字是离散符号。模型要从波形里判断哪些声音对应哪些字、词、标点，甚至还要处理口音、噪声、多人说话、背景音乐、断句和热词。

常见产品：

- 会议转写
- 字幕生成
- 语音输入法
- 视频字幕
- 客服录音质检
- 语音助手听懂用户说什么

### TTS：Text To Speech

TTS 是语音合成，也就是：

```text
文字 -> 音频
```

例子：

```text
输入：你好，欢迎使用 MOSS-TTS-Nano。
输出：一段真实可播放的语音
```

TTS 的难点不是“念出文字”这么简单，而是要决定：

- 每个字怎么读？
- 多音字选哪个读音？
- 什么时候停顿？
- 用什么音色？
- 情绪是什么？
- 语速多快？
- 声音是不是稳定？
- 长文本会不会漏读？
- 音频有没有杂音、破音、机械感？

常见产品：

- 有声书
- 视频配音
- 语音助手回答
- 导航播报
- 无障碍朗读
- 游戏角色语音
- 内容创作者批量配音

### Voice Clone：语音克隆

语音克隆是 TTS 的一个增强方向：

```text
参考音频 + 文字 -> 模仿参考音色的新语音
```

例子：

```text
参考音频：某个人说了 10 秒话
输入文字：今天我们继续讲语音模型。
输出音频：用类似参考音频的音色读出这句话
```

语音克隆又可以分成几类：

- 预训练固定音色：模型自带若干 speakers。
- 零样本克隆：给一小段参考音频，不训练，直接模仿。
- 微调克隆：拿一个人的数据继续训练或 LoRA。
- 跨语言克隆：参考音频是中文，说出来可以是英文、日文等。

MOSS-TTS-Nano 支持内置音色，也支持用参考音频做 voice clone。MOSSTTSKit 里 `makeSpeaker(name:referenceAudioURL:)` 做的就是把参考音频转成模型可用的 speaker prompt。

### Voice Design：语音设计

语音设计比语音克隆更进一步：

```text
音色描述 + 文字 -> 新音色语音
```

例子：

```text
音色描述：年轻男性，声音温暖，语速稍慢，有一点纪录片旁白感。
文字：宇宙并不沉默，只是我们听见得太少。
输出：符合描述的新声音
```

Qwen3-TTS 这类模型开始强调 voice design。它不仅能克隆某个已有声音，还能根据自然语言描述生成或控制声音。这说明 TTS 正在从“读文字的工具”变成“可控语音生成模型”。

## 1.3 语音和文本最大的差异

文本天然是离散的。

一句话：

```text
你好，世界。
```

可以被切成字符、词、subword token：

```text
[你] [好] [，] [世界] [。]
```

而声音天然是连续的。

一段 wav 音频，本质上是大量采样点：

```text
[0.001, 0.003, -0.002, 0.010, ...]
```

如果是 48 kHz 音频，意思是每秒有 48000 个采样点。立体声还要乘以 2 个声道。一分钟音频就是几百万个浮点数。

所以语音模型绕不开一个问题：

> 怎么把连续的声音变成模型容易处理的表示？

历史上有几种答案。

## 1.4 声音的几种表示方式

### 波形 waveform

波形是最原始的音频表示。

```text
waveform = 一串采样点
```

优点：

- 信息最完整。
- 可以直接播放。

缺点：

- 序列太长。
- 建模很难。
- 对大模型来说计算成本很高。

直接生成 waveform 的模型存在，但工程上通常会用中间表示降低难度。

### Mel spectrogram

Mel 频谱是传统神经 TTS 很常见的中间表示。

典型流程是：

```text
文本 -> Mel 频谱 -> vocoder -> waveform
```

Tacotron、FastSpeech、很多早期神经 TTS 都属于这个大方向。

优点：

- 比 waveform 短很多。
- 和人耳感知更接近。
- 训练和生成都比直接波形容易。

缺点：

- Mel 不是最终音频，还需要 vocoder。
- Mel 频谱会损失一部分细节。
- 情绪、音色、韵律控制仍然复杂。

### Codec token / audio code

近年的语音大模型越来越喜欢把音频变成离散 token。

流程大致是：

```text
waveform -> audio tokenizer encode -> audio codes
audio codes -> audio tokenizer decode -> waveform
```

这样声音就从连续信号变成了类似文本 token 的离散序列。

比如 MOSS-Audio-Tokenizer-Nano 会把 48 kHz stereo 音频压缩成低帧率的 audio code。MOSS-TTS-Nano 再像语言模型预测下一个词一样，预测下一个 audio code frame。

这条路线非常关键，因为它把 TTS 变成了一个更像 LLM 的问题：

```text
给定文本 token 和提示音频 token，预测后续音频 token。
```

MOSS-TTS-Nano 和 Qwen3-TTS 都属于这一代思路。

## 1.5 为什么语音模型越来越像 LLM

大语言模型处理的是 token 序列。

```text
输入 token -> Transformer -> 输出下一个 token
```

如果音频也能被变成 token，那么语音生成也可以变成：

```text
文本 token + 音频 prompt token -> Transformer -> 下一个音频 token
```

这就是 Codec LM 路线的核心直觉。

它的好处是：

- 可以复用 Transformer / LLM 的很多经验。
- 自然支持自回归生成。
- 语音克隆可以变成 prompt conditioning。
- 多语言、情绪、音色、风格可以放进同一个 token 序列或条件里。
- 流式生成更自然，因为模型本来就是一步一步生成。

它的代价是：

- 推理过程更复杂。
- 需要 audio tokenizer。
- 需要处理 KV cache。
- 采样策略会影响稳定性。
- 长文本会带来上下文长度和内存问题。
- 生成是概率性的，同一句话多次生成可能不同。

我们在 MOSSTTSKit 里遇到的很多问题，本质上都来自这里：

- tokenizer 不一致，读音就会偏。
- 标点异常，模型可能误判边界。
- `maxGeneratedFrames` 太小，语音会截断。
- 长文本一次性生成，内存会涨。
- 语音克隆参考音频太长，prefill 会变重。
- 采样随机数不对齐，和官方输出就会飘。

## 1.6 一个 TTS 系统通常由哪些部件组成

不管是传统 TTS，还是 MOSS-TTS-Nano 这种 Codec LM TTS，工程上通常都有这些部件。

### 文本规范化

原始文本不能直接喂给模型。

例子：

```text
3.14
2026年5月6日
Taiguanglin：
你好……
```

这些文本对人来说很自然，但对模型来说可能有歧义。

文本规范化要处理：

- 数字
- 日期
- 单位
- 标点
- 换行
- 中英混排
- 多音字
- 特殊符号

MOSSTTSKit 现在有 `TextNormalizer`，先处理省略号、连续破折号、中文引号、换行、结尾悬空标点这类确定会影响稳定性的输入。

### 文本 tokenizer

文本 tokenizer 把文字变成 token id。

```text
你好 -> [token1, token2, ...]
```

MOSS-TTS-Nano 使用 SentencePiece tokenizer。我们之前在 MOSSTTSKit 里花了很多时间处理 tokenizer，就是因为 tokenizer 稍微不一致，模型输入就不一样，生成音频也会不一样。

### 声学模型

声学模型决定“文字如何变成声音表示”。

传统 TTS 里，它可能输出 Mel 频谱。

MOSS-TTS-Nano 里，它输出 audio code。

Qwen3-TTS 里，它也是围绕离散语音 token 做生成。

### Vocoder 或 Audio Tokenizer Decoder

如果模型输出 Mel，需要 vocoder：

```text
Mel -> waveform
```

如果模型输出 audio code，需要 audio tokenizer decoder：

```text
audio codes -> waveform
```

MOSS-TTS-Nano 使用的是 MOSS-Audio-Tokenizer-Nano 的 decode 模型。MOSSTTSKit 中对应 `AudioTokenizerONNX`。

### 采样策略

生成式模型通常不是简单取最大概率，而是会采样。

常见参数：

- temperature
- top-k
- top-p
- repetition penalty
- seed

这些参数会影响：

- 声音是否自然
- 是否稳定
- 是否有随机差异
- 是否重复
- 是否提前结束
- 是否漏读

我们在 MOSSTTSKit 中对齐官方 runtime 的随机源，就是为了让 Swift 版本更接近官方 Python 行为。

## 1.7 ASR 和 TTS 的共同点

ASR 和 TTS 看起来方向相反：

```text
ASR: 音频 -> 文字
TTS: 文字 -> 音频
```

但它们有很多共同问题。

### 都要处理时间对齐

音频有时间轴，文字没有天然时间轴。

ASR 要回答：

```text
这段声音的第几秒对应哪个字？
```

TTS 要回答：

```text
这个字应该说多久？什么时候停顿？
```

### 都要处理 tokenizer

ASR 的输出可以是字符、BPE、wordpiece。

TTS 的输入也可以是字符、拼音、音素、BPE。

新一代语音模型还会有 audio tokenizer，把音频也离散化。

### 都要处理流式

离线处理可以等整段音频或整段文本都准备好。

流式处理则要求边输入边输出。

ASR 流式：

```text
用户边说，系统边出字。
```

TTS 流式：

```text
模型边生成，播放器边播放。
```

流式会带来额外难题：

- 首包延迟
- 缓冲
- chunk 边界
- 上下文保持
- 中途取消
- 内存释放

MOSSTTSKit 现在的长文本优化，本质上就是让合成更接近流式处理，而不是把整段 audio codes 全部堆在内存里。

## 1.8 MOSS-TTS-Nano 在全景图里的位置

MOSS-TTS-Nano 可以这样定位：

```text
任务：TTS / Voice Clone
路线：Audio Tokenizer + LLM / Codec LM
生成方式：自回归生成 audio codes
部署目标：轻量、本地、CPU 友好、流式
音频规格：48 kHz stereo
```

它不是传统的：

```text
文本 -> Mel -> Vocoder
```

而更接近：

```text
文本 token + prompt audio codes
-> Transformer 自回归生成 audio codes
-> Audio Tokenizer decode
-> waveform
```

这解释了为什么 MOSSTTSKit 里会有这些模块：

- `TextNormalizer`：先把文本变成更安全的 prompt。
- `SentencePieceTokenizer`：把文本转成 token id。
- `MOSSInferenceRequestBuilder`：把文本 token 和 speaker prompt 拼成模型输入。
- `MOSSTTSEngine`：执行 prefill、decode step、local decode、sample。
- `AudioTokenizerONNX`：把 audio codes 解码成可播放音频。
- `speakStream(...)`：把生成过程暴露成流式输出。

## 1.9 Qwen3-TTS 在全景图里的位置

Qwen3-TTS 也属于新一代离散语音 token 路线。

根据官方资料，它使用 Qwen3-TTS-Tokenizer-12Hz，并采用 discrete multi-codebook LM 架构。它支持：

- 语音克隆
- 自定义内置音色
- voice design
- instruction control
- 流式生成
- 多语言

可以粗略理解为：

```text
MOSS-TTS-Nano 更偏轻量本地部署。
Qwen3-TTS 更偏完整语音生成能力和自然语言控制。
```

这不是说谁绝对更好，而是产品目标不同：

- 如果你要在 TTSMate 里做本地、低成本、可控部署，MOSS-TTS-Nano 很有价值。
- 如果你要更强的音色设计、情绪控制、跨语言能力，并且能接受更重的模型或服务端部署，Qwen3-TTS 更值得研究。

## 1.10 第一章小结

这一章建立了几件事：

1. ASR 是音频到文字，TTS 是文字到音频。
2. 现代语音模型的核心问题，是如何在连续音频和离散 token 之间转换。
3. 传统 TTS 常见路线是 `文本 -> Mel -> Vocoder`。
4. 新一代 Codec LM 路线是 `文本 token -> audio token -> waveform`。
5. MOSS-TTS-Nano 属于 Audio Tokenizer + LLM 的轻量自回归 TTS。
6. Qwen3-TTS 也属于离散语音 token + LM 路线，但能力更完整、模型更大。
7. MOSSTTSKit 的很多工程问题，本质上来自 tokenizer、文本规范化、采样、流式、长文本和内存控制。

如果只记一句话：

> 今天的 TTS 正在从“文本生成频谱”转向“语言模型生成音频 token”，MOSS-TTS-Nano 就是这个方向的轻量本地化代表。

## 1.11 本章练习

为了真正掌握这一章，可以试着回答几个问题：

1. 为什么 48 kHz stereo waveform 不适合直接让普通 Transformer 逐采样点生成？
2. Mel 频谱和 audio code 的区别是什么？
3. 为什么 audio tokenizer 会让 TTS 更像 LLM？
4. MOSS-TTS-Nano 为什么需要 `moss_audio_tokenizer_decode_step.onnx`？
5. 为什么同一句话多次生成，结果可能略有不同？
6. 为什么 `Taiguanglin：` 这种短文本可能让 TTS 模型异常？

这些问题不要求现在全部答对。后面的章节会逐步展开。

## 参考资料

- OpenMOSS / MOSS-TTS-Nano: https://github.com/OpenMOSS/MOSS-TTS-Nano
- OpenMOSS / MOSS-TTS Family: https://github.com/OpenMOSS/MOSS-TTS
- QwenLM / Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- Qwen TTS API documentation: https://www.alibabacloud.com/help/en/model-studio/qwen-tts
- A Survey on Neural Speech Synthesis: https://github.com/tts-tutorial/survey

---

# 第二章：ASR 主流技术路线

## 2.1 ASR 要解决的核心问题

ASR 是 Automatic Speech Recognition，也就是语音识别：

```text
音频 -> 文字
```

看起来这只是 TTS 的反方向：

```text
TTS: 文字 -> 音频
ASR: 音频 -> 文字
```

但工程上，ASR 有自己的难点。

一段音频不是一串干净的字，而是一串连续采样点。模型要从这些采样点里判断：

- 哪些部分是人声？
- 哪些部分是噪声？
- 说话人说了哪些音？
- 这些音对应哪些字或词？
- 标点应该在哪里？
- 多音字、同音字怎么区分？
- 句子什么时候结束？
- 用户有没有口音？
- 背景音乐、回声、重叠说话怎么处理？

ASR 的根本矛盾是：

> 输入音频有很长的时间轴，输出文字是更短的符号序列，而且训练数据通常没有精确到每一帧对应哪个字。

这就是 ASR 中最重要的问题之一：**对齐问题**。

## 2.2 一个典型 ASR 系统的处理流程

不管模型多新，一个 ASR 系统通常都绕不开这些步骤：

```text
音频输入
 -> 音频预处理
 -> 特征提取
 -> 声学建模
 -> 解码
 -> 文本后处理
 -> 最终转写文本
```

### 音频预处理

常见操作包括：

- 重采样，比如转成 16 kHz。
- 单声道化。
- 音量归一化。
- 降噪。
- VAD，也就是检测哪里有人说话。
- 分段，把长音频切成更短片段。

这一步在产品里非常重要。很多 ASR 效果问题不是模型本身不行，而是音频输入太差。

### 特征提取

早期 ASR 不直接吃 waveform，而是先提取声学特征。

常见特征：

- MFCC
- filter bank
- log-Mel spectrogram

现代模型，比如 Whisper，也会把音频转成 log-Mel spectrogram 再进入 Transformer encoder。

### 声学建模

声学模型负责把声音特征映射到某种中间表示或 token 概率。

早期是 GMM-HMM。

后来是 DNN-HMM。

再后来是 CTC、RNN-T、Attention Encoder-Decoder、Conformer、Whisper 这一类端到端模型。

### 解码

解码就是从模型输出中找出最合理的文字序列。

简单模型可以直接 greedy decode。

复杂一点会用 beam search。

还可能融合语言模型、热词、上下文 biasing。

### 文本后处理

模型输出可能是：

```text
ni hao jin tian tian qi bu cuo
```

也可能是：

```text
你好今天天气不错
```

产品通常还需要：

- 加标点。
- 数字格式化。
- 中英文空格处理。
- 专有名词修正。
- 繁简转换。
- 敏感词或热词处理。

所以 ASR 不只是“跑一个模型”，而是一整条 pipeline。

## 2.3 传统路线：GMM-HMM

早期主流 ASR 是 GMM-HMM。

可以粗略理解为：

```text
音频特征
 -> GMM 判断每一帧像哪个声学状态
 -> HMM 建模状态随时间怎么转移
 -> 词典和语言模型帮助解码成文字
```

这里有几个组件：

- 声学模型：判断声音像哪个音素或状态。
- 发音词典：告诉系统词怎么发音。
- 语言模型：判断哪些词序列更合理。
- HMM：处理时间序列和状态转移。

优点：

- 理论体系成熟。
- 可解释性相对强。
- 在深度学习前长期是主流。

缺点：

- 系统复杂。
- 训练流程繁琐。
- 声学模型、词典、语言模型分开优化。
- 工程维护成本高。

这条路线现在不是新系统首选，但理解它很重要，因为它告诉我们 ASR 长期面对的核心问题是：

> 音频帧和文字标签之间没有天然一一对应关系，需要某种对齐机制。

## 2.4 混合路线：DNN-HMM

深度学习进入 ASR 后，GMM 逐渐被 DNN 替代。

于是出现 DNN-HMM：

```text
音频特征
 -> DNN 预测声学状态概率
 -> HMM 解码
 -> 词典 + 语言模型
 -> 文本
```

相比 GMM-HMM，DNN 更擅长从数据里学习复杂声学模式，所以准确率明显提升。

优点：

- 比 GMM 声学建模能力强。
- 可以继续复用 HMM 解码框架。
- 在工业界有过很长时间的大规模成功。

缺点：

- 仍然不是端到端。
- 仍然依赖复杂词典和解码图。
- 系统组件很多，调试困难。

这条路线可以看成传统 ASR 和现代端到端 ASR 之间的过渡阶段。

## 2.5 端到端路线一：CTC

CTC 是 Connectionist Temporal Classification。

它的核心目标是解决：

> 训练数据只有整句音频和整句文字，没有每一帧音频对应哪个字的对齐标注。

CTC 的思路是让模型每一帧都输出一个 token 分布，但允许输出一个特殊 blank 符号。

例子：

```text
目标文字：你好
模型帧输出：你 _ _ 好 _ _
折叠之后：你好
```

其中 `_` 可以理解为 blank。

CTC 会把很多可能的帧级路径合并成同一个最终文本，并用动态规划计算概率。

### CTC 的典型流程

```text
音频特征
 -> encoder
 -> 每帧 token 概率
 -> 去掉 blank
 -> 合并重复 token
 -> 文本
```

### CTC 的优点

- 训练简单。
- 不需要精确帧级对齐。
- 解码速度快。
- 容易做流式或近似流式。
- 工程上很实用。

### CTC 的缺点

- 条件独立假设较强。
- 输出之间的语言依赖建模较弱。
- 对复杂上下文和长距离依赖不如自回归 decoder。
- 常需要外部语言模型或上下文增强。

很多中文 ASR、轻量 ASR、端侧 ASR 都喜欢 CTC 或 CTC 变体，因为它简单、高效、可控。

## 2.6 端到端路线二：Attention Encoder-Decoder

Attention Encoder-Decoder，也可以叫 AED。

它更像机器翻译：

```text
音频特征序列 -> encoder -> hidden states
decoder 通过 attention 逐 token 生成文字
```

流程：

```text
音频
 -> encoder
 -> decoder attends to encoder states
 -> 输出 token1
 -> 输出 token2
 -> ...
 -> 文本
```

### AED 的优点

- decoder 可以建模输出 token 之间的依赖。
- 更像自然语言生成。
- 对上下文、语言结构建模能力强。
- 可以和 Transformer 很自然地结合。

### AED 的缺点

- 自回归生成，解码可能较慢。
- attention 通常需要看到较完整上下文，流式更难。
- 长音频可能出现 attention 漂移、漏字、重复。

Whisper 可以理解为一种强大的 encoder-decoder Transformer ASR 系统。它把音频切成 30 秒片段，转成 log-Mel spectrogram，然后用 encoder-decoder Transformer 输出文本。

## 2.7 端到端路线三：RNN-T / Transducer

RNN-T 是 Recurrent Neural Network Transducer。

它在语音助手、手机输入、实时字幕等流式 ASR 场景里非常重要。

RNN-T 可以粗略分成三部分：

```text
audio encoder
prediction network
joint network
```

它同时看：

- 当前音频上下文。
- 已经输出过的 token。

然后决定下一个 token 是什么，或者输出 blank 表示暂时不出字。

### RNN-T 的优点

- 天然适合流式识别。
- 可以边听边出字。
- 输出 token 之间有依赖建模。
- 工业实时 ASR 中非常重要。

### RNN-T 的缺点

- 训练和解码比 CTC 更复杂。
- 实现难度较高。
- 调参空间更大。

如果你做的是实时语音输入法、语音助手、直播字幕，RNN-T / Transducer 是必须理解的路线。

## 2.8 主干网络：CNN、RNN、Transformer、Conformer

CTC、AED、RNN-T 更像训练目标或解码框架。

真正提取音频特征的 encoder，还可以有不同架构。

### CNN

CNN 擅长局部模式。

在语音里，局部频谱形状很重要，所以 CNN 常用于前端特征提取或下采样。

### RNN / LSTM / GRU

RNN 擅长时间序列。

早期神经 ASR 里 LSTM 很常见，尤其是 BLSTM。

缺点是并行能力不如 Transformer。

### Transformer

Transformer 擅长建模全局依赖。

语音中长距离上下文也很重要，比如一句话后面的词会帮助判断前面的同音词。

缺点是原始 self-attention 对长序列成本较高，而且纯 Transformer 对局部声学模式不一定最优。

### Conformer

Conformer 是现代 ASR 中非常重要的 encoder 架构。

它把 CNN 和 Transformer 结合起来：

```text
CNN 负责局部声学模式
Transformer 负责全局上下文
```

所以 Conformer 很适合语音识别。很多现代 ASR 模型都会用 Conformer 或类似思想。

一句话记忆：

> Conformer 是为了让 ASR 同时看清局部声音细节和全局语言上下文。

## 2.9 大模型路线：Whisper 和大规模弱监督 ASR

Whisper 代表了另一种非常重要的路线：

```text
大规模多语言、多任务监督数据
 + encoder-decoder Transformer
 + 统一训练
 -> 鲁棒 ASR
```

它的特点是：

- 使用大量多语言音频和文本数据。
- 支持多语言转写。
- 支持语音翻译。
- 对口音、噪声、技术词汇更鲁棒。
- 使用简单端到端架构。

Whisper 给工程界的启发是：

> 当训练数据足够大、任务足够统一时，ASR 可以从复杂拼装系统走向更通用的大模型。

但 Whisper 也有代价：

- 不是天然低延迟流式。
- 长音频需要切片。
- 端侧部署有模型大小和算力压力。
- 标点和时间戳仍然需要工程处理。

## 2.10 Paraformer、SenseVoice、FunASR 这类工程 ASR

在中文和工业落地方向，Paraformer、SenseVoice、FunASR 这类方案也很重要。

它们通常更关注：

- 中文识别效果。
- 推理速度。
- 端侧或本地部署。
- 标点恢复。
- 说话人、情感、事件等扩展能力。
- ONNX Runtime / C++ runtime / 移动端部署。

这些系统不一定只对应一种论文架构，而是把模型结构、训练策略、后处理和 runtime 工程结合起来。

对我们做 MOSSTTSKit 的启发是：

> 模型只是产品能力的一部分，runtime、文本处理、流式、内存、缓存、格式转换、测试样本同样决定最终体验。

## 2.11 ASR 技术路线对比

可以用一张表理解主流路线：

| 路线 | 核心思想 | 优点 | 缺点 | 典型场景 |
| --- | --- | --- | --- | --- |
| GMM-HMM | 统计声学模型 + HMM 对齐 | 成熟、可解释 | 系统复杂、效果有限 | 历史系统 |
| DNN-HMM | DNN 替代 GMM 声学建模 | 效果提升、继承 HMM 工程 | 非端到端、维护复杂 | 传统工业 ASR |
| CTC | 帧级输出 + blank + 路径折叠 | 简单、快、易流式 | 语言建模较弱 | 轻量 ASR、端侧识别 |
| AED | encoder-decoder + attention | 上下文建模强 | 流式困难、可能慢 | 高精度离线 ASR |
| RNN-T | audio encoder + prediction network | 天然流式 | 实现复杂 | 实时输入、助手 |
| Conformer | CNN + Transformer encoder | 局部和全局兼顾 | 模型仍可能较重 | 现代高精度 ASR |
| Whisper 类 | 大规模数据 + Transformer | 鲁棒、多语言 | 低延迟和端侧压力 | 通用转写、字幕 |

## 2.12 ASR 和 TTS 的关系

学习 ASR 对理解 TTS 很有帮助。

因为它们共享几个底层问题。

### 都有 tokenizer

ASR 输出文字 token。

TTS 输入文字 token。

Codec LM TTS 还会生成 audio token。

所以 tokenizer 一旦错，ASR 可能识别错，TTS 也可能读错。

### 都有对齐问题

ASR 要把长音频对齐到短文本。

TTS 要把短文本展开成长音频。

一个是压缩，一个是展开。

```text
ASR: 很长的音频帧 -> 较短的文字 token
TTS: 较短的文字 token -> 很长的音频帧或 audio codes
```

### 都有流式问题

ASR 流式要求边听边出字。

TTS 流式要求边生成边播放。

它们都要处理：

- chunk
- cache
- 首包延迟
- 上下文
- 中途取消
- 内存释放

我们在 MOSSTTSKit 里把长文本改成更接近流式，就是借鉴了语音系统常见的工程原则：不要把整段长序列一次性堆完再处理。

### 都有后处理问题

ASR 后处理包括标点、数字、热词。

TTS 前处理包括文本规范化、标点、数字、多音字。

在产品里，前后处理经常比模型本体更影响用户感知。

## 2.13 从 ASR 反看 MOSSTTSKit

虽然 MOSSTTSKit 是 TTS 包，但 ASR 的经验对它很有启发。

### 经验一：输入预处理必须工程化

ASR 会做 VAD、降噪、切段。

TTS 也必须做文本规范化、句子切分、标点处理。

这就是我们把省略号、连续破折号、中文引号、换行、结尾冒号沉淀到 `TextNormalizer` 的原因。

### 经验二：长序列必须分块

ASR 长音频通常要分段识别。

TTS 长文本也不能无限制整段生成。

分块不是偷懒，而是语音系统的常规工程策略。

### 经验三：流式不是功能点，而是架构选择

流式不是在最后加一个 callback 就行。

真正的流式意味着：

- 模型能增量推理。
- runtime 能保留 cache。
- decoder 能增量输出。
- 调用方能边收边消费。
- 中途取消能释放资源。

MOSSTTSKit 现在的 `speakStream(...)` 和增量 audio decode，就是往这个方向走。

### 经验四：测试不能只看文本，也要听

ASR 可以用 WER、CER 评估。

TTS 不能只看有没有生成文件，还要听：

- 是否漏读？
- 是否多读？
- 是否停顿自然？
- 是否音色稳定？
- 是否有杂音？

这也是为什么我们要保留试听样本和真实模型集成测试。

## 2.14 第二章小结

这一章讲了 ASR 的主流技术路线。

最重要的脉络是：

```text
GMM-HMM
 -> DNN-HMM
 -> CTC / AED / RNN-T
 -> Conformer
 -> Whisper / 大规模语音模型
```

如果只记几个关键词：

- GMM-HMM：传统拼装系统。
- DNN-HMM：用神经网络增强传统系统。
- CTC：简单高效，解决无帧级对齐训练。
- AED：encoder-decoder，语言建模强。
- RNN-T：实时流式 ASR 核心路线。
- Conformer：CNN + Transformer，现代 ASR 强主干。
- Whisper：大规模数据驱动的通用 ASR。

更重要的是，ASR 给我们理解 TTS 提供了几个关键概念：

- 对齐
- token
- chunk
- stream
- cache
- decoding
- 前后处理

后面学习 TTS 时，这些概念会反复出现。

## 2.15 本章练习

1. 为什么 ASR 训练数据通常没有“每一帧音频对应哪个字”的标注？
2. CTC 的 blank 符号解决了什么问题？
3. RNN-T 为什么比普通 AED 更适合实时识别？
4. Conformer 为什么要同时使用 CNN 和 Transformer？
5. Whisper 为什么鲁棒，但不一定天然适合低延迟流式？
6. ASR 的分段识别经验，对 TTS 长文本合成有什么启发？

## 参考资料

- Connectionist Temporal Classification: https://www.cs.toronto.edu/~graves/icml_2006.pdf
- RNN Transducer, Google Research: https://research.google/pubs/recurrent-neural-network-transducer-for-audio-visual-speech-recognition/
- Conformer, Google Research: https://research.google/pubs/conformer-convolution-augmented-transformer-for-speech-recognition/
- Introducing Whisper, OpenAI: https://openai.com/index/whisper/
- Whisper paper, Robust Speech Recognition via Large-Scale Weak Supervision: https://arxiv.org/abs/2212.04356

---

# 第三章：TTS 主流技术路线

## 3.1 TTS 到底在生成什么

TTS 是 Text To Speech：

```text
文字 -> 音频
```

但这句话太粗了。真正理解 TTS，要先问一个更具体的问题：

> 模型到底在生成什么？

不同年代、不同技术路线，对这个问题的答案不一样。

有的系统不“生成”声音，而是从录音库里拼接。

有的系统生成声学参数。

有的系统生成 Mel 频谱。

有的系统直接生成 waveform。

有的系统生成 audio code，再由 audio tokenizer decoder 还原成 waveform。

所以 TTS 技术路线可以先按“中间表示”来理解：

```text
文本 -> 拼接片段 -> waveform
文本 -> 声学参数 -> vocoder -> waveform
文本 -> Mel spectrogram -> neural vocoder -> waveform
文本 -> latent / flow / diffusion -> waveform 或 Mel
文本 -> audio codes -> audio tokenizer decoder -> waveform
```

MOSS-TTS-Nano 属于最后一类：

```text
文本 token + speaker prompt audio codes
 -> 生成 audio codes
 -> audio tokenizer decode
 -> waveform
```

## 3.2 TTS 系统的共同 pipeline

无论模型路线怎么变，一个 TTS 系统通常都有这些部分：

```text
原始文本
 -> 文本规范化
 -> 文本 tokenizer / phoneme converter
 -> 声学模型
 -> vocoder 或 audio decoder
 -> 音频后处理
 -> 输出 wav/mp3/m4a
```

### 文本规范化

TTS 对输入文本非常敏感。

例子：

```text
2026/05/08
3.14
AI
Taiguanglin：
你好……
```

这些文本对人来说自然，但对模型来说可能有歧义。

文本规范化要决定：

- 数字怎么读？
- 日期怎么读？
- 英文字母怎么读？
- 标点是不是停顿？
- 换行是不是断句？
- 省略号是不是长停顿？
- 纯标签是不是应该读出来？

我们在 MOSSTTSKit 里新增 `TextNormalizer`，就是因为 MOSS-TTS-Nano 对省略号、连续破折号、中文引号、换行、结尾悬空标点比较敏感。

### 文本表示

TTS 的文本输入可以有几种形式：

- 字符：直接用汉字、字母。
- 音素：先转成 phoneme。
- 拼音：中文 TTS 常见。
- BPE / SentencePiece token：更接近 LLM。
- 混合表示：文本 token + speaker token + style token。

MOSS-TTS-Nano 使用 SentencePiece tokenizer。它不是传统中文 TTS 那种纯拼音输入，而是更接近语言模型 tokenizer。

### 声学模型

声学模型负责从文本条件生成某种声音表示。

传统神经 TTS 里，它通常生成 Mel。

Codec LM TTS 里，它生成 audio code。

Diffusion / Flow TTS 里，它可能在连续 latent 空间里生成声学表示。

### Vocoder / Decoder

如果声学模型生成 Mel，就需要 vocoder：

```text
Mel -> waveform
```

如果声学模型生成 audio codes，就需要 audio tokenizer decoder：

```text
audio codes -> waveform
```

这一步决定最终音质、采样率、实时性和部署成本。

## 3.3 传统路线一：拼接式 TTS

早期 TTS 很多是拼接式。

它的基本思路是：

```text
录很多人的声音片段
 -> 建一个语音片段库
 -> 输入文本后找合适片段
 -> 拼接成完整语音
```

比如把音节、音素、词、短语录下来，合成时从库里选择最合适的片段。

### 优点

- 如果录音库质量高，局部音质可以非常自然。
- 不需要复杂神经网络。
- 可控性强。

### 缺点

- 录音成本高。
- 音库巨大。
- 拼接边界容易不自然。
- 新音色要重新录制。
- 情绪、风格、语速变化能力有限。

拼接式 TTS 适合固定场景，比如导航、播报、客服模板。但它不适合今天这种“任意文本、任意音色、自然表达”的需求。

## 3.4 传统路线二：参数式 TTS

参数式 TTS 不直接拼接录音，而是预测一组声学参数，再用 vocoder 合成音频。

典型流程：

```text
文本分析
 -> 预测声学参数
 -> vocoder
 -> waveform
```

参数可能包括：

- 基频 F0
- 频谱包络
- 时长
- 清浊音

### 优点

- 比拼接式更灵活。
- 模型和音库更小。
- 可以调节音高、语速等参数。

### 缺点

- 声音容易机械。
- 细节损失明显。
- 自然度有限。

参数式 TTS 是现代神经 TTS 的前身。它让人们意识到：声音可以被拆成可预测的声学表示。

## 3.5 神经 TTS 路线一：Tacotron / Tacotron 2

Tacotron 是神经 TTS 的重要转折点。

它把 TTS 变成了一个 seq2seq 问题：

```text
文本序列 -> Mel spectrogram 序列
```

Tacotron 2 的经典流程是：

```text
文本
 -> encoder-decoder with attention
 -> Mel spectrogram
 -> WaveNet vocoder
 -> waveform
```

### 它解决了什么

Tacotron 让模型自己学习文字到声学特征的映射，不再需要那么多手工规则。

它能生成比传统参数式 TTS 更自然的语音。

### 它的问题

Tacotron 是自回归模型。

它一步一步生成 Mel frame：

```text
Mel frame 1 -> Mel frame 2 -> Mel frame 3 -> ...
```

这带来几个问题：

- 推理速度较慢。
- attention 可能不稳定。
- 长文本可能漏读、重复、提前结束。
- 流式和批量加速都不容易。

我们在 MOSS-TTS-Nano 里遇到的“漏读、吞字、提前停止”，和 Tacotron 的 attention 问题不是同一个结构原因，但从产品体验上很像：模型生成序列时，没有稳定覆盖完整文本。

## 3.6 神经 TTS 路线二：FastSpeech / FastSpeech 2

FastSpeech 试图解决 Tacotron 自回归慢、不稳定的问题。

它是非自回归 TTS。

典型流程：

```text
文本
 -> encoder
 -> duration predictor
 -> length regulator
 -> decoder
 -> Mel spectrogram
 -> vocoder
 -> waveform
```

关键模块是 duration predictor 和 length regulator。

模型先预测每个 token 应该持续多久，然后把文本表示展开成和 Mel 序列长度接近的表示，再并行生成 Mel。

### 优点

- 推理快。
- 并行生成。
- 比 Tacotron 更稳定。
- 更容易控制语速、音高、能量。

### 缺点

- 需要时长信息或 teacher model。
- 韵律自然度依赖 duration / pitch / energy 建模质量。
- 表达力通常不如更强的生成式模型。

FastSpeech 的思想非常工程化：

> 与其让模型自己在 attention 里学对齐，不如显式预测时长，把对齐问题拆出来。

这和 ASR 里的 CTC、RNN-T 一样，本质都在处理序列长度不一致和对齐问题。

## 3.7 神经 TTS 路线三：VITS

VITS 是端到端神经 TTS 中非常重要的一条路线。

它把多个组件放到一个统一框架里：

```text
文本
 -> posterior encoder / prior encoder
 -> flow
 -> decoder
 -> waveform
```

VITS 的关键词包括：

- VAE
- normalizing flow
- adversarial training
- stochastic duration predictor
- end-to-end waveform generation

不用一开始就被这些词吓住。可以先把 VITS 理解成：

> 它试图把文本到声音的多个步骤联合起来训练，让模型直接生成更自然的 waveform。

### 优点

- 音质好。
- 端到端。
- 推理速度可以较快。
- 在开源 TTS 中非常常见。
- 适合做单说话人、多说话人、本地部署。

### 缺点

- 训练复杂。
- 多语言、强控制、零样本克隆能力取决于具体改造。
- 长文本仍然需要切分。
- 和 LLM 式 prompt 控制不是同一路线。

很多开源中文 TTS、角色音色、二次元音色项目都基于 VITS 或其变体。

## 3.8 Diffusion TTS

Diffusion TTS 借鉴了图像生成里的扩散模型思想。

它的大致过程是：

```text
从噪声开始
 -> 多步去噪
 -> 得到声学表示或 waveform
```

在 TTS 中，diffusion 可以用来生成：

- Mel spectrogram
- latent representation
- waveform

### 优点

- 生成质量高。
- 表达力强。
- 可以很好地建模复杂分布。
- 适合高质量语音、歌声、情绪等任务。

### 缺点

- 多步采样可能较慢。
- 工程部署复杂。
- 实时流式不一定容易。

后来很多工作会尝试减少采样步数，或者改成 flow matching，让生成更快。

## 3.9 Flow Matching TTS

Flow Matching 是近年很重要的生成路线。

相比 diffusion 反复去噪，flow matching 学习一个从简单分布到数据分布的连续变换过程。

可以粗略理解为：

```text
噪声 / latent
 -> 学习一条流动路径
 -> 目标语音表示
```

F5-TTS 就是这类路线中很受关注的开源 TTS。

它强调：

- 简洁架构。
- 高质量语音。
- 零样本 voice clone。
- 不依赖复杂 phoneme pipeline。

Flow / diffusion 类 TTS 的特点是音质和自然度可能很好，但部署时要关注：

- 采样步数。
- 推理速度。
- 显存或内存。
- 流式能力。
- ONNX / CoreML / 端侧支持情况。

## 3.10 Codec LM TTS：音频 token 时代

现在进入和 MOSS-TTS-Nano 最相关的路线：Codec LM TTS。

它的核心思想是：

> 先用 audio tokenizer 把音频变成离散 token，再用语言模型生成这些 token。

流程：

```text
训练 audio tokenizer:
waveform -> audio codes -> waveform

训练 TTS model:
文本 token + prompt audio codes -> target audio codes

推理:
文本 -> 生成 audio codes -> decode 成 waveform
```

这条路线的代表包括：

- VALL-E
- SoundStorm
- NaturalSpeech 系列中的相关离散表示思路
- MOSS-TTS-Nano
- Qwen3-TTS

### 为什么这条路线重要

因为它让 TTS 更像 LLM。

文本是 token。

音频也变成 token。

那么模型可以学习：

```text
给定前面的 token，预测后面的 token。
```

这带来几个变化：

- 语音克隆可以变成 prompt。
- 多说话人可以变成条件生成。
- 音色、风格、情绪可以和语言模型能力结合。
- 流式生成更自然。
- 多模态模型可以统一处理文本和语音。

### 代价

这条路线也带来工程复杂度：

- 需要 audio tokenizer。
- audio codes 通常有多个 codebook。
- 自回归生成可能慢。
- KV cache 需要管理。
- 采样策略会影响稳定性。
- 长文本容易累积内存和错误。
- tokenizer、标点、prompt 格式都变得非常关键。

MOSSTTSKit 的复杂性主要来自这里。它不是简单调用一个 `text_to_wav.onnx`，而是要复刻官方 runtime 的一条生成链路。

## 3.11 Vocoder 和 Audio Tokenizer 的区别

很多初学者会把 vocoder 和 audio tokenizer 混在一起。

它们都能把某种中间表示变成音频，但角色不同。

### Vocoder

Vocoder 通常做：

```text
Mel spectrogram -> waveform
```

它是一个 decoder，但输入通常是连续频谱特征。

常见 vocoder：

- WaveNet
- WaveRNN
- MelGAN
- HiFi-GAN
- BigVGAN

### Audio Tokenizer

Audio tokenizer 通常有 encode 和 decode 两个方向：

```text
waveform -> discrete audio codes
discrete audio codes -> waveform
```

它不仅能还原音频，还能把音频压缩成离散 token，让语言模型学习。

常见相关技术：

- SoundStream
- EnCodec
- DAC
- MOSS Audio Tokenizer
- Qwen3-TTS-Tokenizer

一句话区分：

> Vocoder 是把声学特征还原成声音；audio tokenizer 是把声音离散化成 token，并能从 token 还原声音。

MOSS-TTS-Nano 依赖的是 audio tokenizer，而不是传统 Mel vocoder。

## 3.12 语音克隆在不同路线中的实现方式

语音克隆不是单一技术，它在不同 TTS 路线中实现方式不同。

### 多说话人 embedding

早期多说话人 TTS 会给每个 speaker 一个 embedding。

```text
文本 + speaker id -> 语音
```

优点是稳定。

缺点是只能用训练过的 speaker。

### Speaker encoder

后来出现 speaker encoder。

```text
参考音频 -> speaker embedding
文本 + speaker embedding -> 语音
```

这样可以做零样本或少样本克隆。

### Prompt audio codes

Codec LM 路线里，语音克隆可以更像 LLM prompt：

```text
参考音频 -> audio codes
文本 token + prompt audio codes -> 目标 audio codes
```

MOSS-TTS-Nano 就接近这种方式。

MOSSTTSKit 中：

```swift
makeSpeaker(name:referenceAudioURL:)
```

会把参考音频 encode 成 `referenceAudioCodes`。后续 `speak(...)` 时，这些 codes 会参与模型输入。

这也解释了为什么语音克隆会更占内存：

- 参考音频越长，prompt audio codes 越多。
- prefill 序列越长，KV cache 越大。
- 每个 chunk 都需要带上 speaker prompt。

所以我们后来加入了：

- `maxReferenceAudioDuration`
- `maxReferenceAudioPromptFrames`

这是工程上必须做的边界控制。

## 3.13 长文本 TTS 为什么难

长文本 TTS 不是把短句合成循环很多次那么简单。

它难在几个方面。

### 语义和韵律

长文本里有段落、章节、对话、旁白、引用。

模型要决定：

- 哪里停顿？
- 哪句话语气更重？
- 引号怎么读？
- 省略号怎么处理？
- 段落之间停多久？

### 上下文长度

模型上下文有限。

一次性输入太长，可能：

- 超过模型能力。
- 生成慢。
- 内存高。
- 漏读。
- 提前结束。

### 音频长度

文本越长，生成的 audio codes 越多。

如果把所有 codes 都放在内存里，内存会明显上涨。

MOSSTTSKit 之前内存冲高，就是因为长文本路径里有过“先生成完整 codes，再统一 decode”的设计。后来改成增量 decode，内存才明显下降。

### 拼接自然度

如果切得太碎，听起来会不连贯。

如果切得太长，模型可能漏读或爆内存。

所以长文本切分要遵守：

- 优先句末标点。
- 再按从句标点。
- 最后才按 token 预算。
- 段间加自然停顿。
- 不要在词中间硬切。

这也是 MOSSTTSKit 的 `maxTextTokensPerChunk` 和 chunking 策略存在的原因。

## 3.14 TTS 路线对比

可以用一张表理解主流 TTS 路线：

| 路线 | 中间表示 | 优点 | 缺点 | 典型代表 |
| --- | --- | --- | --- | --- |
| 拼接式 TTS | 录音片段 | 局部自然、可控 | 音库大、不灵活 | 传统播报系统 |
| 参数式 TTS | 声学参数 | 小、可控 | 机械感强 | HMM 参数 TTS |
| Tacotron | Mel | 自然度提升、端到端感强 | 慢、attention 不稳 | Tacotron 2 |
| FastSpeech | Mel | 快、稳定、可控 | 依赖时长建模 | FastSpeech 2 |
| VITS | latent / waveform | 音质好、端到端 | 训练复杂、prompt 能力有限 | VITS |
| Diffusion TTS | Mel / latent / waveform | 质量高、表达力强 | 多步采样、部署重 | Grad-TTS、NaturalSpeech 2 |
| Flow Matching TTS | latent / acoustic representation | 质量高、结构简洁 | 工程生态仍在发展 | F5-TTS |
| Codec LM TTS | discrete audio codes | 像 LLM、适合 prompt 和克隆 | runtime 复杂、采样敏感 | VALL-E、MOSS-TTS-Nano、Qwen3-TTS |

## 3.15 MOSS-TTS-Nano 在 TTS 路线中的位置

现在可以更准确地定位 MOSS-TTS-Nano。

它不是：

```text
文本 -> Mel -> vocoder
```

也不是：

```text
文本 -> waveform
```

而是：

```text
文本 token + prompt audio codes
 -> global / local generation
 -> generated audio codes
 -> MOSS Audio Tokenizer decode
 -> waveform
```

它的特点：

- Codec LM 路线。
- 轻量，面向本地和 CPU 友好部署。
- 使用 ONNX Runtime。
- 支持内置音色。
- 支持参考音频克隆。
- 使用 audio tokenizer encode/decode。
- 生成过程包含 prefill、decode step、local decoder、sampler。

MOSSTTSKit 做的事情，就是把这套官方 runtime 翻译成 Swift Package：

```text
模型下载
 -> tokenizer 加载
 -> manifest 读取
 -> speaker prompt 构造
 -> ONNX sessions
 -> audio code generation
 -> incremental audio decode
 -> WAV / stream 输出
```

这也是为什么它比普通 API wrapper 难得多。它封装的是一条生成系统，而不是一个单模型调用。

## 3.16 Qwen3-TTS 在 TTS 路线中的位置

Qwen3-TTS 也属于 Codec LM / discrete speech token 路线。

根据官方资料，它使用 Qwen3-TTS-Tokenizer-12Hz，并采用 discrete multi-codebook LM。它强调：

- 多语言。
- voice clone。
- voice design。
- instruction control。
- hybrid streaming。
- 更强的表达和控制。

可以这样理解：

```text
MOSS-TTS-Nano: 轻量、本地、可集成
Qwen3-TTS: 更大、更强、控制能力更完整
```

两者的共同点是都把音频变成离散 token 来生成。

不同点在于：

- 模型规模。
- 控制能力。
- 部署成本。
- runtime 复杂度。
- 是否适合端侧。
- 是否适合服务端批量或实时生成。

对 TTSMate 来说，MOSS-TTS-Nano 的意义是：它更有机会作为本地包被集成进 macOS 应用。

Qwen3-TTS 的意义是：它代表更强能力的方向，可以作为未来能力对标。

## 3.17 从 TTS 路线反看我们遇到的问题

我们在 MOSSTTSKit 开发中遇到的问题，并不是孤立 bug。

它们都能映射回 TTS 技术路线。

### tokenizer 问题

Codec LM 对 token 极其敏感。

文本 token 不一致，生成就不一致。

所以我们后来选择直接对齐官方 SentencePiece 行为。

### 省略号和标点问题

模型不是人。

`……`、`---`、`Taiguanglin：` 这类输入可能让模型认为 prompt 没结束，或者进入奇怪生成状态。

所以要有 `TextNormalizer`。

### 长文本漏读

长文本是序列生成的典型难题。

切太长，模型容易不稳定。

切太碎，听感不连贯。

所以要按标点和 token 预算切。

### 内存暴涨

Codec LM 路线中，audio codes、KV cache、decode cache 都会占内存。

长文本和 voice clone 都会放大这个问题。

所以要：

- 控制参考音频 prompt 长度。
- 增量 decode。
- chunk 级释放中间状态。
- 避免一次性保存所有中间 codes。

### 每次生成略有差异

生成式模型有采样。

temperature、top-k、seed、随机源都会影响结果。

这不是 bug，而是生成式模型的本质特征。工程上只能通过 seed、采样参数、回归样本来控制稳定性。

## 3.18 第三章小结

这一章讲了 TTS 的主要技术路线。

最重要的脉络是：

```text
拼接式 TTS
 -> 参数式 TTS
 -> Tacotron / Mel spectrogram
 -> FastSpeech / 非自回归 Mel
 -> VITS / 端到端 waveform
 -> Diffusion / Flow
 -> Codec LM / audio token
```

如果只记一句话：

> TTS 技术的核心演进，是不断寻找更好的声音中间表示；MOSS-TTS-Nano 选择的是 audio code，所以它更像一个生成音频 token 的语言模型。

MOSS-TTS-Nano 的定位现在很清楚：

- 它属于 Codec LM TTS。
- 它依赖 audio tokenizer。
- 它生成 audio codes，而不是 Mel。
- 它适合轻量本地部署。
- 它的工程难点集中在 tokenizer、prompt、采样、流式、长文本和内存管理。

下一章会进一步抽象 ASR 和 TTS 的共同点：tokenization、alignment、sampling、streaming、cache、chunking。理解这些共同问题之后，再深入 MOSS-TTS-Nano 的 ONNX 模型结构就会顺很多。

## 3.19 本章练习

1. Tacotron 和 FastSpeech 最大区别是什么？
2. 为什么 FastSpeech 要显式预测 duration？
3. VITS 为什么被称为端到端 TTS？
4. Vocoder 和 audio tokenizer 有什么区别？
5. Codec LM TTS 为什么更适合做零样本 voice clone？
6. 为什么 MOSS-TTS-Nano 不是传统 `文本 -> Mel -> vocoder` 路线？
7. 为什么长文本 TTS 不能简单地把 1 万字一次性丢给模型？

## 参考资料

- Tacotron 2, Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions: https://arxiv.org/abs/1712.05884
- FastSpeech, Fast Robust and Controllable Text to Speech: https://arxiv.org/abs/1905.09263
- FastSpeech 2, Fast and High-Quality End-to-End Text to Speech: https://arxiv.org/abs/2006.04558
- VITS, Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech: https://arxiv.org/abs/2106.06103
- VALL-E, Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers: https://arxiv.org/abs/2301.02111
- NaturalSpeech 2, Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers: https://arxiv.org/abs/2304.09116
- F5-TTS, A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching: https://arxiv.org/abs/2410.06885
- OpenMOSS / MOSS-TTS-Nano: https://github.com/OpenMOSS/MOSS-TTS-Nano
- QwenLM / Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS

---

# 第四章：语音模型中的共同问题

## 4.1 为什么要单独讲共同问题

前两章分别讲了 ASR 和 TTS。

ASR 是：

```text
音频 -> 文字
```

TTS 是：

```text
文字 -> 音频
```

方向相反，但底层问题高度相似。

如果只按模型名字学习，很容易变成记名词：

- CTC
- RNN-T
- Tacotron
- FastSpeech
- VITS
- Whisper
- VALL-E
- MOSS-TTS-Nano
- Qwen3-TTS

但真正做工程时，你会发现反复遇到的其实是同一组问题：

- tokenization
- normalization
- alignment
- sampling
- chunking
- streaming
- cache
- memory
- latency
- evaluation

这一章把这些共同问题抽出来讲。理解这些，后面深入 MOSS-TTS-Nano 的 ONNX 输入输出时，脑子里会有骨架。

## 4.2 Tokenization：把世界变成模型能处理的符号

深度学习模型喜欢张量。

语言模型喜欢 token。

语音模型既要处理文字，也要处理声音，所以会遇到两类 tokenizer：

```text
text tokenizer
audio tokenizer
```

### 文本 tokenizer

文本 tokenizer 把文字转成 token id。

```text
你好，世界。
 -> [id1, id2, id3, id4, ...]
```

常见方式：

- 字符级 tokenizer
- phoneme tokenizer
- BPE
- WordPiece
- SentencePiece

中文 TTS 里，有些模型先转拼音或音素；MOSS-TTS-Nano 使用 SentencePiece，把文本直接转成 token id。

这就是为什么 MOSSTTSKit 必须认真对齐官方 tokenizer。token id 只要变了，模型看到的输入就变了，声音也会变。

### 音频 tokenizer

音频 tokenizer 把连续音频变成离散 audio codes。

```text
waveform
 -> audio tokenizer encode
 -> audio codes
```

再通过 decoder 还原：

```text
audio codes
 -> audio tokenizer decode
 -> waveform
```

MOSS-TTS-Nano 使用 MOSS Audio Tokenizer。Qwen3-TTS 使用 Qwen3-TTS-Tokenizer-12Hz。

音频 tokenizer 的意义很大：

> 它把音频从连续信号变成离散 token，让 TTS 可以像 LLM 一样建模。

## 4.3 Normalization：模型输入不是用户输入

用户输入是给人看的。

模型输入是给模型看的。

这两者不一定相同。

例子：

```text
用户输入：Taiguanglin：
模型更安全输入：Taiguanglin.
```

```text
用户输入：他说完以后……

我沉默了。
模型更安全输入：他说完以后。 我沉默了。
```

### ASR 的 normalization

ASR 里 normalization 通常在输出后做：

- 标点恢复
- 数字格式化
- 热词修正
- 繁简转换
- 大小写处理

### TTS 的 normalization

TTS 里 normalization 通常在输入前做：

- 数字读法
- 日期读法
- 单位读法
- 标点停顿
- 换行边界
- 特殊符号
- 中英混排

MOSSTTSKit 的 `TextNormalizer` 现在处理的是最确定的一层：标点和边界。以后如果继续工程化，可以扩展到数字、日期、英文缩写和书名号等场景。

## 4.4 Alignment：音频和文字长度不一样怎么办

语音模型绕不开对齐。

一句短文字：

```text
你好。
```

读出来可能有几百毫秒。

一段长文字读出来可能是几分钟。

文字 token 数和音频帧数不是一一对应的。

### ASR 中的对齐

ASR 要把很多音频帧压缩成较少文字 token：

```text
audio frames -> text tokens
```

CTC 用 blank 和路径折叠解决对齐。

RNN-T 用 blank 和 prediction network 解决流式对齐。

AED 用 attention 学习对齐。

### TTS 中的对齐

TTS 要把较少文字 token 展开成很多音频帧：

```text
text tokens -> acoustic frames / audio codes
```

Tacotron 用 attention。

FastSpeech 显式预测 duration。

Codec LM TTS 通过自回归生成 audio codes，让模型自己决定生成长度。

这也是为什么 `maxGeneratedFrames` 很敏感。它不是“文字长度”，而是允许模型最多生成多少 audio frames。太小会截断，太大可能浪费时间或生成异常尾音。

## 4.5 Sampling：生成式模型为什么不完全确定

ASR 多数时候追求最可能文本。

TTS 则不只是找“唯一正确答案”。同一句话可以有不同读法：

- 快一点
- 慢一点
- 温柔一点
- 激动一点
- 停顿长一点
- 音高略有变化

所以现代 TTS 经常是概率生成。

常见采样参数：

- temperature
- top-k
- top-p
- repetition penalty
- seed

### temperature

temperature 控制随机性。

低 temperature 更稳定，但可能机械。

高 temperature 更有变化，但可能不稳定。

### top-k / top-p

top-k 只从概率最高的 k 个候选里采样。

top-p 只从累计概率达到 p 的候选集合里采样。

### seed

seed 让随机过程可复现。

我们在 MOSSTTSKit 中对齐官方 NumPy PCG64 随机源，就是为了让默认 seed 行为更接近官方 runtime。

## 4.6 Chunking：长序列必须分块

语音任务经常遇到长序列。

ASR 中：

```text
长音频 -> 分段识别 -> 拼接文本
```

TTS 中：

```text
长文本 -> 分段合成 -> 拼接音频
```

分块不是临时补丁，而是语音工程里的常规策略。

### 好的 chunking 应该满足什么

对 TTS 来说，一个好的分块策略应该：

- 优先按句号、问号、感叹号切。
- 其次按逗号、顿号、分号、冒号切。
- 最后按 token 预算兜底。
- 尽量不要切断词或语义单元。
- 段间加入自然停顿。
- 每个 chunk 不要超过模型稳定范围。

MOSSTTSKit 的长文本策略就是按这个方向做的。

## 4.7 Streaming：流式不是回调那么简单

很多人一开始会把流式理解成：

```text
生成完一段 -> 调用 callback
```

但真正的流式更深。

流式 ASR：

```text
麦克风不断输入
 -> 模型增量识别
 -> UI 增量显示文字
```

流式 TTS：

```text
文本输入
 -> 模型增量生成 audio codes
 -> decoder 增量输出 samples
 -> 播放器边收边播
```

真正的流式需要：

- 模型支持 step 推理。
- runtime 能保留 cache。
- decoder 能增量 decode。
- 调用方能消费增量音频。
- 取消时能及时停止。
- 内存能随着 chunk 生命周期释放。

MOSSTTSKit 里的 `speakStream(...)` 和 incremental audio decode，就是朝这个方向做。

## 4.8 Cache：为什么 prefill 和 decode step 很重要

Transformer 自回归生成时，每一步都要看前面 token。

如果每一步都重新计算全部历史，成本会越来越高。

所以会用 KV cache。

```text
prefill: 一次性处理 prompt，得到初始 KV cache
decode step: 每次只处理新 token，同时更新 KV cache
```

MOSS-TTS-Nano 的 ONNX 模型里有：

- `moss_tts_prefill.onnx`
- `moss_tts_decode_step.onnx`

这就是典型自回归生成结构。

语音克隆时，参考音频 codes 会进入 prompt。prompt 越长，prefill 越重，KV cache 越大。这解释了为什么我们要限制 voice clone 参考音频长度和 prompt frames。

## 4.9 Memory：内存不只是模型大小

做本地语音模型时，很多人只看模型文件大小。

但运行时内存还包括：

- ONNX Runtime session
- 输入 tensor
- 输出 tensor
- KV cache
- audio codes
- decoder cache
- waveform samples
- WAV / M4A 编码 buffer
- app 层保存的 Data

长文本时，最大风险通常不是模型本身，而是中间结果堆积。

MOSSTTSKit 里做过几项优化：

- 长文本按 chunk 合成。
- audio codes 增量 decode。
- 不再先堆完整 codes 再统一 decode。
- ONNX 输入减少不必要复制。
- voice clone prompt frames 加上默认上限。

工程经验是：

> 本地语音模型要把每个阶段的生命周期想清楚，中间数据能流走就不要堆住。

## 4.10 Evaluation：ASR 有指标，TTS 更需要耳朵

ASR 常见指标：

- WER: Word Error Rate
- CER: Character Error Rate
- RTF: Real Time Factor
- latency

TTS 也有指标：

- MOS: Mean Opinion Score
- speaker similarity
- intelligibility
- naturalness
- RTF
- memory peak

但 TTS 的用户感知非常复杂。一个文件能生成，不代表生成质量好。

要听：

- 有没有漏读？
- 有没有多读？
- 有没有奇怪音节？
- 停顿是否自然？
- 音色是否稳定？
- 长文本是否疲劳？
- 克隆是否像？

所以 MOSSTTSKit 不只需要单元测试，也需要固定试听样本。

## 4.11 第四章小结

语音模型的共同问题可以概括为：

```text
normalize -> tokenize -> align -> generate/decode -> stream -> evaluate
```

MOSS-TTS-Nano 的工程难点都能落到这些共同问题上：

- tokenizer 对齐官方。
- TextNormalizer 管理风险文本。
- maxGeneratedFrames 控制生成长度。
- chunking 处理长文本。
- streaming 降低内存。
- KV cache 支持自回归推理。
- 听感样本验证真实质量。

这一章是后面深入 MOSS-TTS-Nano 的桥。

---

# 第五章：MOSS-TTS-Nano 深入

## 5.1 MOSS-TTS-Nano 的定位

MOSS-TTS-Nano 是 OpenMOSS 团队发布的轻量 TTS 模型。

根据官方项目说明，MOSS-TTS-Nano 的目标包括：

- 轻量本地部署。
- ONNX Runtime CPU 推理。
- 支持流式合成。
- 支持 48 kHz stereo 输出。
- 支持内置音色。
- 支持参考音频语音克隆。

它不是一个传统 `text -> mel -> vocoder` 模型。

它属于：

```text
Audio Tokenizer + Codec Language Model
```

也就是：

```text
文本 token + prompt audio codes
 -> 生成 audio codes
 -> audio tokenizer decode
 -> waveform
```

## 5.2 模型文件怎么分工

MOSS-TTS-Nano ONNX 版本通常包含多组模型。

### TTS 主模型

```text
moss_tts_prefill.onnx
moss_tts_decode_step.onnx
moss_tts_local_decoder.onnx
moss_tts_local_cached_step.onnx
moss_tts_local_fixed_sampled_frame.onnx
```

它们负责文本到 audio codes 的生成。

### Audio tokenizer

```text
moss_audio_tokenizer_encode.onnx
moss_audio_tokenizer_decode_full.onnx
moss_audio_tokenizer_decode_step.onnx
```

它们负责：

```text
参考音频 -> audio codes
audio codes -> waveform
```

### tokenizer 和 manifest

```text
tokenizer.model
browser_poc_manifest.json
```

`tokenizer.model` 负责文本 tokenizer。

`browser_poc_manifest.json` 保存内置音色、prompt audio codes、生成默认参数等信息。

## 5.3 为什么有 global 和 local

从 ONNX 输入输出看，MOSS-TTS-Nano 不是一个简单 decoder。

它有 global 阶段和 local 阶段。

可以粗略理解为：

```text
global model:
    处理文本和历史上下文，生成 global_hidden

local model:
    基于 global_hidden，生成 text logits 和 audio logits

sampler:
    根据 logits 和随机数采样出下一帧 audio codes
```

MOSS-TTS-Nano 每一帧 audio code 不是单个 token，而是多 codebook 的 token 组合。官方 ONNX 输出里可以看到：

```text
audio_logits [batch, 16, 1024]
```

这说明一帧里有 16 个 audio token 通道，每个通道大约从 1024 个候选里选择。

## 5.4 Prefill 和 decode step

自回归模型通常分两步：

```text
prefill
decode step
```

### prefill

prefill 处理初始 prompt。

在 MOSS-TTS-Nano 中，prompt 包括：

- 文本 token。
- speaker prompt audio codes。
- 一些特殊行或通道结构。

prefill 输出：

- `global_hidden`
- `present_key_*`
- `present_value_*`

这些 key/value 就是后续 decode step 用的 cache。

### decode step

decode step 每次处理新的输入，并更新 cache。

它输入：

- 新 step 的 `input_ids`
- `past_valid_lengths`
- `past_key_*`
- `past_value_*`

输出：

- 新的 `global_hidden`
- 更新后的 `present_key_*`
- 更新后的 `present_value_*`

这就是为什么 MOSSTTSKit 需要管理一串 ONNX tensor，而不是简单调用一个函数。

## 5.5 Local decoder 和 sampler

global 阶段给出隐藏状态后，还要通过 local decoder 得到 logits。

```text
global_hidden
 + text_token_id
 + audio_prefix_token_ids
 -> text_logits
 -> audio_logits
```

然后 sampler 根据 logits 采样下一帧 audio codes。

`moss_tts_local_fixed_sampled_frame.onnx` 做的就是一部分固定采样逻辑，它输入：

- `global_hidden`
- `repetition_seen_mask`
- `assistant_random_u`
- `audio_random_u`

输出：

- `should_continue`
- `frame_token_ids`

`should_continue` 决定模型是否继续生成。

我们之前遇到的截断、漏读问题，有一部分就和 `should_continue`、frame cap、采样顺序和文本边界有关。

## 5.6 Audio tokenizer encode 和 decode

语音克隆需要把参考音频转成 codes。

```text
reference wav
 -> moss_audio_tokenizer_encode.onnx
 -> referenceAudioCodes
```

合成结束后，要把生成的 audio codes 转成 waveform：

```text
generated audio codes
 -> moss_audio_tokenizer_decode_step.onnx
 -> Float samples
```

MOSSTTSKit 一开始如果把所有 generated codes 都堆起来再 decode，长文本内存就会涨。

后来改成：

```text
生成一批 codes
 -> 增量 decode
 -> 输出 samples
 -> 释放中间 codes
```

这就是长文本内存下降的关键。

## 5.7 内置音色来自哪里

MOSS-TTS-Nano 的内置音色不是 MOSSTTSKit 硬编码的。

它们来自模型目录中的 manifest。

MOSSTTSKit 应该提供 API：

```swift
let speakers = await tts.availableSpeakers
```

调用方不应该自己读 manifest。原因是：

- manifest 格式是模型内部细节。
- 包应该封装模型差异。
- 后续模型更新时调用方不需要改。

这也是我们之前把“获取全部音色”做成包 API 的原因。

## 5.8 语音克隆为什么更吃内存

内置音色通常已经有固定 prompt codes。

语音克隆时，用户传入参考音频，包需要：

```text
读取音频
 -> 重采样 / 声道处理
 -> audio tokenizer encode
 -> 保存 referenceAudioCodes
 -> speak 时作为 prompt
```

风险在于：

- 参考音频太长。
- encode 输入 waveform 太大。
- 生成时 prompt 太长。
- prefill KV cache 变大。

所以 MOSSTTSKit 加了两个控制：

```swift
maxReferenceAudioDuration
maxReferenceAudioPromptFrames
```

这不是牺牲功能，而是本地模型工程必须有边界。

## 5.9 为什么文本规范化对 MOSS-TTS-Nano 很重要

Codec LM TTS 对 prompt 很敏感。

像这些输入：

```text
Taiguanglin：
你好……
---《圣经》
```

人类知道它们的意思，但模型可能把它们当成异常 prompt。

因此 MOSSTTSKit 做了 `TextNormalizer`：

- 省略号转成句子停顿。
- 连续破折号转成句子停顿。
- 中文引号作为排版符号移除。
- 非空换行转成边界。
- 结尾冒号转成句子结束。

这不是要改变用户文本，而是把文本变成模型更稳定的输入。

## 5.10 MOSS-TTS-Nano 的优点和局限

### 优点

- 轻量。
- 本地部署友好。
- ONNX Runtime 可集成。
- 支持 audio tokenizer 和 voice clone。
- 支持流式思路。
- 对 Swift Package 封装有现实价值。

### 局限

- 音质和控制力不一定比大模型强。
- 生成有随机性。
- 长文本需要工程切分。
- 文本规范化需要不断完善。
- voice clone 质量受参考音频影响。
- 对官方 runtime 对齐要求高。

## 5.11 MOSSTTSKit 和官方 runtime 的关系

MOSSTTSKit 的目标不是“重新发明 MOSS-TTS-Nano”。

它应该做的是：

```text
官方 runtime 行为
 -> Swift / ONNX Runtime 可用封装
 -> Apple 平台产品 API
```

因此每次遇到差异，优先问：

- 官方 Python 怎么做？
- 输入 token 是否一致？
- 随机源是否一致？
- sampler 顺序是否一致？
- audio decode 是否按长度裁剪？
- 生成边界是否一致？

这就是我们后来做 tokenizer 对齐、PCG64 随机源、audio_lengths 裁剪、增量 decode 的原因。

## 5.12 第五章小结

MOSS-TTS-Nano 可以理解成：

```text
SentencePiece text tokenizer
 + prompt audio codes
 + autoregressive codec LM
 + multi-codebook audio token generation
 + MOSS Audio Tokenizer decoder
```

它的工程难点不是单个 ONNX 模型，而是一条完整生成链路。

MOSSTTSKit 的价值就在于把这条链路封装成 Swift 项目可以直接使用的 API。

---

# 第六章：Qwen3-TTS 深入

## 6.1 Qwen3-TTS 的定位

Qwen3-TTS 是 Qwen 系列中的语音合成模型。

根据 QwenLM 官方仓库，Qwen3-TTS 在 2026-01-22 发布 0.6B 和 1.7B 系列模型，并提供：

- base model
- instruct model
- pre-trained audio tokenizer
- custom voice checkpoints
- voice clone
- voice design
- hybrid streaming

它和 MOSS-TTS-Nano 一样，也属于离散语音 token 方向。

但 Qwen3-TTS 规模更大，目标更偏通用能力和可控生成。

## 6.2 Qwen3-TTS 的基本路线

官方资料描述 Qwen3-TTS 使用：

```text
Qwen3-TTS-Tokenizer-12Hz
Discrete multi-codebook LM
Dual-Track hybrid streaming
```

可以粗略理解为：

```text
文本 / 指令 / prompt
 -> Qwen3-TTS language model
 -> discrete speech tokens
 -> Qwen3-TTS tokenizer decode
 -> waveform
```

它也是 Codec LM 路线。

## 6.3 Qwen3-TTS-Tokenizer-12Hz

12Hz 表示 audio tokenizer 的时间帧率大约是每秒 12 帧。

这很关键。

音频原始 waveform 很长：

```text
48 kHz stereo = 每秒 96000 个采样值
```

如果 tokenizer 把它压缩成每秒 12 帧的离散 token，语言模型要处理的序列长度就大幅下降。

离散 audio token 的质量决定：

- 还原音质。
- speaker 信息保留。
- 情绪信息保留。
- 模型生成难度。
- 流式输出延迟。

## 6.4 Discrete multi-codebook LM

multi-codebook 指一帧音频不是一个 token，而是一组 token。

这和 MOSS-TTS-Nano 的 `audio_logits [batch, 16, 1024]` 很像。

为什么要多 codebook？

因为单个 token 很难表达足够丰富的音频细节。

多 codebook 可以像分层压缩：

- 前几个 codebook 表达主要内容。
- 后几个 codebook 补充细节。

这样既能离散化，又能保留音质。

## 6.5 Base 和 Instruct

base model 更接近基础生成模型。

instruct model 更强调指令控制。

在 TTS 里，instruct 的意义是：

```text
用自然语言描述如何说
```

例子：

```text
请用温柔、平静、略带纪录片旁白感的声音读下面这段话。
```

这和传统 TTS 的参数控制不同。

传统 TTS 可能调：

- pitch
- speed
- energy
- speaker id

Qwen3-TTS 这类模型则可能通过自然语言 instruction 控制风格。

## 6.6 Voice clone 和 voice design

Qwen3-TTS 强调两种能力：

```text
voice clone
voice design
```

voice clone 是模仿参考音频。

voice design 是按描述创造或控制声音。

区别：

```text
voice clone:
    给我一段参考音频，我要像这个人。

voice design:
    给我一个声音描述，我要一个符合描述的新声音。
```

这说明 TTS 正在从“复刻声音”走向“设计声音”。

## 6.7 Hybrid streaming

Qwen3-TTS 官方提到 Dual-Track hybrid streaming。

可以先用工程直觉理解：

> 模型既要保持生成质量，又要尽快输出可播放音频。

纯离线生成可以等所有 token 都生成完，再解码。

流式生成则希望：

```text
生成一部分 speech tokens
 -> 解码一部分 audio
 -> 播放
 -> 继续生成
```

hybrid streaming 通常就是在质量、延迟、上下文之间做平衡。

这和 MOSSTTSKit 的方向一致：长文本和流式处理不是附加功能，而是语音产品的核心工程能力。

## 6.8 Qwen3-TTS 的优势

Qwen3-TTS 的优势大致在：

- 模型规模更大。
- 多语言能力更强。
- 指令控制更自然。
- 支持 voice design。
- 更适合复杂风格表达。
- 更接近通用语音生成模型。

如果产品需要：

- 声音风格精细控制。
- 多语言强表现。
- 服务端高质量 TTS。
- 通过自然语言设计音色。

Qwen3-TTS 是值得重点研究的方向。

## 6.9 Qwen3-TTS 的工程代价

更强能力通常意味着更高工程成本：

- 模型更大。
- 端侧部署更难。
- GPU 或高性能推理环境更重要。
- runtime 更复杂。
- 商业和许可证使用方式要确认。
- 本地 App 集成成本更高。

对 TTSMate 这类 macOS 本地工具来说，Qwen3-TTS 未必是第一阶段的最佳本地模型，但它很适合作为能力天花板和未来方向。

## 6.10 Qwen3-TTS 和 MOSS-TTS-Nano 的共同点

共同点：

- 都属于 speech token / codec LM 路线。
- 都需要 audio tokenizer。
- 都是生成 discrete audio tokens。
- 都支持 voice clone。
- 都要处理采样、streaming、prompt、长文本。

不同点：

- Qwen3-TTS 更大。
- Qwen3-TTS 更强调 instruct 和 voice design。
- MOSS-TTS-Nano 更强调轻量本地和 ONNX CPU。
- MOSS-TTS-Nano 更适合 Swift Package 本地封装。

## 6.11 第六章小结

Qwen3-TTS 是更大规模、更强控制力的 Codec LM TTS。

它代表的方向是：

```text
语音合成从“选择音色读文本”
 -> 走向“用自然语言控制声音生成”
```

MOSS-TTS-Nano 和 Qwen3-TTS 不应该只看成两个竞争模型。更好的理解是：

```text
MOSS-TTS-Nano: 轻量本地落地样本
Qwen3-TTS: 更强语音生成能力样本
```

研究两者，可以同时理解本地部署和大模型语音生成的两端。

---

# 第七章：MOSS-TTS-Nano 与 Qwen3-TTS 对比

## 7.1 为什么要对比

对比不是为了简单判断谁更好。

它们代表了同一技术方向的两种产品取向：

```text
MOSS-TTS-Nano: 小、轻、本地、ONNX、可集成
Qwen3-TTS: 大、强、可控、instruct、voice design
```

理解这个差异，才能判断什么时候用哪个模型。

## 7.2 技术路线对比

| 维度 | MOSS-TTS-Nano | Qwen3-TTS |
| --- | --- | --- |
| 路线 | Codec LM / audio token | Codec LM / speech token |
| audio tokenizer | MOSS Audio Tokenizer | Qwen3-TTS-Tokenizer-12Hz |
| 生成对象 | multi-codebook audio codes | discrete multi-codebook speech tokens |
| 推理方式 | ONNX Runtime | 官方 PyTorch/Transformers 生态为主 |
| 目标 | 轻量本地 TTS | 高能力语音生成 |
| 语音克隆 | 支持 | 支持 |
| voice design | 较弱或不强调 | 强调 |
| Swift 本地封装 | 可行 | 难度更高 |

## 7.3 部署对比

MOSS-TTS-Nano 的优势是本地部署。

它有 ONNX 模型，并且官方目标就是 CPU runtime。

这对 Apple 平台很重要：

- Swift Package 可以集成 ONNX Runtime。
- 模型可以自动下载。
- 可以在 macOS App 内离线生成。
- 用户隐私更好。
- 成本更可控。

Qwen3-TTS 更适合：

- 服务端。
- GPU 推理。
- 云端 API。
- 高质量生成任务。
- 多语言和强控制任务。

## 7.4 音质和能力对比

通常来说，更大模型在表达能力上有优势。

Qwen3-TTS 可能在这些方面更强：

- 复杂语气。
- 风格控制。
- voice design。
- 多语言。
- 指令跟随。

MOSS-TTS-Nano 的优势在：

- 轻量。
- 成本低。
- 本地可控。
- 更适合包封装和 App 内集成。

对 TTSMate 来说，第一阶段最重要的不是绝对最强，而是：

```text
能稳定生成
能本地运行
能控制内存
能封装 API
能支持长文本
```

这正是 MOSS-TTS-Nano 更合适的地方。

## 7.5 长文本对比

无论 MOSS-TTS-Nano 还是 Qwen3-TTS，长文本都不是简单一次性输入就完事。

长文本一定要考虑：

- 文本切分。
- 段落停顿。
- 生成长度上限。
- 语音一致性。
- 内存释放。
- 输出文件编码。

大模型也会有上下文限制和生成稳定性问题。

所以长文本处理应该放在应用和 SDK 层，而不是完全依赖模型自己解决。

MOSSTTSKit 的长文本策略，未来也可以迁移到其他 TTS 后端：

```text
TextNormalizer
 -> sentence chunking
 -> streaming synthesis
 -> audio stitching
 -> progress reporting
```

## 7.6 语音克隆对比

MOSS-TTS-Nano 语音克隆更像：

```text
参考音频 -> audio codes -> prompt conditioning
```

Qwen3-TTS 也支持 voice clone，但能力更强，可能对参考音频、指令和风格融合有更丰富的建模。

工程上，两者都会遇到：

- 参考音频质量。
- 参考音频长度。
- 背景噪声。
- 说话人一致性。
- prompt cache 成本。
- 克隆失败时的 fallback。

所以 voice clone 不是只做一个 API，而要做完整流程：

```text
音频校验
 -> 长度裁剪
 -> loudness normalization
 -> encode
 -> speaker object
 -> speak
 -> 质量回归
```

## 7.7 产品选择建议

如果目标是 TTSMate 本地版：

优先 MOSS-TTS-Nano。

原因：

- 本地部署可行。
- ONNX Runtime 可封装。
- 成本低。
- 隐私好。
- 包 API 可控。

如果目标是云端高质量生成：

可以研究 Qwen3-TTS。

原因：

- 控制力更强。
- 多语言和风格更好。
- voice design 更适合创作场景。

如果目标是长期技术储备：

两个都要研究。

MOSS-TTS-Nano 帮你掌握本地 runtime。

Qwen3-TTS 帮你理解未来语音生成大模型能力。

## 7.8 第七章小结

MOSS-TTS-Nano 和 Qwen3-TTS 是同一代 Codec LM TTS 思路下的不同取向。

一句话：

> MOSS-TTS-Nano 是适合本地产品化的轻量路线，Qwen3-TTS 是面向更强生成和控制能力的大模型路线。

做产品时，不是模型越大越好，而是要看：

- 用户设备。
- 延迟要求。
- 内存预算。
- 是否离线。
- 是否需要 voice design。
- 是否能接受云端。
- 是否要 Swift Package 集成。

---

# 第八章：从官方 runtime 到 Swift Package

## 8.1 为什么不能只看模型文件

很多模型项目看起来只是几个 ONNX 文件。

但真正能跑起来，靠的是 runtime。

runtime 包含：

- 输入怎么构造。
- tokenizer 怎么用。
- prompt 怎么拼。
- cache 怎么传。
- sampler 怎么采样。
- 什么时候停止。
- audio codes 怎么 decode。
- 输出怎么裁剪。

MOSSTTSKit 的开发，本质上是把官方 runtime 翻译成 Swift runtime。

## 8.2 翻译 runtime 的基本方法

正确方法不是直接猜 ONNX 输入。

应该按顺序做：

1. 跑官方 demo，确认模型能正常输出。
2. 打印官方中间输入输出。
3. 用工具 inspect ONNX 输入输出 shape。
4. 在 Swift 侧复刻同样 tensor。
5. 对齐 tokenizer。
6. 对齐 random source。
7. 对齐 sampler。
8. 对齐 audio decoder。
9. 建回归样本。

我们之前走过弯路，比如围绕 `tokenizer.json` 花了很多时间。后来回到第一性原理，直接使用 `tokenizer.model` 对齐 SentencePiece，问题才稳定下来。

## 8.3 Package API 应该隐藏什么

调用方不应该知道：

- ONNX 输入名字。
- KV cache tensor 列表。
- manifest 格式。
- audio codebook 数量。
- audio tokenizer decode cache。
- sampler 随机数 shape。

调用方应该看到：

```swift
let tts = try await MOSSTTSKit()
let speakers = await tts.availableSpeakers
let result = try await tts.speak(text: "你好。")
```

这才是 Swift Package 的价值。

## 8.4 模型自动下载

MOSSTTSKit 应该负责模型下载。

因为调用方关心的是能力：

```text
我想用 MOSS-TTS-Nano 生成语音
```

而不是：

```text
我应该去哪下载哪个 ONNX 文件，放到哪个目录，manifest 是否齐全
```

自动下载需要明确：

- 默认下载源。
- 默认缓存目录。
- 模型目录结构。
- 断点或重试策略。
- 用户如何指定本地目录。
- 如何清理缓存。

README 里必须写清楚默认下载到哪里。

## 8.5 Speaker API

speaker API 要封装两类音色：

```text
builtin speakers
cloned speakers
```

内置音色来自 manifest。

克隆音色来自参考音频 encode。

调用方应该这样用：

```swift
let speakers = await tts.availableSpeakers
let speaker = speakers.first
let result = try await tts.speak(text: "你好。", speaker: speaker)
```

而不是自己解析模型内部文件。

## 8.6 长文本 API

调用方最好直接调用：

```swift
try await tts.speak(text: longText)
```

而不是自己把 1 万字切成固定 200 字。

原因：

- 固定字数切分可能切断语义。
- 不了解 tokenizer token 预算。
- 不知道模型 frame 限制。
- 不容易处理停顿。
- 不容易统一进度。

SDK 内部更知道如何按标点、token、模型能力切。

## 8.7 流式 API

应该同时提供：

```swift
speak(...)
speakStream(...)
speakToFile(...)
```

三者服务不同场景。

`speak(...)` 适合短文本或调用方想一次拿完整 samples。

`speakStream(...)` 适合播放器边生成边播、进度显示、长文本低内存。

`speakToFile(...)` 适合长文本直接落盘，避免调用方保存巨大 `Data`。

对长文本和有声书来说，`speakToFile(...)` 往往是最稳的 API。

## 8.8 错误设计

SDK 应该给明确错误。

例如：

- 模型文件不存在。
- tokenizer 缺失。
- manifest 缺失。
- 文本为空。
- 参考音频无法读取。
- ONNX session 失败。
- audio decode 失败。
- 取消生成。

不要让调用方看到一堆底层 ONNX 错误却不知道怎么处理。

## 8.9 测试策略

一个语音 SDK 至少要有三类测试。

### 单元测试

测试纯逻辑：

- TextNormalizer
- options validation
- tensor utils
- speaker model

### 集成测试

测试真实模型路径：

- tokenizer encode
- prefill
- decode step
- audio decode
- speak short text
- speak progress cancel

### 试听回归

生成固定 wav：

- 中文短句。
- 中英混排。
- 长文本。
- 省略号。
- 引号。
- voice clone。
- 不同 speaker。

TTS 不能只靠 assert，必须听。

## 8.10 第八章小结

从官方 runtime 到 Swift Package 的关键，不是“能跑一次”，而是：

```text
对齐官方行为
 -> 隐藏内部复杂性
 -> 提供稳定 API
 -> 管理模型和资源
 -> 做长文本和流式
 -> 建测试和试听回归
```

MOSSTTSKit 的目标就是把 MOSS-TTS-Nano 从 demo 变成可被 TTSMate、VideoHero 这类项目稳定引用的包。

---

# 第九章：长文本、流式、内存与产品化

## 9.1 为什么产品化比 demo 难

模型 demo 通常只生成一句话：

```text
你好，这是一个测试音频。
```

产品要面对的是：

- 1 万字章节。
- 用户粘贴的奇怪标点。
- 中英混排。
- 书名号、引号、破折号。
- 多个文件批量生成。
- 中途取消。
- 后台任务。
- 内存限制。
- 进度 UI。
- 文件保存。

所以产品化不是把 demo 包一层 UI。

产品化是把模型放进真实使用环境。

## 9.2 长文本默认策略

对长文本 TTS，推荐原则是：

```text
调用方传完整文本
SDK 内部负责切分和流式处理
```

不要让调用方按固定字数切。

因为模型稳定性更接近 token 和语义边界，而不是字符数。

推荐切分顺序：

1. 文本规范化。
2. 按句末标点切。
3. 超长句按从句标点切。
4. 仍超长时按 token 预算切。
5. 每个 chunk 单独合成。
6. 段间插入短暂停顿。
7. 增量写入或增量播放。

## 9.3 流式处理的内存意义

流式不是只为了快。

它还为了省内存。

非流式长文本可能是：

```text
生成所有 audio codes
 -> 保存所有 codes
 -> 一次性 decode
 -> 保存所有 samples
 -> 写文件
```

流式应该是：

```text
生成一小段 codes
 -> decode 一小段 samples
 -> 播放或写入
 -> 释放 codes
 -> 继续
```

内存占用取决于最长 chunk 和必要 cache，而不是整篇文章。

这就是 MOSSTTSKit 长文本内存从十几 GB 降下来的核心原因。

## 9.4 speak、speakStream、speakToFile 怎么选

### speak

适合：

- 短句。
- 测试。
- 调用方确实需要完整 samples。

不适合：

- 超长文本。
- 批量有声书生成。
- 内存敏感场景。

### speakStream

适合：

- 边生成边播放。
- UI 显示进度。
- 中途取消。
- 长文本但由调用方消费 samples。

### speakToFile

适合：

- 长文本落盘。
- 批量生成。
- 有声书章节。
- 减少调用方内存压力。

对 TTSMate 这种应用，长章节最好逐步转向 `speakToFile(...)` 或真正消费 `speakStream(...)`，避免最后保存文件时 app 层再复制巨大 Data。

## 9.5 进度如何设计

TTS 进度不容易精确。

因为模型生成多少 frames 不完全等于文本进度。

可以提供近似进度：

- 当前 chunk index。
- total chunks。
- 当前 frame step。
- total estimated frames。
- generated sample count。
- chunk boundary。

UI 可以显示：

```text
第 3 / 18 段，约 42%
```

不要承诺精确到字级，除非模型本身提供强对齐信息。

## 9.6 中途取消

取消要从上到下生效：

- 调用方点击取消。
- progress callback 返回 false。
- 当前 chunk 停止生成。
- 不再发送 chunk boundary。
- 不再进入下一个 chunk。
- 释放中间状态。

这就是为什么我们修了 `speak(...)` 的取消语义。

## 9.7 文件保存时的内存峰值

用户观察到 12000 字生成时，平时内存在 1.6G 到 2G，最后保存文件时冲到 3.2G。

这很典型。

因为最后保存可能发生：

- samples 数组还在。
- WAV Data 又复制一份。
- 文件系统写入 buffer。
- app 状态保留结果。

优化方向：

- SDK 提供 `speakToFile(...)`。
- WAV writer 支持增量写入。
- 调用方不要先拿完整 Data 再写。
- 写完及时释放当前任务结果。

## 9.8 Voice clone 的内存策略

voice clone 要特别控制。

参考音频处理应该：

- 限制最大时长。
- 限制 prompt frames。
- 尽量只读需要的音频范围。
- encode 后不要保留原始 waveform。
- speak 时不要把超长 prompt 传进 prefill。

MOSSTTSKit 的 `maxReferenceAudioDuration` 和 `maxReferenceAudioPromptFrames` 就是为此存在。

## 9.9 产品质量检查清单

一个 TTS 功能上线前，至少检查：

- 短中文是否正常。
- 长中文是否漏读。
- 省略号是否稳定。
- 破折号是否稳定。
- 引号和书名号是否自然。
- 中英混排是否可接受。
- 内置音色数量是否完整。
- voice clone 是否限制参考长度。
- 取消是否有效。
- 进度是否递增。
- 长文本内存是否可接受。
- 生成失败是否给用户明确提示。

## 9.10 第九章小结

长文本和流式是 TTS 产品化的核心。

一句话：

> 短句 demo 看模型能力，长文本批量生成看工程能力。

MOSSTTSKit 要成为可用包，必须持续关注：

- chunking
- streaming
- memory
- cancellation
- progress
- file writing
- regression samples

---

# 第十章：测试、试听样本与质量评估

## 10.1 为什么 TTS 测试不能只靠单元测试

普通软件测试可以 assert：

```text
输入 A，输出 B。
```

TTS 不完全行。

同一句话可能有多个合理输出。

音频是否好听，也不是简单字符串能判断。

所以 TTS 需要多层测试：

```text
单元测试
 -> 集成测试
 -> 官方 runtime 对比
 -> 试听样本
 -> 产品场景回归
```

## 10.2 单元测试

单元测试适合确定性逻辑。

MOSSTTSKit 中适合单测的内容：

- `TextNormalizer`
- options validation
- tensor shape utils
- audio format conversion
- speaker metadata
- model path availability
- tokenizer regression ids
- random source sequence

这些测试应该跑得快，失败时定位明确。

## 10.3 集成测试

集成测试验证真实模型路径。

例如：

- 模型是否可加载。
- manifest 是否能读。
- builtin speakers 是否完整。
- prefill 是否能运行。
- decode step 是否能运行。
- sampler 是否输出合理 frame。
- audio tokenizer 是否能 decode。
- `speak(...)` 是否生成非空 samples。
- progress cancel 是否有效。

集成测试比单元测试慢，但能防止 runtime 链路断掉。

## 10.4 官方 runtime 对比

当 Swift 输出和官方 Python 不一致时，优先做对比。

对比顺序：

1. 文本 normalize 后是否一致。
2. tokenizer ids 是否一致。
3. prompt audio codes 是否一致。
4. prefill 输入 rows 是否一致。
5. random numbers 是否一致。
6. sampler 输出 frame 是否一致。
7. audio decode 长度是否一致。

不要一上来听音频猜。

先把中间结果对齐。

## 10.5 试听样本库

TTS 必须建立固定试听样本。

建议按类别保存：

```text
samples/
  short_cn/
  punctuation/
  long_text/
  mixed_language/
  speakers/
  voice_clone/
```

每个样本要记录：

- 输入文本。
- speaker。
- options。
- seed。
- 生成日期。
- 模型版本。
- 是否通过人工试听。

## 10.6 推荐回归文本

至少保留这些类型：

### 短中文

```text
你好，这是一个包内测试音频。
```

### 多句中文

```text
我突然醒来了。不知道睡了多久。我完完全全清醒了，精神饱满，感觉敏捷。
```

### 省略号

```text
她的双手握着，就像她平常睡觉时那样……

我一点都不想再睡了。
```

### 结尾冒号

```text
Taiguanglin：
```

### 长文本

选真实书籍段落，覆盖：

- 引号。
- 书名号。
- 破折号。
- 段落换行。
- 多个短句。
- 一个超长句。

### 中英混排

```text
今天我们测试 MOSS-TTS-Nano 和 Qwen3-TTS 的 mixed language 能力。
```

## 10.7 质量评价维度

试听时不要只说“好”或“不好”。

要按维度记录：

- intelligibility: 能不能听清。
- completeness: 有没有漏读。
- insertion: 有没有多读奇怪字。
- pronunciation: 发音是否正确。
- prosody: 停顿和语气是否自然。
- speaker similarity: 克隆像不像。
- stability: 多次生成是否稳定。
- noise: 是否有噪声或破音。
- duration: 是否异常过长或过短。

这样问题才能回到工程上。

例如：

```text
问题：Taiguanglin：生成 12 秒异常音频
维度：duration abnormal + prompt instability
根因：结尾悬空冒号
修复：TextNormalizer 转成完整句子
测试：TextNormalizerTests + 试听样本
```

## 10.8 内存和性能测试

TTS 包也要测性能。

建议记录：

- 短句耗时。
- 1000 字耗时。
- 10000 字耗时。
- 峰值内存。
- 平稳内存。
- 保存文件时峰值。
- voice clone encode 峰值。
- speak with cloned speaker 峰值。

最好每次大改后用同一组文本和同一台机器对比。

## 10.9 版本发布前检查

发布 MOSSTTSKit 新版本前建议检查：

- `swift build` 通过。
- `swift test` 通过。
- README 和中文 README 更新。
- CHANGELOG 更新。
- Package 版本更新。
- TTSMate 本地包验证。
- 至少一组短文本试听。
- 至少一组长文本试听。
- voice clone 基本试听。
- 内存峰值没有明显回退。

## 10.10 第十章小结

TTS 质量不是靠一次成功生成证明的。

它需要：

```text
确定性单测
 + 真实模型集成测试
 + 官方 runtime 对比
 + 试听样本库
 + 性能和内存回归
```

这套体系建立起来后，MOSSTTSKit 才不会每次遇到问题都靠临时猜测。

## 全书小结

这本书从 ASR/TTS 全景开始，逐步讲到 MOSS-TTS-Nano 的工程落地。

核心脉络是：

```text
ASR/TTS 基础
 -> 主流技术路线
 -> 共同底层问题
 -> MOSS-TTS-Nano 架构
 -> Qwen3-TTS 对比
 -> Swift Package 工程化
 -> 长文本和流式产品化
 -> 测试与质量评估
```

如果你想真正掌握 MOSS-TTS-Nano，不要只看模型文件，也不要只看 MOSSTTSKit 代码。

要把三层连起来：

```text
模型原理
官方 runtime
产品级 SDK
```

MOSSTTSKit 就是这三层之间的桥。

## 后续可继续补充的附录

后面还可以继续补：

- 附录 A：MOSS-TTS-Nano ONNX 输入输出逐项解释。
- 附录 B：MOSSTTSKit 源码导读。
- 附录 C：TTSMate 接入最佳实践。
- 附录 D：常见异常文本和处理策略。
- 附录 E：如何阅读官方 Python runtime。
- 附录 F：如何做一次模型输出对比实验。

## 参考资料

- OpenMOSS / MOSS-TTS-Nano: https://github.com/OpenMOSS/MOSS-TTS-Nano
- OpenMOSS / MOSS-TTS: https://github.com/OpenMOSS/MOSS-TTS
- QwenLM / Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- ONNX Runtime: https://github.com/microsoft/onnxruntime
- SentencePiece: https://github.com/google/sentencepiece
- VALL-E: https://arxiv.org/abs/2301.02111
- Whisper: https://arxiv.org/abs/2212.04356
- VITS: https://arxiv.org/abs/2106.06103
- FastSpeech 2: https://arxiv.org/abs/2006.04558
- F5-TTS: https://arxiv.org/abs/2410.06885
