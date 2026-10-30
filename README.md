# 音乐情绪分析 · Music Mood in Plain Words

> Turn an audio file into five human-sounding mood buckets — with music-theory corrections for the cases where the model gets fooled by timbre.

把一个音频文件变成五组能说出口的情绪倾向：**低落 / 温暖 / 激昂 / 攻击 / 史诗**。

```bash
python music_mood.py song.mp3
```

```
小调（A 调）、110 BPM → 攻击 66%、史诗 17%；听感上带刺的、有冲劲的

  攻击  66.0%  ██████████████████████████
  史诗  17.0%  ███████
  激昂   8.0%  ███
  低落   8.0%  ███
  温暖   1.0%
```

---

## 为什么还要做一层

现成的音乐情绪分类模型给的是 14 个细标签：`sad / sentimental / dark / romantic / relaxing / happy / euphoric / uplifting / energetic / angry / scary / epic / mysterious / glamorous`。

问题是这个粒度对人没什么用。跟朋友说"这首歌悲伤 31%、感伤 22%、阴暗 19%"——他听不出你在讲什么。他想要的是"这歌挺丧的"或者"这歌往上顶"。

所以这层做的事情就两件：

1. **归并**：14 类 → 5 组，按"听感上是不是同一类东西"分，不追求心理学量表上的严谨
2. **修正**：用调式和速度纠偏，专门对付模型被音色骗的情况

## 效果

在一个小样本上做的方向性验证（5 首，人工标注只给方向，不给百分比）：

| 歌 | 人工标注 | 输出 top2 | |
|---|---|---|---|
| Wonderful Tonight | 深情、悲伤 | 温暖 49%、低落 18% | ✅ |
| Live Forever | 激情、积极 | 激昂 30%、史诗 24% | ✅ |
| Hey Jude | 温柔、激励 | 温暖 49%、低落 32% | ✅ |
| Fake Plastic Trees | 悲伤、自卑 | 低落 40%、温暖 40% | ✅ 微弱 |
| Breed | 狂放、愤怒 | 攻击 66% | ✅ |

**5/5 方向对**。纯模型（不做乐理修正）在同样 5 首上只有 3.5/5。

样本很小，别当 benchmark 看。它的价值主要在说明"规则能救模型"，不在证明准确率。

---

## 三处关键设计

### 1. 五组映射

```
低落 = sad + sentimental + dark
温暖 = romantic + relaxing
激昂 = uplifting + euphoric + energetic + happy
攻击 = angry + scary
史诗 = epic + mysterious + glamorous
```

### 2. 调式修正，带置信度门限

大调 → 攻击 ×0.72、激昂 ×1.12；小调 → 低落 ×1.15、攻击 ×1.05。
**只在 Krumhansl 置信度 ≥ 0.75 时生效。**

在混音上判断大小调很容易错——吉他本身是谐波乐器，`librosa.effects.harmonic` 分不干净。实测 Oasis 的 Live Forever 被判成小调，置信度只有 0.52。既然判错了，宁可不动。

另外**只压"攻击"，不大幅压"低落"**。因为大调一样能写悲伤（Radiohead 就是典型）。早先一版为了让结果"更准"把大调的低落压了 20%，结果 Fake Plastic Trees 从"悲伤第一"变成"温柔第一"——修正是错的。

### 3. 向上救援

```
若 BPM ≥ 110 且 (史诗 + 激昂) ≥ 35%
  → 激昂 ×1.55，攻击 ×0.72，史诗 ×1.05
```

失真吉他的音色在声学上天然像"攻击"，但很多这类歌的情绪其实是"希望"。Oasis 那种响亮的、往前推的东西，纯模型会判成愤怒。

这条规则不依赖调式判断（因为调式本身不可靠），只看速度 + 已有情绪里史诗/激昂的分量，所以一直生效。

误伤验证：Breed 是真正的狂放愤怒，BPM 不低但史诗+激昂只有 18%，够不到 35% 的门槛，不会被拉走。

---

## 安装

```bash
git clone https://github.com/<your-name>/music-mood-analysis.git
cd music-mood-analysis
pip install -r requirements.txt
```

还需要 **ffmpeg**（命令行有就行，脚本会自动找）：

```bash
# macOS
brew install ffmpeg
# Ubuntu / Debian
sudo apt install ffmpeg
# Windows：下载后把 bin 加进 PATH
# https://www.gyan.dev/ffmpeg/builds/
```

首次运行会从 Hugging Face 下载约 360 MB 模型权重，之后走本地缓存。

## 用法

```bash
# 默认（前 180 秒，10 段均匀取样）
python music_mood.py song.mp3

# 完整 JSON
python music_mood.py song.mp3 --json

# 本地已有权重，省一次下载
python music_mood.py song.mp3 --model-dir /path/to/music_moods

# 指定 ffmpeg
python music_mood.py song.mp3 --ffmpeg /opt/homebrew/bin/ffmpeg

# 整首分析（长音频会很慢）
python music_mood.py podcast.mp3 --seconds 0

# 套上人工校准
python music_mood.py song.mp3 --calib calibration.json
```

视频文件也能直接喂（`ffmpeg` 会抽音轨）。

### 输出字段

| 字段 | 含义 |
|---|---|
| `groups` | 五组最终概率（已过乐理修正 + 校准） |
| `raw_groups` | 五组原始概率，没修正 |
| `raw14` | 模型给的 14 类 top6，中文名 |
| `key` / `mode` / `key_score` | 主音、调式、Krumhansl 置信度 |
| `bpm` | 估计速度 |
| `chunks` | 实际推理了几段 |
| `desc` | 一句话描述 |

`groups` 和 `raw_groups` 都输出是有意的——想调规则时，能直接看出哪一步改动了什么。

一份完整输出见 [`examples/sample_output.json`](examples/sample_output.json)（真实运行结果，未手工修饰）。

## 校准

模型有偏差、或者你有自己的判断标准时，写一个 JSON 覆盖：

```json
{
  "groups": {
    "攻击": 0.8,
    "史诗": 1.2
  }
}
```

乘数作用在**乐理修正之后**，然后重新归一化。模板见 `calibration.example.json`。

```bash
python music_mood.py song.mp3 --calib calibration.json
```

## 已知局限

- **不看歌词**。判断完全来自声学特征。一首旋律温柔、歌词绝望的歌，会被判成"温暖"。这是目前最大的缺口。
- **CPU 上慢**。10 段推理约 20–40 秒。慢歌体验没问题，分析长视频要等。
- **失真吉他容易偏"攻击"**。救援规则缓解了，但没根治。
- **调性检测在混音上不可靠**。已有置信度门限兜底，代价是一些歌拿不到调式修正。
- **这是统计倾向，不是感受**。同一首歌不同人听出来不一样，工具的输出不该覆盖人自己的判断。

## 模型与出处

- 情绪分类模型：[StanislavKo28/music_moods_classification](https://huggingface.co/StanislavKo28/music_moods_classification)（Apache-2.0）
- 底座：[facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- 调性估计：Krumhansl 调性轮廓 + 色度图相关性，`librosa` 实现
- 速度估计：`librosa` onset strength + tempo

模型权重不随本仓库分发，首次运行自动下载。

## License

MIT（代码部分）。模型权重遵循上游 Apache-2.0。
