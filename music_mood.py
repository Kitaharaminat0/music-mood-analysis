#!/usr/bin/env python3
"""音乐情绪分析：音频 → 五组情绪倾向。

思路：用现成的 14 类音乐情绪分类模型（wav2vec2 底座）拿原始概率，
再做两件事让它更像人话：
  1. 14 类细标签归并成 5 组（低落/温暖/激昂/攻击/史诗）
  2. 用乐理（调式、BPM）做有门限的修正，救回被音色误导的歌

用法:
    python music_mood.py song.mp3
    python music_mood.py song.mp3 --json
    python music_mood.py song.mp3 --model-dir ./music_moods --ffmpeg /usr/bin/ffmpeg
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

SR = 16000

# 官方模型仓库；本地已有权重时用 --model-dir 指过去，省一次 360MB 下载
DEFAULT_MODEL = "StanislavKo28/music_moods_classification"

# 14 类 → 5 组。分组依据是"听感上是不是同一类东西"，不是严格的心理学量表
GROUPS = {
    "低落": ["sad", "sentimental", "dark"],
    "温暖": ["romantic", "relaxing"],
    "激昂": ["uplifting", "euphoric", "energetic", "happy"],
    "攻击": ["angry", "scary"],
    "史诗": ["epic", "mysterious", "glamorous"],
}

CN = {
    "sad": "悲伤", "sentimental": "感伤", "relaxing": "放松", "romantic": "浪漫",
    "happy": "欢快", "euphoric": "狂喜", "uplifting": "励志", "energetic": "有活力",
    "dark": "阴暗", "mysterious": "神秘", "epic": "史诗", "scary": "惊悚",
    "angry": "愤怒", "glamorous": "华丽",
}

# Krumhansl 调性轮廓（听觉实验得出的音级稳定度分布）
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

FEEL = {
    "低落": "像压着的、不太说话的情绪",
    "温暖": "柔软、愿意靠近的那种",
    "激昂": "往上顶的、想跟着动的",
    "攻击": "带刺的、有冲劲的",
    "史诗": "铺得很大、有画面感的",
}


def find_ffmpeg(explicit: str | None = None) -> str:
    """按 显式参数 → 环境变量 → PATH → 常见安装位置 的顺序找 ffmpeg。"""
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise SystemExit(f"指定的 ffmpeg 不存在: {explicit}")

    env = os.environ.get("FFMPEG_PATH")
    if env and os.path.isfile(env):
        return env

    found = shutil.which("ffmpeg")
    if found:
        return found

    for guess in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ):
        if os.path.isfile(guess):
            return guess

    raise SystemExit(
        "找不到 ffmpeg。请安装后加进 PATH，或用 --ffmpeg / --ffmpeg-path 指定，\n"
        "也可设置环境变量 FFMPEG_PATH。\n"
        "  macOS: brew install ffmpeg\n"
        "  Ubuntu: sudo apt install ffmpeg\n"
        "  Windows: https://www.gyan.dev/ffmpeg/builds/"
    )


def resolve_model(spec: str) -> str:
    """本地目录存在就用本地，否则当成 Hugging Face 仓库 id。"""
    if os.path.isdir(spec):
        return spec
    return spec


def load_audio(path: str, ffmpeg: str, seconds: float = 180.0, sr: int = SR) -> np.ndarray:
    """用 ffmpeg 解成单声道 float32 波形。只在这里解一次，后面复用。"""
    if not os.path.isfile(path):
        raise SystemExit(f"文件不存在: {path}")

    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "audio.pcm")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", path,
               "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le"]
        if seconds and seconds > 0:
            cmd += ["-t", str(seconds)]
        cmd.append(raw)
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
        if proc.returncode != 0:
            msg = proc.stderr.decode("utf-8", "replace").strip()
            raise SystemExit(f"ffmpeg 解码失败:\n{msg}")
        return np.fromfile(raw, dtype=np.int16).astype(np.float32) / 32768.0


def detect_key(x: np.ndarray, sr: int = SR) -> dict | None:
    """色度图 + Krumhansl 轮廓相关 → 主音与调式，附带置信度。"""
    try:
        import librosa
    except ImportError:
        return None

    try:
        harm = librosa.effects.harmonic(x, margin=3)
        chroma = librosa.feature.chroma_cqt(y=harm, sr=sr, bins_per_octave=36)
    except Exception:
        return None

    prof = chroma.mean(axis=1)
    if not np.all(np.isfinite(prof)) or prof.sum() <= 0:
        return None
    prof = prof / prof.sum()

    best_score, best_key = -2.0, None
    for shift in range(12):
        rotated = np.roll(prof, -shift)
        for name, ref in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            ref_n = ref / ref.sum()
            c = float(np.corrcoef(rotated, ref_n)[0, 1])
            if not np.isfinite(c):
                continue
            if c > best_score:
                best_score = c
                best_key = (KEYS[shift], "大调" if name == "major" else "小调")

    if best_key is None:
        return None
    return {"key": best_key[0], "mode": best_key[1], "score": round(best_score, 3)}


def detect_tempo(x: np.ndarray, sr: int = SR) -> int:
    """估计 BPM。兼容 librosa 0.10.1 前后两套 API。"""
    try:
        import librosa
    except ImportError:
        return 0

    onset = librosa.onset.onset_strength(y=x, sr=sr)

    # librosa 0.10 起 beat.tempo 迁到 feature.rhythm.tempo，1.0 会删掉旧名。
    # 直接 import 而不用 getattr 探测——librosa 的懒加载会让 hasattr 判断失真。
    tempo_fn = None
    try:
        from librosa.feature.rhythm import tempo as tempo_fn
    except Exception:
        from librosa.beat import tempo as tempo_fn

    try:
        bpm = float(tempo_fn(onset_envelope=onset, sr=sr)[0])
    except TypeError:  # 更老的签名不接受关键字
        bpm = float(tempo_fn(onset, sr=sr)[0])
    if not np.isfinite(bpm):
        return 0
    return int(round(bpm))


def model_probs(x: np.ndarray, model_spec: str, sr: int = SR,
                window: int = 30, hop: int = 15, max_chunks: int = 10) -> tuple[dict, int]:
    """分段推理后取平均。接收已解码波形，不重复读盘。

    按 30s 窗 / 15s 步长切，最多取 max_chunks 段——整首歌都跑一遍没必要，
    情绪是全局属性，均匀取样比从头截 3 分钟更有代表性。
    """
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import torch

    fe = AutoFeatureExtractor.from_pretrained(model_spec)
    model = AutoModelForAudioClassification.from_pretrained(model_spec)
    model.eval()

    # 注意：id2label 的键可能是 int 也可能是 str，转一次免得 KeyError
    id2label = {int(k): v for k, v in dict(model.config.id2label).items()}

    win, step = window * sr, hop * sr
    chunks = []
    for start in range(0, max(1, x.size - win // 2), step):
        seg = x[start:start + win]
        if seg.size >= sr * 5:  # 太短的段不算
            chunks.append(seg)
    if not chunks:
        chunks = [x[:win]]

    if len(chunks) > max_chunks:
        idx = np.linspace(0, len(chunks) - 1, max_chunks).astype(int)
        chunks = [chunks[i] for i in idx]

    probs = np.zeros(len(id2label))
    with torch.no_grad():
        for seg in chunks:
            inputs = fe(seg, sampling_rate=sr, return_tensors="pt", padding=True)
            probs += torch.softmax(model(**inputs).logits, dim=-1).numpy()[0]
    probs /= len(chunks)

    return {id2label[i]: float(probs[i]) for i in range(len(id2label))}, len(chunks)


def group_scores(probs: dict) -> dict:
    """14 类 → 5 组，再归一化。"""
    g = {
        name: sum(probs.get(l, 0.0) for l in labels)
        for name, labels in GROUPS.items()
    }
    tot = sum(g.values()) or 1.0
    return {k: v / tot for k, v in g.items()}


def apply_theory(groups: dict, mode: str, bpm: int,
                 score: float = 0.0, min_score: float = 0.75) -> dict:
    """乐理修正。

    两条规则，来由都写在这儿，别乱改幅度：

    1) 调式修正，只在置信度 >= min_score 时生效。
       色度检测在混音上很容易把大小调搞反（实测 Oasis 的 Live Forever
       被判成小调，置信度只有 0.52）。宁可不动，也别动错。
       而且只压"攻击"、不大幅压"低落"——大调一样能写悲伤（Radiohead），
       早先一版把大调的低落压了 20%，Fake Plastic Trees 直接从"悲伤"变"温柔"。

    2) 向上救援，不依赖调式，一直生效。
       失真吉他的音色天然像"攻击"，但很多这类歌情绪其实是"希望"。
       当节奏够快（>=110）且史诗+激昂已经够重（>=35%）时，往激昂那头拉。
       用 Breed 验证过不会误伤：它的史诗+激昂只有 18%，够不到门槛。
    """
    g = dict(groups)

    if mode and score >= min_score:
        if mode == "大调":
            g["攻击"] = g.get("攻击", 0) * 0.72
            g["激昂"] = g.get("激昂", 0) * 1.12
        elif mode == "小调":
            g["低落"] = g.get("低落", 0) * 1.15
            g["攻击"] = g.get("攻击", 0) * 1.05

    if bpm and bpm >= 110 and (g.get("史诗", 0) + g.get("激昂", 0)) >= 0.35:
        g["激昂"] = g.get("激昂", 0) * 1.55
        g["攻击"] = g.get("攻击", 0) * 0.72
        g["史诗"] = g.get("史诗", 0) * 1.05

    tot = sum(g.values()) or 1.0
    return {k: v / tot for k, v in g.items()}


def load_calibration(path: str | None) -> dict:
    """人工校准覆盖：{"groups": {"攻击": 0.8}} 表示把该组乘 0.8。文件缺失就跳过。"""
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def apply_calibration(groups: dict, calib: dict) -> dict:
    if not calib.get("groups"):
        return groups
    g = dict(groups)
    for k, mult in calib["groups"].items():
        if k in g:
            try:
                g[k] *= float(mult)
            except (TypeError, ValueError):
                continue
    tot = sum(g.values()) or 1.0
    return {k: v / tot for k, v in g.items()}


def describe(groups: dict, mode: str, bpm: int, key: str) -> str:
    top = sorted(groups.items(), key=lambda kv: -kv[1])[:2]
    parts = []
    if mode:
        parts.append(mode + (f"（{key} 调）" if key else ""))
    if bpm:
        parts.append(f"{bpm} BPM")
    line1 = "、".join(parts)
    line2 = "、".join(f"{k} {v * 100:.0f}%" for k, v in top)
    return f"{line1} → {line2}；听感上{FEEL.get(top[0][0], '')}"


def analyze(path: str, model_spec: str, ffmpeg: str, seconds: float,
            calib_path: str | None) -> dict:
    x = load_audio(path, ffmpeg, seconds)
    probs, chunks = model_probs(x, model_spec)

    keyinfo = detect_key(x) or {}
    bpm = detect_tempo(x)

    raw_groups = group_scores(probs)
    groups = apply_theory(raw_groups, keyinfo.get("mode", ""), bpm,
                          float(keyinfo.get("score") or 0))
    groups = apply_calibration(groups, load_calibration(calib_path))

    return {
        "file": os.path.basename(path),
        "chunks": chunks,
        "key": keyinfo.get("key"),
        "mode": keyinfo.get("mode"),
        "key_score": keyinfo.get("score"),
        "bpm": bpm,
        "groups": {k: round(v, 4) for k, v in sorted(groups.items(), key=lambda kv: -kv[1])},
        "raw_groups": {k: round(v, 4) for k, v in sorted(raw_groups.items(), key=lambda kv: -kv[1])},
        "raw14": {CN.get(k, k): round(v, 4) for k, v in
                  sorted(probs.items(), key=lambda kv: -kv[1])[:6]},
        "desc": describe(groups, keyinfo.get("mode", ""), bpm, keyinfo.get("key")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="音乐情绪分析：音频 → 五组情绪倾向",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("audio", help="音频或视频文件路径（ffmpeg 能解的格式都行）")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    ap.add_argument("--model-dir", default=os.environ.get("MOOD_MODEL", DEFAULT_MODEL),
                    help="本地模型目录，或 Hugging Face 仓库 id")
    ap.add_argument("--ffmpeg", dest="ffmpeg_path", default=None,
                    help="ffmpeg 可执行文件路径（默认自动查找）")
    ap.add_argument("--seconds", type=float, default=180.0,
                    help="最多分析多少秒（0 表示全部）")
    ap.add_argument("--calib", default=None,
                    help="人工校准 JSON，形如 calib/examples/calibration.example.json")
    args = ap.parse_args(argv)

    ffmpeg = find_ffmpeg(args.ffmpeg_path)
    result = analyze(args.audio, resolve_model(args.model_dir), ffmpeg,
                     args.seconds, args.calib)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        print(result["desc"])
        print()
        for k, v in result["groups"].items():
            bar = "█" * max(0, round(v * 40))
            print(f"  {k}  {v * 100:5.1f}%  {bar}")
        print()
        print("  （统计倾向，不是某个人的感受。同一首歌，不同人听出来不一样。）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
