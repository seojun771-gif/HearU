"""수집한 녹음 점검: 클래스별 개수(목표 대비)와 파일 형식·녹음 상태.

실행: python check_data.py            (1단계 목표 기준)
      python check_data.py --stage 2  (2단계 목표 기준)
"""
import hashlib
import sys
import wave

import numpy as np

from common import DATA_DIR, LABELS, SR

# 단계별 목표 개수 (docs/DATA_COLLECTION.md)
TARGETS = {
    1: {"fire": 30, "doorbell": 30, "unknown": 60},
    2: {"fire": 50, "doorbell": 50, "unknown": 100, "knock": 30, "baby_cry": 30},
    3: {"fire": 100, "doorbell": 100, "unknown": 300, "knock": 100, "baby_cry": 100},
}
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
QUIET_PEAK = 0.1          # 최대 진폭이 이보다 작으면 너무 작게 녹음된 것 (-20 dBFS)
MIN_SEC = 1.0


def inspect(path) -> list:
    """wav 파일의 문제점 목록. 문제가 없으면 빈 리스트."""
    try:
        with wave.open(str(path), "rb") as w:
            rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
    except Exception as e:
        return [f"읽기 실패({e})"]
    issues = []
    if (rate, ch, width) != (SR, 1, 2):
        issues.append(f"형식 {rate}Hz/{ch}ch/{width * 8}bit (16000/1/16 아님)")
    if width != 2:
        return issues
    y = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch)[:, 0] / 32768.0
    if len(y) / rate < MIN_SEC:
        issues.append(f"너무 짧음 {len(y) / rate:.1f}초")
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak >= 0.999:
        issues.append("소리가 찢어짐(클리핑)")
    elif peak < QUIET_PEAK:
        issues.append(f"소리가 작음 (최대 {20 * np.log10(peak + 1e-9):.0f} dBFS)")
    frame = rate // 20
    if len(y) >= frame * 4:
        env = np.sqrt(np.mean(y[: len(y) // frame * frame].reshape(-1, frame) ** 2, axis=1))
        if env[0] > env.max() * 0.5:
            issues.append("시작부터 소리가 남 (앞부분 여유 없음)")
        if env[-1] > env.max() * 0.5:
            issues.append("끝까지 소리가 남 (잘렸을 수 있음)")
    return issues


def main():
    stage = int(sys.argv[sys.argv.index("--stage") + 1]) if "--stage" in sys.argv else 1
    targets = TARGETS[stage]
    seen = {}
    print(f"데이터 폴더: {DATA_DIR}   ({stage}단계 목표 기준)\n")
    print(f"{'클래스':10s} {'개수':>4s} {'목표':>4s} {'남음':>4s}  문제 파일")
    for name in [LABELS[i]["name"].lower() for i in sorted(LABELS)]:
        folder = DATA_DIR / name
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXT) if folder.exists() else []
        problems = []
        for p in files:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest in seen:
                problems.append(f"{p.name}: {seen[digest]}와 같은 파일")
            seen.setdefault(digest, p.name)
            if p.suffix.lower() == ".wav":
                problems += [f"{p.name}: {msg}" for msg in inspect(p)]
        goal = targets.get(name, 0)
        left = max(goal - len(files), 0)
        print(f"{name:10s} {len(files):4d} {goal:4d} {left:4d}  {len(problems)}건")
        for msg in problems:
            print(f"{'':27s}- {msg}")
    print("\n* 학습하려면 파일이 있는 클래스마다 최소 2개가 필요하다 (학습/테스트 분할).")


if __name__ == "__main__":
    main()
