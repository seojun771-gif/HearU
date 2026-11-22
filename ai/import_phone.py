"""폰 음성 메모(m4a 등) 녹음을 학습용 WAV(16kHz mono 16-bit)로 변환해 ai/data_phone/<label>/에 넣는다.

  python import_phone.py ../../2model/firebell fire --date 20261122
  python import_phone.py ../../2model/dobelll doorbell --date 20261122

- 파일 이름의 번호 순서로 정렬 ('자양동.m4a'처럼 번호가 없으면 1번), 내용이 똑같은 파일은 하나만 쓴다.
- 거리 조건은 구분하지 않는다. 대신 녹음 순서대로 --parts개 묶음(p1, p2, ...)으로 나눠 파일 이름에 남긴다.
  연달아 녹음한 파일끼리 비슷하므로, 교차검증 B에서 같은 묶음은 학습·시험에 함께 들어가지 않는다.
- 저장 이름: <label>_phone_<날짜>_p<묶음>_<번호>.wav, 원본 이름 대응표는 import_info_<날짜>.json
필요: pip install av
"""
import argparse
import hashlib
import json
import re
import wave
from pathlib import Path

import av
import numpy as np

from common import ROOT

RATE = 16000
EXT = {".m4a", ".mp3", ".aac", ".wav", ".caf", ".mp4", ".ogg", ".flac"}


def decode(path):
    """아무 오디오 파일 -> 16kHz mono int16."""
    out = []
    with av.open(str(path)) as f:
        rs = av.AudioResampler(format="s16", layout="mono", rate=RATE)
        for frame in f.decode(audio=0):
            for r in rs.resample(frame):
                out.append(r.to_ndarray().reshape(-1))
        for r in rs.resample(None):
            out.append(r.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.int16)


def number_of(p):
    m = re.search(r"(\d+)(?:\s*\(\d+\))?$", p.stem)
    return int(m[1]) if m else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="원본 녹음 폴더")
    ap.add_argument("label", choices=["fire", "doorbell", "knock", "baby_cry", "call", "unknown"])
    ap.add_argument("--date", required=True, help="녹음 날짜 YYYYMMDD (파일 이름용)")
    ap.add_argument("--parts", type=int, default=3, help="녹음 순서 묶음 수")
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted((p for p in src.iterdir() if p.suffix.lower() in EXT), key=lambda p: (number_of(p), len(p.name), p.name))
    seen, uniq, skipped = {}, [], []
    for p in files:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            skipped.append({"file": p.name, "same_as": seen[h]})
        else:
            seen[h] = p.name
            uniq.append(p)

    out = ROOT / "data_phone" / args.label
    out.mkdir(parents=True, exist_ok=True)
    per = -(-len(uniq) // args.parts)                                # 올림 나눗셈
    rows = []
    for i, p in enumerate(uniq):
        y = decode(p)
        name = f"{args.label}_phone_{args.date}_p{i // per + 1}_{i + 1:02d}.wav"
        with wave.open(str(out / name), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(y.tobytes())
        peak = int(np.abs(y.astype(np.int32)).max())
        rows.append({"wav": name, "source": p.name, "seconds": round(len(y) / RATE, 2), "peak_pcm16": peak})
        print(f"{p.name:20s} -> {name}  {len(y) / RATE:5.2f}s  최대값 {peak}")
    info = {"label": args.label, "device": "phone", "date": args.date, "source_dir": src.name,
            "sample_rate_hz": RATE, "parts": args.parts, "files": rows, "skipped_duplicates": skipped,
            "note": "거리·환경 조건 구분 없음. 묶음(p)은 녹음 순서 기준."}
    (out / f"import_info_{args.date}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n변환 {len(rows)}개, 중복 제외 {len(skipped)}개 {[s['file'] for s in skipped]}")


if __name__ == "__main__":
    main()
