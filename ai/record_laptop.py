"""노트북 마이크로 비교용 녹음 (ESP32 비교 녹음과 같은 조건·형식).

폰으로 ESP32 녹음 때와 같은 소리·같은 볼륨을 틀고, 같은 거리·조건에서 녹음한다.
  python record_laptop.py fire              # 화재경보음: 조건 3개 x 10회
  python record_laptop.py doorbell
  python record_laptop.py fire --takes 5 --conditions tv
  python record_laptop.py --list-devices    # 마이크 목록, --device 번호로 선택

저장: ai/data_laptop/<label>/<label>_laptop_<날짜_시간>_<조건>_<번호>.wav (16kHz mono 16-bit, 5초)
     + ai/data_laptop/<label>/recording_info_<날짜_시간>.json
학습: python train_features.py --data data data_laptop --exclude esp32 --name laptop --test-on data
"""
import argparse
import json
import sys
import wave
from datetime import datetime

import numpy as np

from common import ROOT

RATE = 16000
SECONDS = 5
CONDITIONS = {"quiet_close": "조용함, 30cm", "quiet_far": "조용함, 2m", "tv": "TV 소리, 1m"}
LABELS = ["fire", "doorbell", "knock", "baby_cry", "call", "unknown"]


def record(sd, device):
    try:
        y = sd.rec(RATE * SECONDS, samplerate=RATE, channels=1, dtype="int16", device=device)
        sd.wait()
        return y[:, 0]
    except Exception:                                   # 16kHz를 직접 못 받는 장치: 기본 주파수로 받고 변환
        import librosa
        sr = int(sd.query_devices(device, "input")["default_samplerate"])
        y = sd.rec(sr * SECONDS, samplerate=sr, channels=1, dtype="float32", device=device)
        sd.wait()
        y = librosa.resample(y[:, 0], orig_sr=sr, target_sr=RATE)[:RATE * SECONDS]
        return np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?", choices=LABELS)
    ap.add_argument("--takes", type=int, default=10, help="조건별 녹음 횟수")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--device", type=int, default=None, help="마이크 번호 (--list-devices로 확인)")
    ap.add_argument("--volume", default="볼륨최대", help="폰 재생 볼륨 기록용")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    import sounddevice as sd
    if args.list_devices:
        print(sd.query_devices())
        return
    if not args.label:
        ap.error("label이 필요합니다 (예: fire)")

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data_laptop" / args.label
    out.mkdir(parents=True, exist_ok=True)
    mic = sd.query_devices(args.device, "input")["name"]
    print(f"마이크: {mic}\n저장 위치: {out}\n각 녹음 전 Enter, 그만두려면 q + Enter", flush=True)

    info = {"label": args.label, "device": "laptop", "microphone": mic, "playback": "phone",
            "playback_volume": args.volume, "sample_rate_hz": RATE, "channels": 1, "bits": 16,
            "seconds": SECONDS, "conditions": {c: CONDITIONS[c] for c in args.conditions}, "recordings": []}
    info_path = out / f"recording_info_{session}.json"
    try:
        for cond in args.conditions:
            print(f"\n=== 조건: {CONDITIONS[cond]} — 폰과 노트북 거리·환경을 맞춰 주세요 ===", flush=True)
            for take in range(1, args.takes + 1):
                if input(f"[{args.label} {cond} {take}/{args.takes}] Enter > ").strip().lower() == "q":
                    raise KeyboardInterrupt
                print("녹음 중... 지금 소리를 틀어 주세요 (5초)", flush=True)
                y = record(sd, args.device)
                path = out / f"{args.label}_laptop_{session}_{cond}_{take:02d}.wav"
                with wave.open(str(path), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(y.tobytes())
                peak = int(np.abs(y.astype(np.int32)).max())
                clipped = int((np.abs(y.astype(np.int32)) >= 32767).sum())
                info["recordings"].append({"filename": path.name, "condition": cond,
                                           "created_at": datetime.now().isoformat(timespec="seconds"),
                                           "peak_pcm16": peak, "clipped_samples_at_16bit_limit": clipped})
                warn = "  ⚠ 소리가 너무 작음" if peak < 300 else ("  ⚠ 잘림(클리핑) 있음" if clipped else "")
                print(f"저장: {path.name}  최대값 {peak}{warn}", flush=True)
    except KeyboardInterrupt:
        print("\n중단했습니다.")
    finally:
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n녹음 {len(info['recordings'])}개, 기록: {info_path.name}")


if __name__ == "__main__":
    sys.exit(main())
