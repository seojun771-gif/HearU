"""추론 + 신뢰도 판정 + UART 메시지 생성.

파일 분류:   python infer.py sample.wav
마이크 실시간: python infer.py --mic
"""
import sys
import time

import joblib
import numpy as np

from common import (CONFIRM, LABELS, MODEL_PATH, N_SAMPLES, SR, UNKNOWN_ID, WEAK, extract,
                    load_wav)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def classify(y: np.ndarray) -> dict:
    proba = _get_model().predict_proba(extract(y)[None])[0]
    classes = _get_model().classes_
    cid = int(classes[int(np.argmax(proba))])
    conf = float(np.max(proba))

    if conf >= CONFIRM:
        level = "CONFIRMED"
    elif conf >= WEAK:
        level = "WEAK"
    else:
        level, cid = "UNKNOWN", UNKNOWN_ID   # 신뢰도 낮으면 기타 처리

    label = LABELS[cid]
    return {"class_id": cid, "label": label["name"], "ko": label["ko"],
            "confidence": conf, "level": level, "priority": label["priority"]}


def to_uart(r: dict) -> str:
    """RESULT,class_id,label,confidence(0~100),priority\\n"""
    return f"RESULT,{r['class_id']},{r['label']},{round(r['confidence'] * 100)},{r['priority']}\n"


def run_mic(rms_gate=0.01):
    """1.5초 창을 0.5초 간격으로 슬라이딩. 같은 클래스는 쿨다운 동안 재알림 안 함."""
    import sounddevice as sd
    from common import _cfg
    cooldown = _cfg["cooldown_sec"]
    buf = np.zeros(N_SAMPLES, dtype=np.float32)
    last = {}
    hop = SR // 2
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32") as stream:
        print("듣는 중... (Ctrl+C 종료)")
        while True:
            chunk, _ = stream.read(hop)
            buf = np.concatenate([buf[hop:], chunk[:, 0]])
            if np.sqrt(np.mean(buf ** 2)) < rms_gate:   # 무음 구간은 건너뜀
                continue
            t0 = time.perf_counter()
            r = classify(buf)
            ms = (time.perf_counter() - t0) * 1000
            if r["level"] == "UNKNOWN" or time.time() - last.get(r["label"], 0) < cooldown:
                continue
            last[r["label"]] = time.time()
            print(f"[{r['level']}] {r['ko']} {r['confidence']:.2f} ({ms:.0f}ms) -> {to_uart(r).strip()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mic":
        run_mic()
    elif len(sys.argv) > 1:
        r = classify(load_wav(sys.argv[1]))
        print(r)
        print(to_uart(r), end="")
    else:
        print(__doc__)
