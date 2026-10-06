"""공통 설정 + 특징 추출 (학습/추론이 같은 코드를 쓰도록 한 곳에 둔다)."""
import json
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS_PATH = ROOT.parent / "shared" / "labels.json"
DATA_DIR = ROOT / "data"          # data/<LABEL_NAME>/*.wav
MODEL_PATH = ROOT / "model.joblib"

SR = 16000
DURATION = 1.5                    # 초
N_SAMPLES = int(SR * DURATION)
N_MFCC = 40

_cfg = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
LABELS = {l["id"]: l for l in _cfg["labels"]}
NAME_TO_ID = {l["name"]: l["id"] for l in _cfg["labels"]}
CONFIRM = _cfg["thresholds"]["confirm"]
WEAK = _cfg["thresholds"]["weak"]
UNKNOWN_ID = NAME_TO_ID["UNKNOWN"]


def fix_length(y: np.ndarray) -> np.ndarray:
    """N_SAMPLES 길이로 맞춘다. 길면 가장 에너지 큰 구간을 자르고, 짧으면 0 패딩."""
    if len(y) > N_SAMPLES:
        hop = SR // 10
        energy = [np.sum(y[i:i + N_SAMPLES] ** 2) for i in range(0, len(y) - N_SAMPLES + 1, hop)]
        start = int(np.argmax(energy)) * hop
        return y[start:start + N_SAMPLES]
    return np.pad(y, (0, N_SAMPLES - len(y)))


def extract(y: np.ndarray) -> np.ndarray:
    """16kHz mono float 파형 -> MFCC(mean/std) + delta(mean) 특징 벡터 (120차원)."""
    y = fix_length(y.astype(np.float32))
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak              # 마이크 거리/볼륨 차이 완화
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=256)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([mfcc.mean(1), mfcc.std(1), delta.mean(1)])


def load_wav(path) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y
