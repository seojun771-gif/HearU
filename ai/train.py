"""MFCC + MLP 기준 모델 학습.

데이터 구조:  ai/data/FIRE/*.wav, BABY_CRY/, DOORBELL/, KNOCK/, CALL/, UNKNOWN/
실행:        python train.py
"""
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA_DIR, LABELS, MODEL_PATH, NAME_TO_ID, SR, extract, load_wav

SEED = 42
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
rng = np.random.default_rng(SEED)


def augment(y: np.ndarray) -> np.ndarray:
    """학습 데이터 부족 대비: 볼륨, 시간 이동, 배경 잡음."""
    y = y * rng.uniform(0.5, 1.5)
    y = np.roll(y, int(rng.uniform(-0.2, 0.2) * SR))
    noise_level = rng.uniform(0.0, 0.02) * (np.max(np.abs(y)) + 1e-9)
    return y + rng.normal(0, noise_level, size=y.shape)


def load_files():
    paths, labels = [], []
    for name, cid in NAME_TO_ID.items():
        folder = DATA_DIR / name
        files = [p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXT] if folder.exists() else []
        print(f"{name:9s} {len(files)}개")
        paths += files
        labels += [cid] * len(files)
    if not paths:
        raise SystemExit(f"데이터가 없습니다: {DATA_DIR}/<LABEL>/*.wav")
    return paths, np.array(labels)


def build(paths, labels, n_aug):
    X, y = [], []
    for p, l in zip(paths, labels):
        wav = load_wav(p)
        X.append(extract(wav)); y.append(l)
        for _ in range(n_aug):
            X.append(extract(augment(wav))); y.append(l)
    return np.array(X), np.array(y)


def main():
    paths, labels = load_files()
    # 증강 누수를 막기 위해 '파일 단위'로 먼저 split 한 뒤 train에만 증강 적용
    p_tr, p_te, y_tr, y_te = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=SEED)
    X_tr, y_tr = build(p_tr, y_tr, n_aug=3)
    X_te, y_te = build(p_te, y_te, n_aug=0)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                      max_iter=500, random_state=SEED),
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    ids = sorted(LABELS)
    names = [LABELS[i]["name"] for i in ids]
    print(classification_report(y_te, pred, labels=ids, target_names=names, zero_division=0))
    print("confusion matrix (행=정답, 열=예측)\n", confusion_matrix(y_te, pred, labels=ids))

    joblib.dump(model, MODEL_PATH)
    print("저장:", MODEL_PATH)


if __name__ == "__main__":
    main()
