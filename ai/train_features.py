"""MFCC + 주파수 변화 + 소리 크기 변화 특징으로 MLP 기준 모델 학습.

실행 예:
  python train_features.py                                  # ai/data -> models/baseline.joblib
  python train_features.py --data data_laptop --name laptop # 노트북 녹음으로 두 번째 모델

평가: 파일 단위 5겹 층화 교차검증(증강은 학습 폴드에만). 최종 모델은 전체 데이터로 다시 학습해 저장.
결과: results/<name>.md, results/<name>.json
"""
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import LABELS, NAME_TO_ID, ROOT, load_wav
from features import FEATURE_NAMES, GROUPS, extract_groups
from train import augment

warnings.filterwarnings("ignore")
SEED = 42
N_AUG = 3
FOLDS = 5
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
ABLATION = {"MFCC만": ["MFCC"], "MFCC+주파수변화": ["MFCC", "주파수변화"],
            "MFCC+소리크기변화": ["MFCC", "소리크기변화"], "전체(3종)": list(GROUPS)}


def new_model():
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=500, random_state=SEED))


def load_files(data_dir):
    paths, labels = [], []
    for name, cid in NAME_TO_ID.items():
        folder = data_dir / name.lower()
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXT) if folder.exists() else []
        paths += files
        labels += [cid] * len(files)
    if not paths:
        raise SystemExit(f"데이터가 없습니다: {data_dir}/<label>/*.wav")
    return np.array(paths, dtype=object), np.array(labels)


def cross_validate(feats, labels, keys):
    vec = lambda g: np.concatenate([g[k] for k in keys])
    pred = np.zeros_like(labels)
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(labels, labels):
        X = np.array([vec(g) for i in tr for g in feats[i]])
        m = new_model().fit(X, np.repeat(labels[tr], N_AUG + 1))
        pred[te] = m.predict(np.array([vec(feats[i][0]) for i in te]))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="ai/ 기준 데이터 폴더 (<label>/*.wav)")
    ap.add_argument("--name", default="baseline", help="모델·결과 파일 이름")
    args = ap.parse_args()
    data_dir = (ROOT / args.data).resolve()

    rng = np.random.default_rng(SEED)
    import train; train.rng = rng                           # 증강 난수 고정
    paths, labels = load_files(data_dir)
    ids = sorted(set(labels.tolist()))
    names = [LABELS[i]["name"] for i in ids]
    counts = {LABELS[i]["name"]: int((labels == i).sum()) for i in ids}
    print("데이터:", counts)

    feats = []                                              # 파일마다 [원본, 증강1..N]
    for p in paths:
        y = load_wav(p)
        feats.append([extract_groups(y)] + [extract_groups(augment(y)) for _ in range(N_AUG)])

    ablation = {}
    for title, keys in ABLATION.items():
        pred = cross_validate(feats, labels, keys)
        _, rec, f1, _ = precision_recall_fscore_support(labels, pred, labels=ids, zero_division=0)
        ablation[title] = {"dims": sum(len(GROUPS[k]) for k in keys), "accuracy": float((pred == labels).mean()),
                           "macro_f1": float(f1.mean()), "recall": dict(zip(names, rec.round(3).tolist()))}
        print(f"{title:14s} acc={ablation[title]['accuracy']:.3f} macroF1={ablation[title]['macro_f1']:.3f}")
        if keys == list(GROUPS):
            main_pred = pred

    p, r, f, s = precision_recall_fscore_support(labels, main_pred, labels=ids, zero_division=0)
    per_class = {n: {"precision": float(a), "recall": float(b), "f1": float(c), "n": int(d)}
                 for n, a, b, c, d in zip(names, p, r, f, s)}
    cm = confusion_matrix(labels, main_pred, labels=ids)
    wrong = [{"file": Path(q).relative_to(data_dir).as_posix(),"true": LABELS[t]["name"], "pred": LABELS[h]["name"]}
             for q, t, h in zip(paths, labels, main_pred) if t != h]

    # 최종 모델: 전체 데이터 + 증강으로 학습
    X_all = np.array([np.concatenate([g[k] for k in GROUPS]) for fl in feats for g in fl])
    model = new_model().fit(X_all, np.repeat(labels, N_AUG + 1))
    (ROOT / "models").mkdir(exist_ok=True)
    model_path = ROOT / "models" / f"{args.name}.joblib"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "feature_module": "features.py",
                 "data": str(data_dir), "counts": counts}, model_path)

    rel = lambda q: Path(q).resolve().relative_to(ROOT).as_posix() if Path(q).resolve().is_relative_to(ROOT) else str(q)
    result = {"name": args.name, "date": datetime.now().isoformat(timespec="seconds"), "data": "ai/" + rel(data_dir),
              "counts": counts, "total": int(len(labels)), "feature_dims": len(FEATURE_NAMES),
              "eval": f"{FOLDS}-fold stratified CV, file-level, aug x{N_AUG} on train folds only",
              "ablation": ablation, "per_class": per_class, "confusion": {"labels": names, "matrix": cm.tolist()},
              "misclassified": wrong, "model_path": "ai/" + rel(model_path)}
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{args.name}.md").write_text(report_md(result, names), encoding="utf-8")
    print("모델:", model_path, "\n결과:", out / f"{args.name}.md")


def report_md(r, names):
    L = [f"# 학습 결과: {r['name']}", "", f"- 날짜: {r['date']}", f"- 데이터: `{r['data']}` (총 {r['total']}개)",
         f"- 평가: {r['eval']}", f"- 특징: {r['feature_dims']}차원 (MFCC 120 + 주파수 변화 10 + 소리 크기 변화 6)", "",
         "## 데이터 수", "", "| " + " | ".join(names) + " |", "|" + "---|" * len(names),
         "| " + " | ".join(str(r["counts"][n]) for n in names) + " |", "",
         "## 특징 조합별 결과 (재현율)", "",
         "| 조합 | 차원 | 정확도 | Macro F1 | " + " | ".join(names) + " |", "|" + "---|" * (4 + len(names))]
    for t, a in r["ablation"].items():
        L.append(f"| {t} | {a['dims']} | {a['accuracy']:.2f} | {a['macro_f1']:.2f} | "
                 + " | ".join(f"{a['recall'][n]:.2f}" for n in names) + " |")
    L += ["", "## 기준 모델(전체 3종) 종류별 성능", "", "| 종류 | 개수 | 정밀도 | 재현율 | F1 |", "|---|---|---|---|---|"]
    for n, v in r["per_class"].items():
        L.append(f"| {n} | {v['n']} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} |")
    L += ["", "## 혼동 행렬 (행=정답, 열=예측)", "", "| 정답＼예측 | " + " | ".join(names) + " |", "|" + "---|" * (1 + len(names))]
    for n, row in zip(names, r["confusion"]["matrix"]):
        L.append(f"| {n} | " + " | ".join(map(str, row)) + " |")
    L += ["", f"## 틀린 파일 ({len(r['misclassified'])}개)", "", "| 파일 | 정답 | 예측 |", "|---|---|---|"]
    L += [f"| {w['file']} | {w['true']} | {w['pred']} |" for w in r["misclassified"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
