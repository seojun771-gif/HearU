"""MFCC + 주파수 변화 + 소리 크기 변화 특징으로 MLP 모델 학습·평가.

실행 예:
  python train_features.py --name esp32
      ai/data 전체 -> models/esp32.joblib, results/esp32.md·json
  python train_features.py --exclude esp32_20261117 --name esp32_1122 --test-on data_phone
      비교용 모델 1: 공통 데이터 + 11/22 ESP32 녹음만, 폰 녹음으로 교차 시험
  python train_features.py --data data data_phone --exclude esp32 --name phone --test-on data --test-match esp32_20261122
      비교용 모델 2: 공통 데이터 + 폰 녹음(import_phone.py로 변환), 11/22 ESP32 녹음으로 교차 시험

평가 (둘 다 파일 단위, 증강은 학습 폴드에만):
  A. 파일 단위 5겹 층화 교차검증 - 이전 결과와 비교용
  B. 녹음 묶음 단위 5겹 교차검증 - 같은 날·같은 조건으로 녹음한 묶음(또는 ESC-50 같은 원본 클립)이
     학습과 시험에 함께 들어가지 않게 나눔. 처음 보는 녹음 환경에서의 성능에 더 가깝다.
최종 모델은 전체 데이터로 다시 학습해 저장한다.
"""
import argparse
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train
from common import LABELS, NAME_TO_ID, ROOT, load_wav
from features import FEATURE_NAMES, GROUPS, extract_groups

warnings.filterwarnings("ignore")
SEED = 42
N_AUG = 3
FOLDS = 5
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
CONDITIONS = {"quiet_close": "조용함 30cm", "quiet_far": "조용함 2m", "tv": "TV 소리 1m"}
ABLATION = {"MFCC만": ["MFCC"], "MFCC+주파수변화": ["MFCC", "주파수변화"],
            "MFCC+소리크기변화": ["MFCC", "소리크기변화"], "전체(3종)": list(GROUPS)}


def new_model():
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=500, random_state=SEED))


def rel(p):
    """ai/ 기준 상대 경로 (결과 파일에 개인 PC 경로가 남지 않게)."""
    p = Path(p).resolve()
    return "ai/" + p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name


def device_of(name):
    for d in ("esp32", "laptop", "phone", "esc50"):
        if d in name:
            return d
    return "기타"


def condition_of(name):
    return next((c for c in CONDITIONS if f"_{c}_" in name), None)


def session_of(path):
    """녹음 묶음 id. 같은 묶음은 교차검증 B에서 한쪽(학습 또는 시험)에만 들어간다."""
    n = Path(path).stem
    if n.startswith("esc50_"):
        return "esc50_" + n.split("_")[1].split("-")[1]           # 같은 원본 클립(A/B/C 조각)
    m = re.search(r"((?:esp32|laptop)_\d{8}_\d{6})", n)
    if m:
        return m[1] + "_" + (condition_of(n) or "")
    m = re.search(r"(phone_\d{8}_p\d+)", n)                     # 폰 녹음: 녹음 순서 묶음 (import_phone.py)
    if m:
        return n.split("_")[0] + "_" + m[1]
    m = re.search(r"_(\d{8})_\d+$", n)
    if m:
        return n.split("_")[0] + "_" + m[1]
    return n.split("_")[0] + "_기타"                             # 예: doorbell_001 (출처 하나로 봄)


def load_files(data_dirs, exclude):
    paths, labels = [], []
    for d in data_dirs:
        for name, cid in NAME_TO_ID.items():
            folder = d / name.lower()
            files = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_EXT
                           and not any(x in p.name for x in exclude)) if folder.exists() else []
            paths += files
            labels += [cid] * len(files)
    if not paths:
        raise SystemExit(f"데이터가 없습니다: {[str(d) for d in data_dirs]}/<label>/*.wav")
    return np.array(paths, dtype=object), np.array(labels)


def features_for(paths, n_aug):
    out = []                                                       # 파일마다 [원본, 증강1..N]
    for p in paths:
        y = load_wav(p)
        out.append([extract_groups(y)] + [extract_groups(train.augment(y)) for _ in range(n_aug)])
    return out


def vec(g, keys):
    return np.concatenate([g[k] for k in keys])


def cross_validate(feats, labels, keys, groups=None):
    if groups is None:
        splits = StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(labels, labels)
    else:
        splits = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=0).split(labels, labels, groups)
    pred = np.zeros_like(labels)
    for tr, te in splits:
        X = np.array([vec(g, keys) for i in tr for g in feats[i]])
        m = new_model().fit(X, np.repeat(labels[tr], N_AUG + 1))
        pred[te] = m.predict(np.array([vec(feats[i][0], keys) for i in te]))
    return pred


def summarize(paths, labels, pred, ids, base_dirs):
    names = [LABELS[i]["name"] for i in ids]
    p, r, f, s = precision_recall_fscore_support(labels, pred, labels=ids, zero_division=0)
    per_class = {n: {"precision": float(a), "recall": float(b), "f1": float(c), "n": int(d)}
                 for n, a, b, c, d in zip(names, p, r, f, s)}
    present = s > 0
    by_cond = {}
    for c in CONDITIONS:
        for i in ids:
            mask = np.array([condition_of(Path(q).name) == c for q in paths]) & (labels == i)
            if mask.any():
                by_cond.setdefault(c, {})[LABELS[i]["name"]] = {
                    "recall": float((pred[mask] == i).mean()), "n": int(mask.sum())}
    def short(q):
        for d in base_dirs:
            if Path(q).is_relative_to(d):
                return Path(q).relative_to(d).as_posix()
        return Path(q).name
    wrong = [{"file": short(q), "true": LABELS[t]["name"], "pred": LABELS[h]["name"]}
             for q, t, h in zip(paths, labels, pred) if t != h]
    return {"accuracy": float((pred == labels).mean()), "macro_f1": float(f[present].mean()),
            "per_class": per_class, "by_condition": by_cond,
            "confusion": {"labels": names, "matrix": confusion_matrix(labels, pred, labels=ids).tolist()},
            "misclassified": wrong}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data"], help="ai/ 기준 데이터 폴더들 (<label>/*.wav)")
    ap.add_argument("--exclude", nargs="*", default=[], help="파일 이름에 이 글자가 있으면 학습에서 뺌 (예: esp32)")
    ap.add_argument("--test-on", nargs="*", default=[], help="최종 모델을 시험할 다른 폴더 (학습에 안 쓴 다른 장치 녹음만)")
    ap.add_argument("--test-match", default=None, help="--test-on에서 파일 이름에 이 글자가 있는 것만 시험 (예: esp32_20261122). 기본: 학습에 없는 장치의 녹음 전부")
    ap.add_argument("--name", default="baseline", help="모델·결과 파일 이름")
    args = ap.parse_args()
    data_dirs = [(ROOT / d).resolve() for d in args.data]

    train.rng = np.random.default_rng(SEED)                        # 증강 난수 고정
    paths, labels = load_files(data_dirs, args.exclude)
    ids = sorted(set(labels.tolist()))
    names = [LABELS[i]["name"] for i in ids]
    counts = {n: int((labels == i).sum()) for n, i in zip(names, ids)}
    devices = {}
    for q, l in zip(paths, labels):
        k = device_of(Path(q).name)
        devices.setdefault(k, {}).setdefault(LABELS[l]["name"], 0)
        devices[k][LABELS[l]["name"]] += 1
    groups = np.array([session_of(q) for q in paths])
    print("데이터:", counts, "\n장치별:", devices, "\n녹음 묶음 수:", len(set(groups)))

    feats = features_for(paths, N_AUG)

    ablation = {}
    for title, keys in ABLATION.items():
        pred = cross_validate(feats, labels, keys)
        _, rec, f1, sup = precision_recall_fscore_support(labels, pred, labels=ids, zero_division=0)
        ablation[title] = {"dims": sum(len(GROUPS[k]) for k in keys), "accuracy": float((pred == labels).mean()),
                           "macro_f1": float(f1[sup > 0].mean()), "recall": dict(zip(names, rec.round(3).tolist()))}
        print(f"[A] {title:14s} acc={ablation[title]['accuracy']:.3f} macroF1={ablation[title]['macro_f1']:.3f}")
        if keys == list(GROUPS):
            pred_a = pred
    pred_b = cross_validate(feats, labels, list(GROUPS), groups)
    eval_a = summarize(paths, labels, pred_a, ids, data_dirs)
    eval_b = summarize(paths, labels, pred_b, ids, data_dirs)
    print(f"[B] 녹음 묶음 단위      acc={eval_b['accuracy']:.3f} macroF1={eval_b['macro_f1']:.3f}")

    X_all = np.array([vec(g, GROUPS) for fl in feats for g in fl])
    model = new_model().fit(X_all, np.repeat(labels, N_AUG + 1))
    (ROOT / "models").mkdir(exist_ok=True)
    model_path = ROOT / "models" / f"{args.name}.joblib"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "feature_module": "features.py",
                 "data": [rel(d) for d in data_dirs], "exclude": args.exclude, "counts": counts}, model_path)

    cross = None
    if args.test_on:
        test_dirs = [(ROOT / d).resolve() for d in args.test_on]
        train_devs = set(devices) - {"esc50", "기타"}
        tp, tl = load_files(test_dirs, [])
        want = (lambda n: args.test_match in n) if args.test_match else \
               (lambda n: device_of(n) not in train_devs | {"esc50", "기타"})
        keep = np.array([want(Path(q).name) for q in tp])
        tp, tl = tp[keep], tl[keep]
        if len(tp):
            tf = features_for(tp, 0)
            tpred = model.predict(np.array([vec(g[0], GROUPS) for g in tf]))
            cross = summarize(tp, tl, tpred, ids, test_dirs)           # 혼동 행렬은 모델의 모든 종류로
            cross["per_class"] = {k: v for k, v in cross["per_class"].items() if v["n"] > 0}
            cross.update({"data": [rel(d) for d in test_dirs], "n": int(len(tl)),
                          "devices": sorted({device_of(Path(q).name) for q in tp})})
            print(f"[교차 시험] {cross['devices']} {len(tl)}개 acc={cross['accuracy']:.3f}")
        else:
            print("[교차 시험] 시험할 파일이 없습니다.")

    result = {"name": args.name, "date": datetime.now().isoformat(timespec="seconds"),
              "data": [rel(d) for d in data_dirs], "exclude": args.exclude, "counts": counts,
              "devices": devices, "total": int(len(labels)), "sessions": int(len(set(groups))),
              "feature_dims": len(FEATURE_NAMES), "n_aug": N_AUG, "folds": FOLDS,
              "ablation": ablation, "eval_file": eval_a, "eval_session": eval_b, "cross_test": cross,
              "model_path": rel(model_path)}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"{args.name}.md").write_text(report_md(result, names), encoding="utf-8")
    print("모델:", rel(model_path), "\n결과:", rel(out / f"{args.name}.md"))


def _eval_md(title, e, names):
    L = [f"### {title}", "", f"정확도 {e['accuracy']:.2f} · Macro F1 {e['macro_f1']:.2f}", "",
         "| 종류 | 개수 | 정밀도 | 재현율 | F1 |", "|---|---|---|---|---|"]
    for n, v in e["per_class"].items():
        L.append(f"| {n} | {v['n']} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} |")
    if e["by_condition"]:
        L += ["", "녹음 조건별 재현율", "", "| 조건 | 종류 | 개수 | 재현율 |", "|---|---|---|---|"]
        for c, d in e["by_condition"].items():
            for n, v in d.items():
                L.append(f"| {CONDITIONS[c]} | {n} | {v['n']} | {v['recall']:.2f} |")
    cn = e["confusion"]["labels"]
    L += ["", "혼동 행렬 (행=정답, 열=예측)", "", "| 정답＼예측 | " + " | ".join(cn) + " |", "|" + "---|" * (1 + len(cn))]
    L += [f"| {n} | " + " | ".join(map(str, row)) + " |" for n, row in zip(cn, e["confusion"]["matrix"])]
    L += ["", f"<details><summary>틀린 파일 {len(e['misclassified'])}개</summary>", "",
          "| 파일 | 정답 | 예측 |", "|---|---|---|"]
    L += [f"| {w['file']} | {w['true']} | {w['pred']} |" for w in e["misclassified"]]
    return L + ["", "</details>", ""]


def report_md(r, names):
    L = [f"# 학습 결과: {r['name']}", "", f"- 날짜: {r['date']}",
         f"- 데이터: {', '.join('`' + d + '`' for d in r['data'])} (총 {r['total']}개, 녹음 묶음 {r['sessions']}개)"
         + (f", 제외: {r['exclude']}" if r["exclude"] else ""),
         f"- 특징: {r['feature_dims']}차원 (MFCC 120 + 주파수 변화 10 + 소리 크기 변화 6), 학습 파일당 증강 {r['n_aug']}개",
         f"- 모델 파일: `{r['model_path']}` (git 제외)", "",
         "## 데이터 수", "", "| 장치 | " + " | ".join(names) + " |", "|" + "---|" * (1 + len(names))]
    for d, c in r["devices"].items():
        L.append(f"| {d} | " + " | ".join(str(c.get(n, 0)) for n in names) + " |")
    L.append("| **합계** | " + " | ".join(f"**{r['counts'][n]}**" for n in names) + " |")
    L += ["", "## 특징 조합별 결과 (A. 파일 단위 교차검증, 숫자는 재현율)", "",
          "| 조합 | 차원 | 정확도 | Macro F1 | " + " | ".join(names) + " |", "|" + "---|" * (4 + len(names))]
    for t, a in r["ablation"].items():
        L.append(f"| {t} | {a['dims']} | {a['accuracy']:.2f} | {a['macro_f1']:.2f} | "
                 + " | ".join(f"{a['recall'][n]:.2f}" for n in names) + " |")
    L += ["", "## 3종 전체 모델 평가", ""]
    L += _eval_md(f"A. 파일 단위 {r['folds']}겹 교차검증 (이전 결과와 비교용)", r["eval_file"], names)
    L += _eval_md(f"B. 녹음 묶음 단위 {r['folds']}겹 교차검증 (같은 묶음은 학습·시험에 함께 안 들어감)", r["eval_session"], names)
    if r["cross_test"]:
        c = r["cross_test"]
        L += _eval_md(f"C. 다른 장치 녹음으로 시험 ({', '.join(c['devices'])} {c['n']}개, 학습에 안 씀)", c, names)
    return "\n".join(L)


if __name__ == "__main__":
    main()
