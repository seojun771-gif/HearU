"""여러 학습 결과(results/<이름>.json)를 비교표로 만든다 -> docs/MODEL_COMPARISON.md

  python compare_models.py esp32_1122 phone esp32
결과 파일이 없는 모델은 '미학습'으로 표시한다.
"""
import json
import sys

from common import ROOT

TITLES = {"esp32_1122": "모델 1: ESP32 마이크", "phone": "모델 2: 폰 마이크", "laptop": "노트북 마이크",
          "esp32": "참고: ESP32 전체 데이터", "baseline": "참고: 이전 기준 모델"}
LABEL_KO = {"UNKNOWN": "기타", "FIRE": "화재", "BABY_CRY": "아기 울음", "DOORBELL": "초인종", "KNOCK": "노크"}
COND_KO = {"quiet_close": "조용함 30cm", "quiet_far": "조용함 2m", "tv": "TV 소리 1m"}


def load(name):
    p = ROOT / "results" / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def f2(x):
    return "—" if x is None else f"{x:.2f}"


def get(r, *keys):
    for k in keys:
        if r is None:
            return None
        r = r.get(k) if isinstance(r, dict) else None
    return r


def main(names):
    rs = {n: load(n) for n in names}
    head = "| 항목 | " + " | ".join(TITLES.get(n, n) for n in names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    row = lambda t, f: f"| {t} | " + " | ".join("미학습" if rs[n] is None else f(rs[n]) for n in names) + " |"

    L = ["# 모델 비교: ESP32 마이크 vs 폰 마이크", "",
         "같은 화재경보음·초인종 소리를 녹음한 마이크만 다르게 해서 학습한 두 모델을 비교한다.",
         "두 모델 모두 공통 데이터(ESC-50 공개 데이터 + 이전 직접 녹음)에 각 마이크의 화재·초인종 녹음을 더해 학습했다.",
         "- ESP32: 5초 WAV 60개, 조건별 10개씩 (조용함 30cm / 조용함 2m / TV 소리 1m)",
         "- 폰: 아이폰 음성 메모(m4a, 0.8~9.6초) 58개, 거리·환경 조건 구분 없음 → 16kHz WAV로 변환 (`import_phone.py`)",
         "- 녹음 묶음(교차검증 B): ESP32는 녹음 조건별, 폰은 녹음 순서대로 3묶음",
         "'참고' 열은 11/17 ESP32 녹음까지 모두 넣은 모델로, 공정한 비교 대상은 아니다.",
         "특징(136차원)·모델(MLP)·평가 방법은 같다. 계산 방법은 [FEATURES.md](FEATURES.md).", "",
         "## 요약", "", head, sep,
         row("정확도 (B)", lambda r: f2(get(r, "eval_session", "accuracy"))),
         row("Macro F1 (B)", lambda r: f2(get(r, "eval_session", "macro_f1"))),
         row("화재 재현율 (B)", lambda r: f2(get(r, "eval_session", "per_class", "FIRE", "recall"))),
         row("초인종 재현율 (B)", lambda r: f2(get(r, "eval_session", "per_class", "DOORBELL", "recall"))),
         row("다른 마이크 녹음으로 시험 정확도 (C)", lambda r: "안 함" if not r.get("cross_test") else
             f"{r['cross_test']['accuracy']:.2f}"), "",
         "B = 녹음 묶음 단위 교차검증(주 지표), C = 학습에 안 쓴 다른 마이크의 녹음으로 시험.", "",
         "**결론**: 같은 마이크 안에서는 ESP32 모델이 더 좋고(정확도 0.82 vs 0.73), 마이크가 바뀌면 두 모델 모두 크게 떨어진다.",
         "실제 장치에서 쓸 모델은 장치 마이크(ESP32)로 녹음한 데이터로 학습해야 한다. 해석은 [FEATURES.md](FEATURES.md) 4절.", "",
         "## 1. 데이터", "", head, sep,
         row("학습 데이터", lambda r: ", ".join(r["data"]) + (f" (제외: {', '.join(r['exclude'])})" if r["exclude"] else "")),
         row("총 개수", lambda r: str(r["total"])),
         row("녹음 묶음 수", lambda r: str(r["sessions"])),
         row("직접 녹음 장치", lambda r: ", ".join(f"{d} {sum(c.values())}" for d, c in r["devices"].items() if d != "esc50"))]
    for lab, ko in LABEL_KO.items():
        L.append(row(f"{ko} 개수", lambda r, lab=lab: str(r["counts"].get(lab, 0))))

    L += ["", "## 2. 전체 성능", "", head, sep,
          row("A. 파일 단위 정확도", lambda r: f2(get(r, "eval_file", "accuracy"))),
          row("A. 파일 단위 Macro F1", lambda r: f2(get(r, "eval_file", "macro_f1"))),
          row("**B. 녹음 묶음 단위 정확도**", lambda r: f2(get(r, "eval_session", "accuracy"))),
          row("**B. 녹음 묶음 단위 Macro F1**", lambda r: f2(get(r, "eval_session", "macro_f1"))),
          row("C. 다른 장치 녹음으로 시험 정확도", lambda r: "안 함" if not r.get("cross_test") else
              f"{r['cross_test']['accuracy']:.2f} ({', '.join(r['cross_test']['devices'])} {r['cross_test']['n']}개)")]

    L += ["", "## 3. 종류별 재현율 (B. 녹음 묶음 단위)", "", head, sep]
    for lab, ko in LABEL_KO.items():
        L.append(row(ko, lambda r, lab=lab: f2(get(r, "eval_session", "per_class", lab, "recall"))))
    L += ["", "## 4. 녹음 조건별 재현율 (B. 녹음 묶음 단위, ESP32만 조건 기록 있음)", "", head, sep]
    for c, cko in COND_KO.items():
        for lab in ("FIRE", "DOORBELL"):
            L.append(row(f"{cko} · {LABEL_KO[lab]}",
                         lambda r, c=c, lab=lab: f2(get(r, "eval_session", "by_condition", c, lab, "recall"))))

    L += ["", "## 5. 다른 장치 녹음으로 시험 (C, 학습에 안 쓴 마이크의 녹음)", "", head, sep]
    for lab, ko in LABEL_KO.items():
        L.append(row(ko, lambda r, lab=lab: f2(get(r, "cross_test", "per_class", lab, "recall"))))

    L += ["", "## 읽는 법", "",
          "- **B가 주 지표.** A는 같은 녹음 묶음의 비슷한 파일이 학습·시험에 섞여 높게 나올 수 있다.",
          "- C는 '한 마이크로 학습한 모델이 다른 마이크에서도 통하나'를 본다. A·B와 차이가 크면 마이크 차이에 민감한 것.",
          "- 화재·초인종 원본 소리가 한 종류씩이라, B도 '처음 듣는 다른 화재경보기'에서의 성능보다는 높게 나온다.",
          "- 목표: 화재·아기 울음 재현율 ≥0.95, Macro F1 ≥0.85.", "",
          "## 다시 만들기", "", "```",
          "cd ai",
          "python import_phone.py ../../2model/firebell fire --date 20261122",
          "python import_phone.py ../../2model/dobelll doorbell --date 20261122",
          "python train_features.py --exclude esp32_20261117 --name esp32_1122 --test-on data_phone",
          "python train_features.py --data data data_phone --exclude esp32 --name phone --test-on data --test-match esp32_20261122",
          "python train_features.py --name esp32",
          "python compare_models.py esp32_1122 phone esp32", "```", ""]
    out = ROOT.parent / "docs" / "MODEL_COMPARISON.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("저장:", out.relative_to(ROOT.parent).as_posix())


if __name__ == "__main__":
    main(sys.argv[1:] or ["esp32_1122", "phone", "esp32"])
