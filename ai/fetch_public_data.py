"""공개 데이터셋(ESC-50)에서 knock / baby_cry / unknown 녹음을 받아 ai/data/ 에 넣는다.

실행:  python fetch_public_data.py --dry-run      (받을 목록만 확인)
       python fetch_public_data.py                (다운로드 + 16kHz mono 16bit 변환)
       python fetch_public_data.py --unknown-per-class 15

- 받는 것: knock <- door_wood_knock, baby_cry <- crying_baby, unknown <- 생활소리 여러 종류
- 못 받는 것: fire(연기감지기 경보음), doorbell, 사람 말소리 -> 직접 녹음하거나 다른 출처가 필요하다
- 파일 이름은 esc50_ 로 시작한다. 지우려면 esc50_*.wav 만 삭제하면 된다.
- 어느 클립이 어디서 왔는지는 data/public_sources.csv 에 기록된다 (src_file = Freesound 원본 ID).

라이선스: ESC-50은 CC BY-NC 3.0(비영리)이고 클립별 원 라이선스는 제각각이다.
개발·중간보고용으로만 쓰고, 제품(상업) 학습 데이터로는 직접 녹음이나 라이선스가 확인된 것으로 바꿔야 한다.
"""
import argparse
import csv
import io
import random
import tempfile
import time
import urllib.request
from pathlib import Path

import librosa
import soundfile as sf

from common import DATA_DIR, SR

BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
USER_AGENT = "HearU-data-fetch"
MANIFEST = DATA_DIR / "public_sources.csv"
SEED = 0

# 라벨 -> 가져올 ESC-50 카테고리
KNOCK = "door_wood_knock"
BABY = "crying_baby"
# unknown: 집 안에서 흔한 소리. clock_alarm·siren은 화재경보와 헷갈리기 쉬운 '어려운 오답'이라 일부러 넣었다.
UNKNOWN_CATEGORIES = [
    "keyboard_typing", "mouse_click", "coughing", "laughing", "sneezing", "footsteps",
    "clapping", "door_wood_creaks", "glass_breaking", "vacuum_cleaner", "washing_machine",
    "toilet_flush", "pouring_water", "water_drops", "can_opening", "clock_tick",
    "dog", "cat", "clock_alarm", "siren",
]


def http_get(url: str, retries: int = 3) -> bytes:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(1 + k)


def load_meta() -> list:
    return list(csv.DictReader(io.StringIO(http_get(f"{BASE}/meta/esc50.csv").decode("utf-8"))))


def pick(meta: list, category: str, n: int) -> list:
    """카테고리에서 n개 선택. 원본(src_file)이 다양하도록 서로 다른 원본을 먼저 고른다."""
    rows = [r for r in meta if r["category"] == category]
    random.Random(SEED).shuffle(rows)
    chosen, seen = [], set()
    for r in rows:                       # 1순위: 원본이 겹치지 않게
        if r["src_file"] not in seen:
            chosen.append(r)
            seen.add(r["src_file"])
    chosen += [r for r in rows if r not in chosen]   # 부족하면 같은 원본의 다른 구간
    return chosen[:n]


def plan(meta: list, n_knock: int, n_baby: int, n_unknown: int) -> list:
    items = [("knock", r) for r in pick(meta, KNOCK, n_knock)]
    items += [("baby_cry", r) for r in pick(meta, BABY, n_baby)]
    for cat in UNKNOWN_CATEGORIES:
        items += [("unknown", r) for r in pick(meta, cat, n_unknown)]
    return items


def out_path(label: str, r: dict) -> Path:
    return DATA_DIR / label / f"esc50_{Path(r['filename']).stem}_{r['category']}.wav"


def fetch(label: str, r: dict, dest: Path) -> None:
    raw = http_get(f"{BASE}/audio/{r['filename']}")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / r["filename"]
        src.write_bytes(raw)
        y, _ = librosa.load(src, sr=SR, mono=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, y, SR, subtype="PCM_16")


def write_manifest(items: list) -> None:
    rows = {}
    if MANIFEST.exists():
        rows = {r["file"]: r for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))}
    for label, r in items:
        p = out_path(label, r)
        if p.exists():
            rows[p.name] = {"file": p.name, "label": label, "source": "ESC-50", "esc50_file": r["filename"],
                            "category": r["category"], "src_file": r["src_file"], "fold": r["fold"]}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "source", "esc50_file", "category", "src_file", "fold"])
        w.writeheader()
        w.writerows(sorted(rows.values(), key=lambda r: r["file"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knock", type=int, default=40, help="knock 개수 (최대 40)")
    ap.add_argument("--baby-cry", type=int, default=40, help="baby_cry 개수 (최대 40)")
    ap.add_argument("--unknown-per-class", type=int, default=8, help="unknown 카테고리당 개수 (최대 40)")
    ap.add_argument("--dry-run", action="store_true", help="받지 않고 목록만 출력")
    a = ap.parse_args()

    print("* ESC-50은 비영리(CC BY-NC 3.0) 라이선스입니다. 개발·중간보고용으로만 쓰세요.\n")
    items = plan(load_meta(), a.knock, a.baby_cry, a.unknown_per_class)
    todo = [(l, r) for l, r in items if not out_path(l, r).exists()]

    counts = {}
    for l, _ in items:
        counts[l] = counts.get(l, 0) + 1
    print("선택:", ", ".join(f"{l} {n}개" for l, n in counts.items()), f"/ 이미 있음 {len(items) - len(todo)}개")
    if a.dry_run:
        for l, r in todo:
            print(f"  {l:9s} {r['filename']:22s} {r['category']}")
        return

    failed = []
    for i, (label, r) in enumerate(todo, 1):
        try:
            fetch(label, r, out_path(label, r))
            print(f"[{i}/{len(todo)}] {label:9s} {r['filename']}")
        except Exception as e:
            failed.append(r["filename"])
            print(f"[{i}/{len(todo)}] 실패 {r['filename']}: {e}")
    write_manifest(items)

    print(f"\n완료: {len(todo) - len(failed)}개 저장, 실패 {len(failed)}개. 출처 기록: {MANIFEST}")
    print("확인: python check_data.py")
    print("\n* 이 스크립트로 못 채우는 것: fire(연기감지기 경보음), doorbell, 사람 말소리(대화).")
    print("  fire는 경보음을 스마트폰으로 재생해 다시 녹음하거나 FSD50K 'Smoke detector' 등에서 별도로 구하세요.")


if __name__ == "__main__":
    main()
