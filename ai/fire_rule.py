"""화재경보 규칙 검출기 (AI 모델과 독립적으로 동작하는 안전망).

화재경보음은 3kHz 부근의 고음이 정해진 리듬으로 반복된다.
- T-3 (화재): 0.5초 울림 / 0.5초 쉼 × 3회 → 1.5초 쉼 → 반복
- T-4 (일산화탄소): 0.1초 울림 / 0.1초 쉼 × 4회 → 5초 쉼 → 반복
모델 확률 대신 "대역 에너지 + 반복 주기" 규칙으로 판단한다.

판정:
  CONFIRMED  T-3 3회 또는 T-4 4회가 규칙적으로 확인됨
  CANDIDATE  T-3 2회 / T-4 2회 / 대역 고음이 1초 이상 지속 → 빠른 예비 알림용

파일 검사:    python fire_rule.py alarm.wav
마이크 실시간: python fire_rule.py --mic
"""
import sys
import time

import numpy as np

from common import SR, load_wav

N_FFT = 512                  # 32ms 창
HOP = SR // 100              # 10ms 간격
FRAME_SEC = HOP / SR

BAND = (2700, 3800)          # 일반 연기감지기 경보음 대역 (약 3.0~3.5kHz)
BAND_RATIO = 0.30            # 프레임 에너지 중 대역이 차지해야 하는 최소 비율
RMS_GATE = 0.003             # 이보다 작은 프레임은 무음 (약 -50 dBFS)
MERGE_GAP = 0.03             # 이보다 짧은 끊김은 한 울림으로 합침 (초)
MIN_PULSE = 0.04             # 이보다 짧은 울림은 버림 (초)

# (울림 길이 범위, 울림 사이 쉼 범위, 확정에 필요한 횟수)  단위: 초
PATTERNS = {
    "T3": ((0.30, 0.80), (0.30, 0.80), 3),
    "T4": ((0.05, 0.20), (0.05, 0.25), 4),
}
CONTINUOUS_SEC = 1.0


def band_frames(y: np.ndarray) -> np.ndarray:
    """프레임(10ms)마다 '경보 대역 고음이 울리는 중인가'를 bool 배열로 반환."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < N_FFT:
        return np.zeros(0, dtype=bool)
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * np.hanning(N_FFT), axis=1)) ** 2
    freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
    in_band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    ratio = power[:, in_band].sum(1) / (power.sum(1) + 1e-12)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return (ratio >= BAND_RATIO) & (rms >= RMS_GATE)


def pulses(on: np.ndarray) -> list:
    """bool 프레임 → [(시작초, 끝초, 버퍼 끝에서 아직 울리는 중인가)]."""
    out = []
    i, n = 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        out.append([i, j])
        i = j
    merged = []
    max_gap = round(MERGE_GAP / FRAME_SEC)
    for p in out:
        if merged and p[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = p[1]
        else:
            merged.append(p)
    return [(s * FRAME_SEC, e * FRAME_SEC, e == n) for s, e in merged
            if (e - s) * FRAME_SEC >= MIN_PULSE]


def _longest_run(ps, on_rng, gap_rng) -> int:
    """규칙적인 울림이 연속으로 몇 번 이어지는지 (가장 긴 구간)."""
    best = run = 0
    for k, (s, e, is_open) in enumerate(ps):
        d = e - s
        # 버퍼 끝에서 아직 울리는 중이면 길이가 덜 찼으므로 최소 길이만 확인
        ok = d >= on_rng[0] and (is_open or d <= on_rng[1])
        if not ok:
            run = 0
            continue
        gap = s - ps[k - 1][1] if k > 0 else None
        run = run + 1 if run and gap_rng[0] <= gap <= gap_rng[1] else 1
        best = max(best, run)
    return best


def analyze(y: np.ndarray) -> dict:
    """파형(16kHz mono float) 전체를 보고 화재경보 여부를 판정."""
    ps = pulses(band_frames(y))
    result = {"level": None, "pattern": None, "pulses": [(round(s, 2), round(e, 2)) for s, e, _ in ps]}
    for name, (on_rng, gap_rng, need) in PATTERNS.items():
        run = _longest_run(ps, on_rng, gap_rng)
        if run >= need:
            result.update(level="CONFIRMED", pattern=name)
            return result
        if run >= 2 and result["level"] is None:
            result.update(level="CANDIDATE", pattern=name)
    if result["level"] is None and any(e - s >= CONTINUOUS_SEC for s, e, _ in ps):
        result.update(level="CANDIDATE", pattern="CONTINUOUS")
    return result


class FireRuleDetector:
    """실시간용. 짧은 조각을 계속 넣으면 최근 window_sec 초를 보고 판정한다."""

    def __init__(self, window_sec: float = 6.0):
        self.buf = np.zeros(0, dtype=np.float32)
        self.max_len = int(SR * window_sec)

    def feed(self, chunk: np.ndarray) -> dict:
        self.buf = np.concatenate([self.buf, np.asarray(chunk, dtype=np.float32)])[-self.max_len:]
        return analyze(self.buf)


def run_mic(hop_sec=0.1, cooldown=5.0):
    import sounddevice as sd
    det = FireRuleDetector()
    hop = int(SR * hop_sec)
    last_level, last_alert = None, 0.0
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32") as stream:
        print("화재경보 규칙 검출 중... (Ctrl+C 종료)")
        while True:
            chunk, _ = stream.read(hop)
            t0 = time.perf_counter()
            r = det.feed(chunk[:, 0])
            ms = (time.perf_counter() - t0) * 1000
            # 같은 단계는 쿨다운 동안 다시 알리지 않되, CANDIDATE → CONFIRMED 승격은 바로 알림
            if r["level"] and (r["level"] != last_level or time.time() - last_alert >= cooldown):
                print(f"[{r['level']}] 화재경보 의심 pattern={r['pattern']} ({ms:.1f}ms)")
                last_alert = time.time()
            last_level = r["level"]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mic":
        run_mic()
    elif len(sys.argv) > 1:
        print(analyze(load_wav(sys.argv[1])))
    else:
        print(__doc__)
