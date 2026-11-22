"""기준 모델용 특징 추출: MFCC + 주파수 변화 + 소리 크기 변화 (총 136차원).

계산 방법과 계수 구성은 docs/FEATURES.md 참고. 학습/추론이 반드시 이 함수 하나를 같이 써야 한다.
"""
import librosa
import numpy as np

from common import SR, fix_length

N_FFT = 512          # 32ms 분석 창 (16kHz)
HOP = 256            # 16ms 간격
N_MFCC = 40          # c0~c39 (c0 = 전체 에너지에 해당)
N_MELS = 128         # librosa 기본값, 0~8000Hz
ROLLOFF = 0.85       # 롤오프: 에너지 85%가 모이는 주파수
RISE_DB = 6.0        # 한 프레임 사이 이만큼 커지면 '급격한 증가'로 셈
MIN_HZ = 80          # 대표 주파수 탐색 하한 (전원 험·직류 성분 제외)


def _names():
    mfcc = [f"mfcc{i}_mean" for i in range(N_MFCC)] + [f"mfcc{i}_std" for i in range(N_MFCC)] \
        + [f"mfcc{i}_delta_mean" for i in range(N_MFCC)]
    freq = ["centroid_mean", "centroid_std", "centroid_change", "rolloff_mean", "rolloff_std",
            "dominant_mean", "dominant_std", "dominant_change", "flux_mean", "flux_std"]
    loud = ["db_mean", "db_std", "db_range", "db_change", "db_max_rise", "db_rise_count"]
    return {"MFCC": mfcc, "주파수변화": freq, "소리크기변화": loud}


GROUPS = _names()
FEATURE_NAMES = [n for g in GROUPS.values() for n in g]


def prepare(y: np.ndarray) -> np.ndarray:
    """원본 파형 -> 1.5초(가장 에너지 큰 구간) + 최대값 1로 정규화."""
    y = fix_length(y.astype(np.float32))
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def extract_groups(y: np.ndarray) -> dict:
    y = prepare(y)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2      # 파워 스펙트럼 (257 x 프레임)
    fr = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    # 1) MFCC: 40계수의 평균, 표준편차, 1차 변화량(delta) 평균
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    delta = librosa.feature.delta(mfcc)
    f_mfcc = np.concatenate([mfcc.mean(1), mfcc.std(1), delta.mean(1)])

    # 2) 주파수 변화
    cen = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    roll = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=ROLLOFF)[0]
    lo = int(np.searchsorted(fr, MIN_HZ, side="right"))
    dom = fr[lo + S[lo:].argmax(0)]
    Sn = S / (S.sum(0, keepdims=True) + 1e-10)                            # 음량과 무관한 모양 변화
    flux = np.sqrt((np.diff(Sn, axis=1) ** 2).sum(0))
    f_freq = np.array([cen.mean(), cen.std(), np.abs(np.diff(cen)).mean(), roll.mean(), roll.std(),
                       dom.mean(), dom.std(), np.abs(np.diff(dom)).mean(), flux.mean(), flux.std()])

    # 3) 소리 크기 변화 (프레임 에너지 dB)
    db = 10 * np.log10(S.sum(0) + 1e-10)
    d = np.diff(db)
    f_loud = np.array([db.mean(), db.std(), db.max() - db.min(), np.abs(d).mean(), d.max(),
                       float((d > RISE_DB).sum())])

    return {"MFCC": f_mfcc, "주파수변화": f_freq, "소리크기변화": f_loud}


def extract_features(y: np.ndarray) -> np.ndarray:
    g = extract_groups(y)
    return np.concatenate([g[k] for k in GROUPS])
