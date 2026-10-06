# 코드 읽기 가이드 (ai/)

`ai/` 폴더의 코드를 처음 보는 팀원을 위한 안내서입니다. 위에서 아래로 순서대로 읽으면서 해당 파일을 같이 열어 보세요.
실행 방법과 전체 구조는 [README](../README.md)를 참고하세요.

## 한눈에 보기

```
소리(wav / 마이크)
   │  common.load_wav / 마이크 버퍼
   ▼
common.extract      ← 소리를 숫자 120개(MFCC)로 요약
   ▼
model.joblib        ← train.py가 만든 MLP 모델
   ▼
infer.classify      ← 클래스 + 신뢰도 + 판정(확정/약함/UNKNOWN)
   ▼
infer.to_uart       ← "RESULT,1,FIRE,89,1\n"  → FPGA
```

| 읽는 순서 | 파일 | 한 줄 요약 | 이런 팀원에게 중요 |
|---|---|---|---|
| 1 | `shared/labels.json` | 클래스 번호·이름·우선순위·임계값의 단일 기준 | 모두 |
| 2 | `ai/common.py` | 소리 → 숫자 변환 (학습·추론 공용) | AI |
| 3 | `ai/train.py` | 데이터로 모델 학습·평가·저장 | AI |
| 4 | `ai/infer.py` | 추론, 신뢰도 판정, UART 메시지, 실시간 마이크 | **FPGA**, AI |

---

## 1. `shared/labels.json` — 약속 파일

```json
{"id": 1, "name": "FIRE", "ko": "화재경보기", "priority": 1}
```

- `id`: UART로 나가는 class_id. **FPGA의 클래스 번호와 반드시 같아야 합니다.**
- `priority`: 숫자가 작을수록 우선 (FIRE=1 … UNKNOWN=6).
- `thresholds`: `confirm 0.80`(확정), `weak 0.60`(약한 알림). 이 아래는 UNKNOWN.
- `cooldown_sec`: 같은 클래스 재알림 최소 간격(5초).

| id | name | 의미 | priority |
|---|---|---|---|
| 0 | UNKNOWN | 기타 | 6 |
| 1 | FIRE | 화재경보기 | 1 |
| 2 | BABY_CRY | 아기 울음 | 2 |
| 3 | DOORBELL | 초인종 | 3 |
| 4 | KNOCK | 문 두드림 | 4 |
| 5 | CALL | 사람 호출 | 5 |

번호나 임계값을 바꾸고 싶으면 코드가 아니라 **이 파일만** 수정하세요. 코드는 여기서 읽어 갑니다.

---

## 2. `ai/common.py` — 소리를 숫자로 바꾸기

모델은 파형을 그대로 못 읽기 때문에 "소리의 특징"을 숫자 벡터로 바꿔서 넣습니다. 학습할 때와 추론할 때 **같은 함수**를 써야 결과가 어긋나지 않아서 한 파일에 모아 뒀습니다.

### 상수

| 이름 | 값 | 의미 |
|---|---|---|
| `SR` | 16000 | 샘플링 레이트 (Hz) |
| `DURATION` / `N_SAMPLES` | 1.5초 / 24000 | 모델 입력 길이 |
| `N_MFCC` | 40 | MFCC 계수 개수 |

### 함수

- **`load_wav(path)`**: 파일을 16kHz mono로 읽습니다. 다른 샘플레이트 파일도 자동 변환됩니다.
- **`fix_length(y)`**: 길이를 24000샘플로 맞춥니다.
  - 길면: 0.1초씩 밀면서 에너지(제곱합)가 가장 큰 1.5초 구간을 잘라 씁니다.
  - 짧으면: 뒤를 0으로 채웁니다.
- **`extract(y)`**: 특징 벡터(120차원)를 만듭니다.
  1. `fix_length`로 길이 맞춤
  2. 최대 진폭으로 나눠 정규화 → 마이크 거리·볼륨 차이 완화
  3. MFCC 40개 × 시간축 → 계수별 **평균(40)** + **표준편차(40)**
  4. MFCC 변화량(delta)의 **평균(40)** → 합쳐서 120차원

> **전자과 관점 설명**: MFCC는 FFT로 주파수 성분을 구한 뒤, 사람 귀처럼 저주파를 촘촘하게(멜 스케일) 묶고 로그·DCT로 압축한 값입니다. "이 소리의 주파수 모양"을 40개 숫자로 요약한 것이라고 보면 됩니다.

---

## 3. `ai/train.py` — 학습

```
data/<label>/*.wav → 파일 단위 train/test 분리 → train에만 증강 → 특징 추출
                  → StandardScaler + MLP 학습 → 평가 출력 → model.joblib 저장
```

### 읽을 함수

- **`load_files()`**: `ai/data/fire/`, `baby_cry/`, … 폴더를 읽어 (파일 경로, 클래스 id) 목록을 만듭니다. 클래스별 파일 수를 출력하고, 데이터가 하나도 없으면 안내 후 종료합니다.
- **`augment(y)`**: 학습 데이터 부족을 보완하는 증강. 볼륨 0.5~1.5배, ±0.2초 시간 이동, 약한 백색 잡음.
- **`build(paths, labels, n_aug)`**: 각 파일의 특징을 추출합니다. `n_aug`가 3이면 원본 1개 + 증강 3개 = 4개로 늘립니다.
- **`main()`**: 전체 흐름.

### 꼭 알아둘 설계 포인트

1. **파일 단위로 먼저 나눈 뒤 증강**합니다. 같은 녹음의 변형이 학습과 테스트에 동시에 들어가면 성능이 부풀려지기 때문입니다.
2. 모델은 `StandardScaler → MLPClassifier(128, 64)`를 `make_pipeline`으로 묶었습니다. 저장/로드 시 정규화까지 함께 따라갑니다.
3. 출력되는 `classification_report`에서 **FIRE, BABY_CRY의 recall**을 가장 먼저 보세요(목표 95% 이상). 위험 소리를 놓치지 않는 것이 핵심 지표입니다.
4. 혼동행렬은 행=정답, 열=예측입니다. 대각선 밖의 큰 값이 서로 헷갈리는 클래스 쌍입니다.

### 가이드와 아직 다른 점

- 데이터 분할이 8:2입니다. 가이드는 70/15/15 (검증 세트 없음).
- 클래스 가중치와 Mel-spectrogram + CNN 비교는 아직 없습니다.

---

## 4. `ai/infer.py` — 추론과 FPGA 출력 (FPGA 팀원 필독)

### 함수

- **`classify(y)`**: 파형 → 결과 딕셔너리.
  1. `extract`로 특징 추출 → 모델의 `predict_proba`로 클래스별 확률
  2. 가장 높은 확률 = `confidence`, 해당 클래스 = 후보
  3. 판정:

     | confidence | level | 동작 |
     |---|---|---|
     | ≥ 0.80 | `CONFIRMED` | 해당 클래스 확정 |
     | 0.60 ~ 0.80 | `WEAK` | 클래스는 유지, 약한 알림 |
     | < 0.60 | `UNKNOWN` | 클래스를 UNKNOWN(0)으로 덮어씀 |

  반환 예:
  ```python
  {'class_id': 1, 'label': 'FIRE', 'ko': '화재경보기',
   'confidence': 0.89, 'level': 'CONFIRMED', 'priority': 1}
  ```

- **`to_uart(r)`**: 결과를 FPGA용 한 줄 텍스트로 만듭니다.
  ```
  RESULT,class_id,label,confidence(0~100),priority\n
  RESULT,1,FIRE,89,1
  ```
  confidence는 0~100 정수로 반올림해서 보냅니다. **FPGA가 이 형식을 파싱할 수 있는지** 확인이 필요합니다.

- **`run_mic()`**: PC 마이크 실시간 모드 (`python infer.py --mic`).
  - 1.5초 버퍼를 0.5초마다 한 칸씩 밀면서 매번 추론합니다(슬라이딩 윈도).
  - 평균 음량(RMS)이 0.01 미만이면 무음으로 보고 건너뜁니다.
  - UNKNOWN이거나, 같은 클래스가 5초 안에 또 감지되면 출력하지 않습니다(쿨다운).
  - 추론 시간(ms)을 같이 출력합니다. 목표는 500ms 이하입니다.

### 아직 없는 것

- `run_mic()`는 **PC 마이크용**입니다. FPGA에서 UART로 PCM/특징값을 받는 입력부와, 결과를 UART 포트로 실제로 보내는 코드(`serial_bridge.py`)는 아직 없습니다. 지금은 메시지 문자열을 만들어 화면에 출력하는 단계입니다.
- 가이드의 "FPGA 임계값 기반 기본 경고 유지"(AI 서버가 죽었을 때)는 FPGA 쪽 구현입니다.

---

## 직접 따라 해 보기

```powershell
cd ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. **특징 추출 확인**: 파이썬 셸에서 아래를 실행하면 `(120,)`이 나옵니다.
   ```python
   from common import load_wav, extract
   print(extract(load_wav("sample.wav")).shape)
   ```
2. **학습**: `ai/data/<label>/`에 오디오를 넣고 `python train.py`. 실제 데이터가 아직 없으면 사인파 등으로 만든 가짜 파일로 흐름만 확인할 수 있지만, **그 수치는 성능이 아닙니다.**
3. **추론**: `python infer.py sample.wav` → 결과 딕셔너리와 `RESULT,...` 줄이 출력됩니다.
4. **실시간**: `python infer.py --mic` (마이크 필요).

## 팀원에게 확인받을 것

- [ ] UART `RESULT,...` 텍스트 형식을 FPGA에서 받을 수 있는가 (아니면 바이너리 패킷으로 변경)
- [ ] FPGA가 PCM 원본을 보낼지, 특징값을 보낼지 (UART 전송량 문제)
- [ ] `labels.json`의 class_id가 FPGA 알림 FSM의 번호와 일치하는가
- [ ] 앱 → 장치 설정(`CONFIG`) 명령 규격을 누가 언제 정하는가
- [ ] 실제 녹음 데이터 수집 분담 (클래스별 100개 이상, unknown 300개 이상)

## 알려진 한계

- 실제 데이터로 학습·평가한 적이 없어 성능 수치가 없습니다.
- 화재경보기의 반복 주기·지속시간 규칙(가이드)은 미구현이라 현재는 모델 확률만 사용합니다.
- 환경별(조용/잡음/문 닫힘) 평가, 추론시간 통계, 임계값별 Precision-Recall은 `evaluate.py`로 추가 예정입니다.
