# HearU

생활소리(화재경보기·아기 울음·초인종·문 두드림·사람 호출)를 AI로 분류하고, FPGA 장치(LED·진동·OLED)와 스마트폰 앱으로 알려주는 청각 보조 시스템.

> **현재 상태**: AI 분류 MVP(MFCC + MLP) 구현. 백엔드·앱·FPGA 연동은 미구현.
> 개발 원칙: 최고 정확도보다 **위험 소리를 놓치지 않는 Recall**, 불필요한 알림을 줄이는 Precision, **1초 이내 반응**, **폰이 끊겨도 유지되는 장치 알림**을 우선한다.

## 1. MVP 목표

마이크로 수집한 생활소리를 AI가 분류하고 결과를 FPGA와 앱에 전달한다.

**완료 조건**
- 입력 오디오를 5종 + 기타로 분류
- 분류 결과와 신뢰도를 FPGA에 UART로 반환
- 앱에서 실시간 알림과 감지 기록 표시
- 화재경보기는 다른 알림보다 우선 처리
- 앱·네트워크가 끊겨도 FPGA 기본 알림 유지

### 클래스

`shared/labels.json`이 클래스·우선순위·임계값의 **단일 기준**이다. AI·FPGA·앱은 이 파일의 `class_id`/`label`/`priority`를 따른다.

| ID | label | 의미 | 우선순위 | 비고 |
|---|---|---|---|---|
| 0 | `UNKNOWN` | 기타 소리 | 6 | 신뢰도가 낮거나 미학습 소리 |
| 1 | `FIRE` | 화재경보기 | 1 | 최우선 위험 알림 |
| 2 | `BABY_CRY` | 아기 울음 | 2 | 사용자 설정으로 ON/OFF |
| 3 | `DOORBELL` | 초인종 | 3 | 일상 알림 |
| 4 | `KNOCK` | 문 두드림 | 4 | 초인종이 없는 상황 보완 |
| 5 | `CALL` | 사람 호출 | 5 | MVP는 통제된 호출음 중심 |

## 2. 시스템 구조

```
INMP441 마이크
      ↓ I2S
FPGA  · 오디오 입력 · 음량 계산 · 유효 소리 구간 검출
      ↓ USB/UART
Python AI 서비스  · 버퍼 수신 · 전처리/특징 추출 · 분류 · 신뢰도
      ├────────→ UART → FPGA 알림 FSM
      └────────→ WebSocket → FastAPI → PWA/앱
```

| 영역 | 담당 기능 | 상태 |
|---|---|---|
| FPGA | I2S 입력, 소리 구간 검출, UART, LED·진동·OLED | 별도 개발 |
| AI (`ai/`) | 전처리, 특징 추출, 분류, 신뢰도·성능 평가 | **MVP 구현** |
| 백엔드 (`backend/`) | 실시간 결과 중계, 기록 저장, 설정 API | 예정 |
| 앱 (`app/`) | 알림 표시, 기록 조회, 감지 설정, 연결 상태 | 예정 |

## 3. AI (`ai/`)

### 입력 사양

| 항목 | 값 |
|---|---|
| 샘플링 레이트 | 16 kHz |
| 채널 / 형식 | Mono / 16-bit PCM (내부에서 -1.0~1.0 float) |
| 입력 구간 | 1.5초 (길면 에너지 최대 구간을 자르고, 짧으면 0 패딩) |

초기 개발은 FPGA 없이 WAV 파일 또는 PC 마이크로 진행한다. FPGA 통합 시에는 (1) PCM 샘플 전송 → (2) 통신 속도가 부족하면 특징값 전송 순으로 간다.

### 파이프라인

```
WAV/PCM → 길이 맞춤 → 피크 정규화 → MFCC(40) → mean·std + delta mean (120차원)
        → StandardScaler → MLP(128, 64) → Softmax 확률 → 신뢰도 판정
```

- 증강(학습 데이터만): 볼륨 변경, 시간 이동, 배경 잡음. **파일 단위로 split한 뒤 증강**해 같은 원본의 변형이 학습·테스트에 동시에 들어가지 않게 한다.
- 비교 모델: Mel-spectrogram + 소형 CNN (데이터가 충분해진 뒤 실험)

| 파일 | 역할 |
|---|---|
| `ai/common.py` | 설정, 오디오 로딩, 특징 추출 (학습·추론 공용) |
| `ai/train.py` | 학습, 평가(클래스별 P/R/F1, 혼동행렬), `model.joblib` 저장 |
| `ai/infer.py` | 파일/마이크 추론, 신뢰도 판정, UART 메시지 생성 |
| `shared/labels.json` | 클래스·임계값 정의 |

### 신뢰도 처리

```
confidence >= 0.80            → 해당 클래스 확정
0.60 <= confidence < 0.80     → 약한 알림 또는 재확인
confidence < 0.60             → UNKNOWN
```

화재경보기는 단일 AI 확률만 쓰지 않고 반복 주기·지속시간 규칙을 함께 확인할 계획이다(미구현).

## 4. 시작하기

Python **3.12** 권장 (librosa/numba가 3.14에서 설치되지 않을 수 있음).

```powershell
cd ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 데이터 준비

`ai/data/<label>/` 폴더에 오디오(`.wav` `.flac` `.ogg` `.mp3`)를 넣는다. 폴더명은 소문자 label이며 `ai/data/`는 git에 올라가지 않는다.

```
ai/data/
├─ fire/        ├─ knock/
├─ baby_cry/    ├─ call/
├─ doorbell/    └─ unknown/   # 대화·TV·음악·키보드·청소기·물소리·반려동물 등 다양하게
```

| 구분 | 최소 권장량 |
|---|---|
| 클래스별 원본 녹음 | 100개 이상 |
| 기타(unknown) | 300개 이상 |
| 총 데이터 | 800개 이상 |

수집 조건: 조용한 방 / TV 켜짐 / 대화 중, 거리 0.5·1·3m, 음량 소·중·대, 스마트폰·노트북·실제 장치 등 다양한 음원, 문 열림·닫힘.
`unknown`이 부족하면 문 두드림·사람 호출을 오인식하기 쉬우므로 목표 클래스보다 다양하게 확보한다.

### 학습 / 추론

```powershell
python train.py                # 학습 + 평가 + model.joblib 저장
python infer.py sample.wav     # 파일 분류
python infer.py --mic          # 마이크 실시간 (1.5초 창, 0.5초 간격, 쿨다운 적용)
```

## 5. 평가 기준

전체 정확도만 보지 않고 안전성과 오경보를 함께 평가한다.

| 지표 | 목표 |
|---|---|
| 화재경보기 Recall | ≥ 95% |
| 아기 울음 Recall | ≥ 95% |
| Macro F1-score | ≥ 0.85 |
| 전체 Precision | ≥ 90% |
| UNKNOWN 오인식률 | ≤ 10% |
| AI 추론시간 | ≤ 500ms (전체 알림 ≤ 1초) |

**필수 산출물**: Confusion Matrix, 클래스별 P/R/F1, 환경별 성능(조용/잡음/문 닫힘), 평균·최대 추론시간, 임계값별 Precision-Recall, 실패 사례 목록과 원인 분석.

> 현재 `train.py`는 Confusion Matrix와 클래스별 P/R/F1까지 출력한다. 나머지는 `evaluate.py`로 추가할 예정이다.

## 6. 인터페이스

### AI 이벤트 (WebSocket / 기록 공통)

```json
{
  "event_id": "evt_20260929_142301_001",
  "class_id": 3,
  "label": "DOORBELL",
  "confidence": 0.94,
  "detected_at": "2026-09-29T14:23:01+09:00",
  "duration_ms": 1450,
  "device_id": "hearu-001",
  "priority": 3
}
```

### UART 결과 메시지 (AI → FPGA)

```
RESULT,class_id,label,confidence(0~100),priority\n
예) RESULT,3,DOORBELL,94,3
```

초기에는 사람이 읽기 쉬운 텍스트를 쓰고, 전송량이 문제가 되면 고정 길이 바이너리 패킷으로 바꾼다. 앱 → 장치 설정(`CONFIG`) 명령은 아직 미정.

### REST / WebSocket API (백엔드 초안, 미구현)

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/status` | 장치 연결·AI 준비 상태, 마지막 감지 시각 |
| GET | `/api/events?label=FIRE&limit=50` | 감지 기록 조회 |
| GET / PUT | `/api/settings` | 감지 소리 ON/OFF, 신뢰도 임계값, 야간 모드 |
| POST | `/api/events/{event_id}/feedback` | 오분류 신고 `{is_correct, correct_label}` |
| WS | `/ws/events` | 위 AI 이벤트 JSON을 그대로 전송 |

설정 예: `{"enabled_labels": ["FIRE","BABY_CRY","DOORBELL","KNOCK","CALL"], "confidence_threshold": 0.80, "night_mode": false}`

### 기록 DB (SQLite `events`)

| 필드 | 형식 | 설명 |
|---|---|---|
| event_id | TEXT | 이벤트 고유값 |
| device_id | TEXT | 감지 장치 ID |
| label | TEXT | 분류 결과 |
| confidence | REAL | AI 신뢰도 |
| detected_at | DATETIME | 감지 시각 |
| duration_ms | INTEGER | 소리 길이 |
| priority | INTEGER | 알림 우선순위 |
| acknowledged | BOOLEAN | 사용자 확인 여부 |
| feedback_label | TEXT/NULL | 오분류 수정값 |

원본 오디오는 기본적으로 저장하지 않는다. 모델 개선용으로 저장해야 하면 사용자의 명시적 동의, 저장 여부 표시, 삭제 기능을 제공한다.

## 7. 앱 (PWA, 예정)

같은 Wi-Fi에서 QR 코드로 접속하는 PWA로 시작하고, 검증 후 Flutter 확장을 검토한다.

| 화면 | 내용 |
|---|---|
| 홈 | 장치 연결 상태, 마지막 감지 소리, 신뢰도, 최근 알림 시각 |
| 실시간 알림 | 소리 종류·아이콘, 신뢰도, 감지 시각, 우선순위, 확인 버튼 |
| 기록 | 날짜별 목록, 소리 종류 필터, 확인 여부, 잘못된 분류 신고 |
| 설정 | 감지 소리 ON/OFF, 클래스별 최소 신뢰도, LED 색·진동 패턴, 야간 모드, 테스트 알림 |

## 8. 예외·우선순위 처리

- 우선순위: `FIRE > BABY_CRY > DOORBELL > KNOCK > CALL > UNKNOWN`
- 화재경보기 발생 중에는 낮은 우선순위 알림이 출력을 덮어쓰지 않는다.
- 동일 클래스는 5초 쿨다운 동안 하나의 이벤트로 묶는다. (`infer.py --mic`에 구현)
- `UNKNOWN`은 기본적으로 진동 알림을 내지 않는다.
- 아기 울음이 사용자 설정 OFF이면 기록·알림을 생략한다.
- 연결 장애: AI 서버 실패 시 FPGA 임계값 기반 기본 경고 유지 / 앱 실패 시 서버에 기록 후 재연결 때 표시 / UART 오류는 재전송·오류 카운터 / WebSocket 종료 시 앱 자동 재연결.

## 9. 개발 순서와 체크리스트

| 단계 | 내용 |
|---|---|
| 1. 인터페이스 모의 | AI는 WAV로 가짜 결과 생성, 앱은 샘플 JSON으로 화면 구현, FPGA 없이 UART 송수신 테스트 |
| 2. 기준 모델 | 데이터 폴더·라벨 확정, MFCC + MLP 학습, 평가 리포트, 모델 저장/로드 ← **현재** |
| 3. 실시간 추론 | PC 마이크 → 버퍼 추론 → 임계값·UNKNOWN 처리 → FastAPI·WebSocket 연결 |
| 4. FPGA 통합 | UART로 PCM/특징값 수신, 결과 UART 반환, 알림 FSM과 class_id 확인 |
| 5. 검증·개선 | 클래스별 20회 이상 시연, 잡음·거리·문 닫힘 테스트, 실패 데이터 재학습, 지연·오경보 최적화 |

**AI**
- [x] 5종 + UNKNOWN 라벨 구조 확정 (`shared/labels.json`)
- [x] MFCC + MLP 기준 모델 학습 (`train.py`)
- [x] 클래스별 평가 리포트 (P/R/F1, Confusion Matrix)
- [x] 실시간 추론 함수, 신뢰도 임계값·UNKNOWN 처리 (`infer.py`)
- [ ] 데이터 수집·분리 스크립트 (현재 train.py가 8:2 단순 분할, 가이드는 70/15/15)
- [ ] 클래스 가중치 MLP, Mel-spectrogram + CNN 비교
- [ ] `evaluate.py`: 환경별 성능, 추론시간, 임계값별 PR, 실패 사례
- [ ] UART 입출력 연결 (`serial_bridge.py`)

**앱·백엔드**
- [ ] FastAPI 서버, WebSocket 실시간 이벤트, SQLite 저장
- [ ] 홈·알림·기록·설정 화면, 감지 항목 ON/OFF
- [ ] 연결 끊김·재연결, 오분류 피드백
- [ ] 실행·배포 방법 문서화

**통합**
- [ ] class_id·label·priority 일치 확인, 화재경보기 우선순위 테스트
- [ ] 전체 알림 지연 1초 이내, 앱 미연결 상태에서 장치 알림 확인
- [ ] 30분 연속 동작 테스트, 시연 시나리오 리허설

## 10. 미정 사항

- 개인 맞춤 소리 등록(녹음 UI, 업로드 API, 커스텀 라벨, 학습 방식)
- 백그라운드 알림: WebSocket은 화면 꺼짐 시 끊기므로 Web Push(HTTPS) 필요. iOS는 홈 화면에 설치한 PWA만 가능
- 앱 → 장치 `CONFIG` 명령 규격(LED 색·진동 패턴·야간 모드)
- FPGA가 PCM 원본 / 특징값 중 무엇을 보낼지
- 위치 정보·다중 장치 지원
- 오분류 피드백의 재학습 반영 방식

## 11. 디렉터리

```
hearu/
├─ ai/            # 분류 모델 (현재). 이후 evaluate.py, models/ 추가
├─ shared/        # labels.json, (예정) protocol.md
├─ backend/       # 예정: main.py, database.py, schemas.py, serial_bridge.py
├─ app/           # 예정: src/, public/
└─ README.md
```

> 화재는 센서 직접 감지가 표준이므로 HearU의 음향 인식은 **보조 수단**이다.
