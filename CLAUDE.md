# HearU 프로젝트 컨텍스트 (Cowork 대화 인계용)

> 이 파일은 Claude Cowork 대화에서 정리한 내용입니다. Claude Code 세션 시작 시 이 파일을 먼저 읽고 작업하세요.

## 프로젝트 개요
- **HearU**: 생활소리(화재경보기·아기 울음·초인종·문 두드림·사람 호출 + 기타)를 AI로 분류하고, FPGA 장치의 LED·진동·OLED와 스마트폰 앱으로 알려주는 청각 보조 시스템
- 팀: 2인 창업 프로젝트 (전자과 친구 = FPGA/Verilog·하드웨어, 나 = 인공지능과, AI + 백엔드 + 앱 담당 예정)
- 예산: 최대 40만원 / 8주 일정 / RISE 사업단 제출용
- 원문 문서 (Notion):
  - 사업계획서: https://app.notion.com/p/RISE-3eadd8ed714e808db498f7f29f987a80
  - AI·앱 개발 가이드: https://app.notion.com/p/HearU-AI-4dda23b8e3e2456cba07e03582ace8f2

## 기술 구성 (개발 가이드 기준)
- 데이터 흐름: INMP441 마이크 → FPGA(I2S, 구간 검출) → UART → 노트북 Python AI → UART → FPGA 알림 / FastAPI → PWA
- AI: 16kHz mono 16-bit, 1~2초 구간, MFCC + MLP 기준 모델 → Mel-spectrogram + 소형 CNN 비교
- 신뢰도: ≥0.80 확정 / 0.60~0.80 약한 알림 / <0.60 UNKNOWN
- 목표 지표: 화재·아기 울음 Recall ≥95%, Macro F1 ≥0.85, Precision ≥90%, UNKNOWN 오인식 ≤10%, 추론 ≤500ms, 전체 알림 ≤1초
- 백엔드: FastAPI (REST: /api/status, /api/events, /api/settings, /api/events/{id}/feedback), WebSocket /ws/events, SQLite
- 앱: PWA (같은 Wi-Fi, QR 접속) → 추후 Flutter 가능. 화면: 홈 / 실시간 알림 / 기록 / 설정
- UART 결과 메시지: `RESULT,class_id,label,confidence(0~100),priority\n`
- 우선순위: FIRE > BABY_CRY > DOORBELL > KNOCK > CALL > UNKNOWN, 동일 클래스 쿨다운 5초
- 권장 폴더: `hearu/{ai, backend, app, shared/labels.json, shared/protocol.md}`

## 기획서 검토에서 발견한 앱 쪽 구멍
1. **개인 맞춤 소리 등록**(사업계획서의 핵심 차별점)이 개발 가이드에 없음 → 녹음 UI, 업로드 API, 커스텀 라벨, 학습 방식 미정
2. **백그라운드 알림 문제**: WebSocket은 화면 꺼짐/백그라운드에서 끊김 → Web Push(HTTPS, 푸시 서버) 필요. iOS는 홈 화면 설치 PWA만 푸시 가능. 보호자 알림은 같은 Wi-Fi로 불가
3. **앱 → 장치 설정 프로토콜 없음**: LED 색상·진동 패턴·야간 모드를 FPGA로 보낼 `CONFIG` 명령 필요
4. **개인정보 충돌**: "원본 미저장" vs 맞춤 학습용 녹음 저장 → 동의·삭제 설계 필요
5. 문서 불일치: 모델(CNN류 vs MLP), 분류 수(4종 vs 5종), 위치 필드 없음(차별화는 "현관—초인종" 위치 안내), 웹앱 일정 7주차 1주에 집중
6. 업무 과부하: AI + 백엔드 + 앱을 1인이 담당 → 5~7주차 몰림
7. 접근성: 소리 피드백 없는 UI, 큰 아이콘, 색에만 의존하지 않는 구분, 쉬운 문장, 고령 사용자 고려

## 팀과 정해야 할 질문
- 맞춤 소리 등록을 MVP에 넣나, 2차로 미루나?
- 실제 청각장애인·난청인 사용자 인터뷰 가능한가?
- 보호자 알림을 MVP에 넣나? 외부 서버/푸시 예산·시간은?
- 화면 꺼진 폰 알림이 시연에 필요한가?
- 앱→장치 UART 명령 규격은 누가 언제 정하나?
- 장치별 위치 정보, 다중 장치 지원 여부?
- FPGA가 PCM 원본 vs 특징값 중 무엇을 보내나?
- labels.json / protocol.md를 단일 기준으로 공동 관리할지?
- PWA 유지 vs Flutter 전환 시점?
- 친구가 기획·디자인·문서를 얼마나 가져갈지?
- 오분류 피드백을 재학습에 어떻게 반영할지?

## 벤치마킹 결과 요약
| 유형 | 제품 | 특징 | HearU와 겹치는 점 |
|---|---|---|---|
| OS 기능 | iPhone 소리 인식 | 화재·초인종·아기 울음 등, 맞춤 초인종·가전음 5회 등록 | 맞춤 등록 |
| OS 기능 | iPhone 이름 인식 (iOS 26) | 이름 호출 감지, 현재 영어만 | 내 이름 호출 |
| OS 기능 | Android 소리 알림 | 생활소리 알림 + 사용자 소리 녹음 등록 발표(2022) | 맞춤 등록 |
| AI 앱 | Taptic (미국) | 온디바이스 AI(YAMNet), 진동·플래시, BT 팔찌·조명, 프리미엄 $5/월 | 위험도 3단계 |
| 고정형 AI 기기 | Earzz (영국) | 소리 발생 위치 근처 설치, 앱·워치 알림, 오디오 미저장, 20+종 중 6개 선택, 피드백 개선, 알림 5초 내 | **가장 유사한 경쟁자** |
| 전통 신호장치 | Bellman Visit | 연기 센서 직접 감지 → 수신기·침대 진동기, 화재 세트 약 $470 | 장치 자체 알림 |
| 국내 신호장치 | 무선 초인등 (벨시스 등) | 버튼 송신기 → LED 수신기, 세트 약 14.5만원 | 초인종 알림 |

### 시사점
- "맞춤 학습", "공간 고정 설치", "원본 미저장"만으로는 차별화 약함
- 남은 강점: **폰 없이 장치에서 즉시 알림 + 네트워크 장애에도 유지**, **한국어 이름 호출**, **보조기기 교부사업 '신호장치'(기준액 58만원) 진입 가능성** (기존 국내 제품은 버튼식, AI 음향 인식형은 못 찾음)
- 화재는 센서 직접 감지가 표준 → 음향 인식은 "보조"로 명시 유지
- 1초 알림 목표 달성 시 Earzz(5초) 대비 비교 포인트
- 시장: 2025년 말 등록장애인 262.8만 명 중 청각장애 17.1%(약 45만 명, 2위). 2025년 신규 등록 중 청각장애 30.6%로 1위, 신규 등록자 59.5%가 65세 이상 → 고령 사용자·보호자 페르소나 필요

### 앱 개발자 실습 과제
1. iPhone 소리 인식에 초인종 직접 등록 → 등록 UX 기준으로 삼기
2. Taptic·Earzz 앱 화면 캡처 → 알림·기록·위험도 표시 비교
3. 문 닫힌 방에서 iPhone vs HearU 비교 시연 → 발표 근거

## 출처
- https://www.macrumors.com/how-to/never-miss-doorbell-wearing-iphone/
- https://maketecheasier.com/iphone-name-recognition-feature/
- https://9to5google.com/2022/09/08/custom-sound-notifications-pixel/
- https://zeroproject.org/view/project/5158227c-0747-f011-877a-6045bde1137d
- https://www.earzz.com/deaf
- https://us.bellman.com/products/alerting-signaling-devices-smoke-fire-alarm-system-smokeclockbedshaker
- https://knat.go.kr/knw/home/knat_DB/assist_detail.php?assist_biz_idx=1
- https://eiec.kdi.re.kr/policy/materialView.do?num=279704
