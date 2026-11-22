# 학습 결과: esp32

- 날짜: 2026-11-22T18:31:00
- 데이터: `ai/data` (총 244개, 녹음 묶음 131개)
- 특징: 136차원 (MFCC 120 + 주파수 변화 10 + 소리 크기 변화 6), 학습 파일당 증강 3개
- 모델 파일: `ai/models/esp32.joblib` (git 제외)

## 데이터 수

| 장치 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|
| esc50 | 80 | 0 | 20 | 0 | 23 |
| esp32 | 0 | 40 | 0 | 40 | 10 |
| 기타 | 0 | 0 | 0 | 21 | 10 |
| **합계** | **80** | **40** | **20** | **61** | **43** |

## 특징 조합별 결과 (A. 파일 단위 교차검증, 숫자는 재현율)

| 조합 | 차원 | 정확도 | Macro F1 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|---|---|---|
| MFCC만 | 120 | 0.80 | 0.79 | 0.74 | 0.80 | 0.70 | 0.89 | 0.86 |
| MFCC+주파수변화 | 130 | 0.85 | 0.83 | 0.82 | 0.78 | 0.70 | 0.92 | 0.93 |
| MFCC+소리크기변화 | 126 | 0.83 | 0.81 | 0.84 | 0.72 | 0.70 | 0.92 | 0.86 |
| 전체(3종) | 136 | 0.84 | 0.81 | 0.81 | 0.72 | 0.65 | 0.95 | 0.93 |

## 3종 전체 모델 평가

### A. 파일 단위 5겹 교차검증 (이전 결과와 비교용)

정확도 0.84 · Macro F1 0.81

| 종류 | 개수 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| UNKNOWN | 80 | 0.81 | 0.81 | 0.81 |
| FIRE | 40 | 0.91 | 0.72 | 0.81 |
| BABY_CRY | 20 | 0.62 | 0.65 | 0.63 |
| DOORBELL | 61 | 0.92 | 0.95 | 0.94 |
| KNOCK | 43 | 0.83 | 0.93 | 0.88 |

녹음 조건별 재현율

| 조건 | 종류 | 개수 | 재현율 |
|---|---|---|---|
| 조용함 30cm | FIRE | 10 | 0.80 |
| 조용함 30cm | DOORBELL | 10 | 1.00 |
| 조용함 2m | FIRE | 10 | 0.80 |
| 조용함 2m | DOORBELL | 10 | 1.00 |
| TV 소리 1m | FIRE | 10 | 0.90 |
| TV 소리 1m | DOORBELL | 10 | 0.90 |

혼동 행렬 (행=정답, 열=예측)

| 정답＼예측 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|
| UNKNOWN | 65 | 2 | 5 | 0 | 8 |
| FIRE | 4 | 29 | 3 | 4 | 0 |
| BABY_CRY | 6 | 0 | 13 | 1 | 0 |
| DOORBELL | 2 | 1 | 0 | 58 | 0 |
| KNOCK | 3 | 0 | 0 | 0 | 40 |

<details><summary>틀린 파일 39개</summary>

| 파일 | 정답 | 예측 |
|---|---|---|
| unknown/esc50_1-155858-A-25_footsteps.wav | UNKNOWN | KNOCK |
| unknown/esc50_1-21934-A-38_clock_tick.wav | UNKNOWN | KNOCK |
| unknown/esc50_1-34094-A-5_cat.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_1-51805-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_2-209475-A-25_footsteps.wav | UNKNOWN | KNOCK |
| unknown/esc50_2-70936-A-42_siren.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_3-216280-A-39_glass_breaking.wav | UNKNOWN | KNOCK |
| unknown/esc50_3-58772-A-42_siren.wav | UNKNOWN | FIRE |
| unknown/esc50_4-108352-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_4-121532-A-42_siren.wav | UNKNOWN | FIRE |
| unknown/esc50_4-133047-C-5_cat.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_4-188293-B-15_water_drops.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-191687-A-0_dog.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-194981-A-25_footsteps.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-201800-A-31_mouse_click.wav | UNKNOWN | KNOCK |
| fire/fire_esp32_20261117_150305_01.wav | FIRE | BABY_CRY |
| fire/fire_esp32_20261117_150305_06.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_07.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261117_150305_08.wav | FIRE | BABY_CRY |
| fire/fire_esp32_20261117_150305_09.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_10.wav | FIRE | BABY_CRY |
| fire/fire_esp32_20261122_174248_quiet_close_05.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_quiet_close_09.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_quiet_far_08.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_quiet_far_10.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_tv_10.wav | FIRE | UNKNOWN |
| baby_cry/esc50_1-211527-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_2-107351-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_2-50665-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_2-66637-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_3-151081-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_4-59579-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_5-198411-F-20_crying_baby.wav | BABY_CRY | DOORBELL |
| doorbell/doorbell_20261117_06.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_esp32_20261117_150305_03.wav | DOORBELL | FIRE |
| doorbell/doorbell_esp32_20261122_175419_tv_02.wav | DOORBELL | UNKNOWN |
| knock/esc50_4-186518-A-30_door_wood_knock.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_03.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_08.wav | KNOCK | UNKNOWN |

</details>

### B. 녹음 묶음 단위 5겹 교차검증 (같은 묶음은 학습·시험에 함께 안 들어감)

정확도 0.80 · Macro F1 0.79

| 종류 | 개수 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| UNKNOWN | 80 | 0.68 | 0.86 | 0.76 |
| FIRE | 40 | 0.93 | 0.70 | 0.80 |
| BABY_CRY | 20 | 0.67 | 0.70 | 0.68 |
| DOORBELL | 61 | 0.96 | 0.89 | 0.92 |
| KNOCK | 43 | 0.86 | 0.70 | 0.77 |

녹음 조건별 재현율

| 조건 | 종류 | 개수 | 재현율 |
|---|---|---|---|
| 조용함 30cm | FIRE | 10 | 0.80 |
| 조용함 30cm | DOORBELL | 10 | 1.00 |
| 조용함 2m | FIRE | 10 | 0.70 |
| 조용함 2m | DOORBELL | 10 | 1.00 |
| TV 소리 1m | FIRE | 10 | 0.90 |
| TV 소리 1m | DOORBELL | 10 | 0.80 |

혼동 행렬 (행=정답, 열=예측)

| 정답＼예측 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|
| UNKNOWN | 69 | 1 | 4 | 1 | 5 |
| FIRE | 10 | 28 | 1 | 1 | 0 |
| BABY_CRY | 5 | 1 | 14 | 0 | 0 |
| DOORBELL | 5 | 0 | 2 | 54 | 0 |
| KNOCK | 13 | 0 | 0 | 0 | 30 |

<details><summary>틀린 파일 49개</summary>

| 파일 | 정답 | 예측 |
|---|---|---|
| unknown/esc50_1-21934-A-38_clock_tick.wav | UNKNOWN | KNOCK |
| unknown/esc50_1-34094-A-5_cat.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_1-51805-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_2-70936-A-42_siren.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_2-83667-A-34_can_opening.wav | UNKNOWN | KNOCK |
| unknown/esc50_3-216280-A-39_glass_breaking.wav | UNKNOWN | KNOCK |
| unknown/esc50_3-58772-A-42_siren.wav | UNKNOWN | FIRE |
| unknown/esc50_4-108352-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_4-121532-A-42_siren.wav | UNKNOWN | DOORBELL |
| unknown/esc50_4-188293-B-15_water_drops.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-191687-A-0_dog.wav | UNKNOWN | KNOCK |
| fire/fire_esp32_20261117_150305_01.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_06.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_07.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_08.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_09.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_10.wav | FIRE | BABY_CRY |
| fire/fire_esp32_20261122_174248_quiet_close_05.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_quiet_close_09.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_07.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_09.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_10.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_tv_10.wav | FIRE | UNKNOWN |
| baby_cry/esc50_1-211527-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_2-50665-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_2-66637-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_3-151081-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_4-59579-A-20_crying_baby.wav | BABY_CRY | FIRE |
| baby_cry/esc50_5-198411-F-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| doorbell/doorbell_20261117_06.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_esp32_20261117_150305_03.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_esp32_20261117_150305_04.wav | DOORBELL | BABY_CRY |
| doorbell/doorbell_esp32_20261117_150305_06.wav | DOORBELL | BABY_CRY |
| doorbell/doorbell_esp32_20261117_150305_08.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_esp32_20261122_175419_tv_02.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_esp32_20261122_175419_tv_04.wav | DOORBELL | UNKNOWN |
| knock/esc50_4-186518-A-30_door_wood_knock.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_02.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_03.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_05.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_08.wav | KNOCK | UNKNOWN |
| knock/knock_esp32_20261117_150305_02.wav | KNOCK | UNKNOWN |
| knock/knock_esp32_20261117_150305_03.wav | KNOCK | UNKNOWN |
| knock/knock_esp32_20261117_150305_04.wav | KNOCK | UNKNOWN |
| knock/knock_esp32_20261117_150305_05.wav | KNOCK | UNKNOWN |
| knock/knock_esp32_20261117_150305_06.wav | KNOCK | UNKNOWN |
| knock/knock_esp32_20261117_150305_07.wav | KNOCK | UNKNOWN |
| knock/knock_esp32_20261117_150305_09.wav | KNOCK | UNKNOWN |
| knock/knock_esp32_20261117_150305_10.wav | KNOCK | UNKNOWN |

</details>
