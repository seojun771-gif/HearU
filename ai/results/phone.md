# 학습 결과: phone

- 날짜: 2026-11-22T19:40:15
- 데이터: `ai/data`, `ai/data_phone` (총 212개, 녹음 묶음 130개), 제외: ['esp32']
- 특징: 136차원 (MFCC 120 + 주파수 변화 10 + 소리 크기 변화 6), 학습 파일당 증강 3개
- 모델 파일: `ai/models/phone.joblib` (git 제외)

## 데이터 수

| 장치 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|
| esc50 | 80 | 0 | 20 | 0 | 23 |
| 기타 | 0 | 0 | 0 | 21 | 10 |
| phone | 0 | 29 | 0 | 29 | 0 |
| **합계** | **80** | **29** | **20** | **50** | **33** |

## 특징 조합별 결과 (A. 파일 단위 교차검증, 숫자는 재현율)

| 조합 | 차원 | 정확도 | Macro F1 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|---|---|---|
| MFCC만 | 120 | 0.79 | 0.77 | 0.85 | 0.72 | 0.80 | 0.70 | 0.82 |
| MFCC+주파수변화 | 130 | 0.78 | 0.77 | 0.81 | 0.76 | 0.80 | 0.76 | 0.76 |
| MFCC+소리크기변화 | 126 | 0.78 | 0.76 | 0.85 | 0.66 | 0.80 | 0.68 | 0.88 |
| 전체(3종) | 136 | 0.78 | 0.77 | 0.81 | 0.79 | 0.75 | 0.70 | 0.85 |

## 3종 전체 모델 평가

### A. 파일 단위 5겹 교차검증 (이전 결과와 비교용)

정확도 0.78 · Macro F1 0.77

| 종류 | 개수 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| UNKNOWN | 80 | 0.81 | 0.81 | 0.81 |
| FIRE | 29 | 0.72 | 0.79 | 0.75 |
| BABY_CRY | 20 | 0.68 | 0.75 | 0.71 |
| DOORBELL | 50 | 0.85 | 0.70 | 0.77 |
| KNOCK | 33 | 0.76 | 0.85 | 0.80 |

혼동 행렬 (행=정답, 열=예측)

| 정답＼예측 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|
| UNKNOWN | 65 | 0 | 6 | 1 | 8 |
| FIRE | 0 | 23 | 0 | 5 | 1 |
| BABY_CRY | 5 | 0 | 15 | 0 | 0 |
| DOORBELL | 6 | 8 | 1 | 35 | 0 |
| KNOCK | 4 | 1 | 0 | 0 | 28 |

<details><summary>틀린 파일 46개</summary>

| 파일 | 정답 | 예측 |
|---|---|---|
| unknown/esc50_1-34094-A-5_cat.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_1-51805-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_2-118072-A-0_dog.wav | UNKNOWN | KNOCK |
| unknown/esc50_2-209475-A-25_footsteps.wav | UNKNOWN | KNOCK |
| unknown/esc50_2-70936-A-42_siren.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_3-170383-A-38_clock_tick.wav | UNKNOWN | KNOCK |
| unknown/esc50_3-182710-A-35_washing_machine.wav | UNKNOWN | DOORBELL |
| unknown/esc50_3-58772-A-42_siren.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_4-108352-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_4-121532-A-42_siren.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-133047-C-5_cat.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_4-188293-B-15_water_drops.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-191687-A-0_dog.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-194981-A-25_footsteps.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-201800-A-31_mouse_click.wav | UNKNOWN | KNOCK |
| baby_cry/esc50_1-211527-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_2-66637-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_3-151081-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_4-59579-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_5-198411-F-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| doorbell/doorbell_20261117_02.wav | DOORBELL | BABY_CRY |
| doorbell/doorbell_20261117_05.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_06.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_08.wav | DOORBELL | UNKNOWN |
| knock/esc50_3-151089-A-30_door_wood_knock.wav | KNOCK | UNKNOWN |
| knock/esc50_4-186518-A-30_door_wood_knock.wav | KNOCK | UNKNOWN |
| knock/esc50_5-218981-A-30_door_wood_knock.wav | KNOCK | FIRE |
| knock/knock_20261117_01.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_03.wav | KNOCK | UNKNOWN |
| fire/fire_phone_20261122_p1_07.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p1_09.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p1_10.wav | FIRE | KNOCK |
| fire/fire_phone_20261122_p3_22.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p3_25.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p3_26.wav | FIRE | DOORBELL |
| doorbell/doorbell_phone_20261122_p1_03.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p1_05.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p1_06.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p1_07.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p2_14.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_phone_20261122_p2_15.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p2_18.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_phone_20261122_p3_25.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_phone_20261122_p3_26.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p3_28.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p3_29.wav | DOORBELL | FIRE |

</details>

### B. 녹음 묶음 단위 5겹 교차검증 (같은 묶음은 학습·시험에 함께 안 들어감)

정확도 0.73 · Macro F1 0.71

| 종류 | 개수 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| UNKNOWN | 80 | 0.74 | 0.85 | 0.79 |
| FIRE | 29 | 0.68 | 0.59 | 0.63 |
| BABY_CRY | 20 | 0.68 | 0.65 | 0.67 |
| DOORBELL | 50 | 0.72 | 0.62 | 0.67 |
| KNOCK | 33 | 0.79 | 0.79 | 0.79 |

혼동 행렬 (행=정답, 열=예측)

| 정답＼예측 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|
| UNKNOWN | 68 | 0 | 4 | 3 | 5 |
| FIRE | 1 | 17 | 1 | 9 | 1 |
| BABY_CRY | 7 | 0 | 13 | 0 | 0 |
| DOORBELL | 9 | 8 | 1 | 31 | 1 |
| KNOCK | 7 | 0 | 0 | 0 | 26 |

<details><summary>틀린 파일 57개</summary>

| 파일 | 정답 | 예측 |
|---|---|---|
| unknown/esc50_1-34094-A-5_cat.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_1-51805-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_2-135728-A-38_clock_tick.wav | UNKNOWN | KNOCK |
| unknown/esc50_2-70936-A-42_siren.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_2-83667-A-34_can_opening.wav | UNKNOWN | KNOCK |
| unknown/esc50_3-182710-A-35_washing_machine.wav | UNKNOWN | DOORBELL |
| unknown/esc50_3-58772-A-42_siren.wav | UNKNOWN | DOORBELL |
| unknown/esc50_4-108352-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_4-121532-A-42_siren.wav | UNKNOWN | DOORBELL |
| unknown/esc50_4-181865-A-38_clock_tick.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-188293-B-15_water_drops.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-191687-A-0_dog.wav | UNKNOWN | KNOCK |
| baby_cry/esc50_1-211527-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_2-66637-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_3-151080-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_3-151081-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_4-59579-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_5-151085-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_5-198411-F-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| doorbell/doorbell_20261117_02.wav | DOORBELL | BABY_CRY |
| doorbell/doorbell_20261117_03.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_04.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_05.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_06.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_08.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_09.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_10.wav | DOORBELL | UNKNOWN |
| knock/esc50_1-26188-A-30_door_wood_knock.wav | KNOCK | UNKNOWN |
| knock/esc50_4-186518-A-30_door_wood_knock.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_01.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_03.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_04.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_05.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_08.wav | KNOCK | UNKNOWN |
| fire/fire_phone_20261122_p1_08.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p1_10.wav | FIRE | KNOCK |
| fire/fire_phone_20261122_p2_13.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p2_14.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p2_15.wav | FIRE | UNKNOWN |
| fire/fire_phone_20261122_p2_17.wav | FIRE | BABY_CRY |
| fire/fire_phone_20261122_p3_21.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p3_22.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p3_23.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p3_24.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p3_25.wav | FIRE | DOORBELL |
| fire/fire_phone_20261122_p3_26.wav | FIRE | DOORBELL |
| doorbell/doorbell_phone_20261122_p1_01.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p1_03.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p1_05.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p1_06.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p1_07.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_phone_20261122_p2_13.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p2_15.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p2_18.wav | DOORBELL | KNOCK |
| doorbell/doorbell_phone_20261122_p3_25.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_phone_20261122_p3_27.wav | DOORBELL | FIRE |
| doorbell/doorbell_phone_20261122_p3_28.wav | DOORBELL | FIRE |

</details>

### C. 다른 장치 녹음으로 시험 (esp32 60개, 학습에 안 씀)

정확도 0.53 · Macro F1 0.56

| 종류 | 개수 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| FIRE | 30 | 1.00 | 0.13 | 0.24 |
| DOORBELL | 30 | 0.82 | 0.93 | 0.88 |

녹음 조건별 재현율

| 조건 | 종류 | 개수 | 재현율 |
|---|---|---|---|
| 조용함 30cm | FIRE | 10 | 0.10 |
| 조용함 30cm | DOORBELL | 10 | 1.00 |
| 조용함 2m | FIRE | 10 | 0.10 |
| 조용함 2m | DOORBELL | 10 | 1.00 |
| TV 소리 1m | FIRE | 10 | 0.20 |
| TV 소리 1m | DOORBELL | 10 | 0.80 |

혼동 행렬 (행=정답, 열=예측)

| 정답＼예측 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|
| UNKNOWN | 0 | 0 | 0 | 0 | 0 |
| FIRE | 20 | 4 | 0 | 6 | 0 |
| BABY_CRY | 0 | 0 | 0 | 0 | 0 |
| DOORBELL | 1 | 0 | 1 | 28 | 0 |
| KNOCK | 0 | 0 | 0 | 0 | 0 |

<details><summary>틀린 파일 28개</summary>

| 파일 | 정답 | 예측 |
|---|---|---|
| fire/fire_esp32_20261122_174248_quiet_close_01.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_close_02.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_close_04.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_quiet_close_05.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_quiet_close_06.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_close_07.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_close_08.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_close_09.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_close_10.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_01.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_02.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_03.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_04.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_05.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_quiet_far_06.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_07.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_quiet_far_08.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_quiet_far_10.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_tv_01.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_tv_02.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_tv_03.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_tv_04.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_tv_05.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261122_174248_tv_06.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_tv_07.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261122_174248_tv_10.wav | FIRE | UNKNOWN |
| doorbell/doorbell_esp32_20261122_175419_tv_02.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_esp32_20261122_175419_tv_04.wav | DOORBELL | BABY_CRY |

</details>
