# 학습 결과: baseline

- 날짜: 2026-11-22T00:19:12
- 데이터: `ai/data` (총 184개)
- 평가: 5-fold stratified CV, file-level, aug x3 on train folds only
- 특징: 136차원 (MFCC 120 + 주파수 변화 10 + 소리 크기 변화 6)

## 데이터 수

| UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|
| 80 | 10 | 20 | 31 | 43 |

## 특징 조합별 결과 (재현율)

| 조합 | 차원 | 정확도 | Macro F1 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|---|---|---|
| MFCC만 | 120 | 0.78 | 0.68 | 0.82 | 0.20 | 0.70 | 0.71 | 0.93 |
| MFCC+주파수변화 | 130 | 0.76 | 0.68 | 0.76 | 0.20 | 0.70 | 0.71 | 0.95 |
| MFCC+소리크기변화 | 126 | 0.79 | 0.70 | 0.81 | 0.20 | 0.70 | 0.81 | 0.93 |
| 전체(3종) | 136 | 0.77 | 0.70 | 0.79 | 0.30 | 0.65 | 0.77 | 0.91 |

## 기준 모델(전체 3종) 종류별 성능

| 종류 | 개수 | 정밀도 | 재현율 | F1 |
|---|---|---|---|---|
| UNKNOWN | 80 | 0.77 | 0.79 | 0.78 |
| FIRE | 10 | 0.60 | 0.30 | 0.40 |
| BABY_CRY | 20 | 0.59 | 0.65 | 0.62 |
| DOORBELL | 31 | 0.96 | 0.77 | 0.86 |
| KNOCK | 43 | 0.78 | 0.91 | 0.84 |

## 혼동 행렬 (행=정답, 열=예측)

| 정답＼예측 | UNKNOWN | FIRE | BABY_CRY | DOORBELL | KNOCK |
|---|---|---|---|---|---|
| UNKNOWN | 63 | 0 | 6 | 0 | 11 |
| FIRE | 4 | 3 | 2 | 1 | 0 |
| BABY_CRY | 7 | 0 | 13 | 0 | 0 |
| DOORBELL | 4 | 2 | 1 | 24 | 0 |
| KNOCK | 4 | 0 | 0 | 0 | 39 |

## 틀린 파일 (42개)

| 파일 | 정답 | 예측 |
|---|---|---|
| unknown/esc50_1-100032-A-0_dog.wav | UNKNOWN | KNOCK |
| unknown/esc50_1-155858-A-25_footsteps.wav | UNKNOWN | KNOCK |
| unknown/esc50_1-21934-A-38_clock_tick.wav | UNKNOWN | KNOCK |
| unknown/esc50_1-34094-A-5_cat.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_1-51805-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_2-118072-A-0_dog.wav | UNKNOWN | KNOCK |
| unknown/esc50_2-209475-A-25_footsteps.wav | UNKNOWN | KNOCK |
| unknown/esc50_2-70936-A-42_siren.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_2-82274-A-5_cat.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_3-170383-A-38_clock_tick.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-108352-A-33_door_wood_creaks.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_4-121532-A-42_siren.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-133047-C-5_cat.wav | UNKNOWN | BABY_CRY |
| unknown/esc50_4-188293-B-15_water_drops.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-191687-A-0_dog.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-194981-A-25_footsteps.wav | UNKNOWN | KNOCK |
| unknown/esc50_4-204123-A-39_glass_breaking.wav | UNKNOWN | KNOCK |
| fire/fire_esp32_20261117_150305_01.wav | FIRE | BABY_CRY |
| fire/fire_esp32_20261117_150305_03.wav | FIRE | DOORBELL |
| fire/fire_esp32_20261117_150305_06.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_07.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_08.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_09.wav | FIRE | UNKNOWN |
| fire/fire_esp32_20261117_150305_10.wav | FIRE | BABY_CRY |
| baby_cry/esc50_1-211527-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_2-50665-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_2-66637-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_3-151081-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_3-152007-C-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_4-59579-A-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| baby_cry/esc50_5-198411-F-20_crying_baby.wav | BABY_CRY | UNKNOWN |
| doorbell/doorbell_20261117_02.wav | DOORBELL | BABY_CRY |
| doorbell/doorbell_20261117_04.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_05.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_20261117_06.wav | DOORBELL | UNKNOWN |
| doorbell/doorbell_esp32_20261117_150305_03.wav | DOORBELL | FIRE |
| doorbell/doorbell_esp32_20261117_150305_06.wav | DOORBELL | FIRE |
| doorbell/doorbell_esp32_20261117_150305_09.wav | DOORBELL | UNKNOWN |
| knock/esc50_3-151089-A-30_door_wood_knock.wav | KNOCK | UNKNOWN |
| knock/esc50_4-186518-A-30_door_wood_knock.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_03.wav | KNOCK | UNKNOWN |
| knock/knock_20261117_08.wav | KNOCK | UNKNOWN |
