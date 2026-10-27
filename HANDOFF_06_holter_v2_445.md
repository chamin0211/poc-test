# 인수인계: Holter v2 전처리·증강 → 구글 드라이브 업로드 (2026-10-27)

> **가장 최신 (2026-10-27 저녁) — 아래 "최신 상태"보다 우선**
> - **사용자 결정: 특징 계산(30초 조각 특징 28개, HRV)은 쓰지 않는다.** 필요한 건 정제된 파형 `.npy`(원본 445 + 증강본 890)뿐. 드라이브는 `ECG_tem_clean`만 채움. 학습 계획은 CNN + LSTM/Transformer(파형 직접 학습)
> - 노트북 3 전체 실행은 **8명 처리 후 사용자 요청으로 중지함**. 만들다 만 결과가 남아 있음 → 다시 돌리기 전에 `/mnt/c/data/music/SCD_Holter_v2_pca/`(clean_signal, segment_features)와 `results/preprocess_swee/record_log.csv`를 지울 것
> - **3유도를 섞는 기준이 결정 대기 중** (사용자에게 물어볼 것). 지금 코드는 첫 번째 주성분(전체 분산, 주로 QRS) = 근거 없는 자체 판단. 선택지: A 그대로 / B T파 기준 PCA(Ramírez 2017 JAHA, Palmieri 2021 BSPC 68:102719 — 저널 근거 있음, T파 위치 검출 필요) / C 섞지 않고 3유도 그대로 `.npy` 저장(약 105 GB, C: 여유 약 143 GB). Claude는 C를 가장 단순하다고 제안함. 섞는 결정(10-27)을 바꾸는 것이라 사용자 결정 필요
> - D안도 있음: **Swee처럼 Y 유도만** (Swee 2026 본문 "Only Lead Y was used", 근거 가장 분명, 구현 가장 쉬움)
> - B안 세부 근거 찾기 (원문 미확인): Palmieri 2021 BSPC 68:102719 (오픈 액세스, "PCA learned on the T wave", **MUSIC 아님**: 혈액투석 환자 29명, Holter 여부 미확인) → Castells 2007 → Zaragoza 그룹의 다른 MUSIC 논문. 확인할 것: T파 구간을 잡는 방법(R파 기준 고정 창/파형 구분), 쓴 박동(평균/전체), 몇 번째 주성분, 환자마다 계산했는지
> - QRS 기준 PCA 합치기의 저널 근거는 못 찾음 (Martens 2005는 학회 논문. Scientia Iranica 2011 Homaeinezhad는 PCA로 유도를 합쳤는지 확인 안 됨)
> - 결정되면 노트북 3 수정: `segment_features`·CSV 저장·R파 검출(`nk.ecg_peaks`) 제거, `.npy` 저장만 남김 (환자 1명 1~2분 예상). 기준 표 6~10번은 "사용 안 함"으로 표시, 결정 기록에 추가

> **최신 상태 (2026-10-27 16:50)**: 계획이 바뀌어 **학교 컴퓨터에서 실행 중**이다. 아래 4장의 "집에서" 순서는 학교 컴퓨터에서 이미 진행됨.
> - 원본 Holter 936개 다시 받음 (`/mnt/c/data/music/Holter_ECG`, 크기 검사 통과, 크기가 다른 39개는 SHA256 일치)
> - 노트북 1 실행 완료: 445명 / SCD 50명, `results/final_holter_ecg/cohort_final.csv` 생성
> - **neurokit2 0.2.12 → 0.2.13으로 올림 (pandas 3.1 → 2.3.3)**. 0.2.12는 numpy 2.5에서 `np.trapz`가 없어 HRV 주파수·비선형 계산이 전부 실패했음 (HRV 21개 100% 빈 값). 0.2.13으로 시험 환자 HRV 빈 값 0.2%
> - 노트북 3: `N_JOBS = 8`(1명 최대 메모리 1.24 GB, 1명 약 12분)로 **전체 실행 중** (16:40 시작, 약 11시간 예상). `setsid nohup jupyter nbconvert --execute --inplace`로 따로 떨어진 프로세스라 VS Code를 닫아도 돎. 진행 확인: `results/preprocess_swee/record_log.csv` 줄 수 − 1 = 끝난 환자 수. 끊기면 노트북 3을 다시 실행하면 남은 환자부터 이어서 함
> - 끝나면: ⑥ 출력 확인(전원 ok, `.npy`·CSV 1,335개씩, 짝 안 맞는 이름 없음) → 드라이브 README 작성 → 업로드 (5장)
> - 참고: `HRV_SampEn`에 inf, 잡음 조각은 `HRV_SDNN`이 매우 큼. neurokit이 그대로 낸 값이고 Swee도 같음 → README에 적을 것
> - 특징 CSV는 Swee의 MIL-TCN(딥러닝) 입력이다. 증강본 특징(`P0001-1.csv` 등)도 증강 학습에 필요해서 계산함

다음 AI가 이어서 할 일: **집 컴퓨터에서 사용자가 노트북 1 → 3 (필요하면 2)을 실행하도록 안내하고, 결과를 팀 구글 드라이브에 올리는 것까지 돕기.** 모델 학습은 이 인수인계 범위가 아니다.

## 1. 사용자 작업 방식 (꼭 지킬 것)

| 규칙 | 내용 |
|---|---|
| 실행은 사용자가 | 노트북 셀·스크립트·학습을 대신 실행하지 않는다. 순서, 확인할 출력, 짧은 코드만 안내한다. 사용자가 "너가 해줘", "수정해줘"라고 특정 작업을 시킬 때만 직접 한다 |
| 학습은 절대 대신 안 함 | train / validation / evaluation은 사용자가 직접 한다 |
| 결정 기록 | 노트북의 "결정 기록"에는 사용자가 정한 것만. Claude 제안은 "검토 중 (제안)" 표에 |
| 기준에는 근거 | 새 임계값·기준은 논문 근거부터. 근거가 없으면 "근거 없음"이라고 분명히 쓴다 |
| 쉬운 말 | 설명과 정리 페이지는 쉬운 한국어로. 데이터 칸 이름·코드는 보조로만. 지금 쓰는 기준(445명)만 쓰고 지난 숫자(730명 등)는 꺼내지 않는다 |
| 언어 | **항상 한국어로 답한다** |
| 저장소 | 이 저장소(`chamin0211/poc-test`)는 개인 저장소라 main에 바로 push해도 된다고 사용자가 말함 (공개 저장소라 원본 파형·논문 PDF는 올리지 않음) |

## 2. 지금까지 확정된 것 (사용자 결정)

| 항목 | 내용 |
|---|---|
| 환자 | **445명으로 고정** (앞으로 계속). 팀 분할 파일 `final_split.csv`(508명)에서 등록 전 심장 기기 환자 63명(`Prior implantable device` ≠ 0)을 뺀 것 |
| 정답 | `SCD_4year_label` = 연구 시작 후 4년(1,460일) 안에 SCD로 사망하면 1. **SCD 50명 / 없음 395명**. 팀 파일의 `SCD` 칸(= `Cause of death` = 3, 기간 무관)과는 4년 뒤 SCD 3명(P0043, P0346, P0539)만 다름 |
| 팀 공유 라벨 | `results/final_holter_ecg/labels_445.csv` (`pid`, `SCD_4year_label`). 팀원이 이걸로 split을 다시 나눌 예정. 새 분할 파일이 오면 `results/final_holter_ecg/final_split.csv`를 바꾸고 노트북 1을 다시 실행 |
| 전처리 | Swee 2026 규칙 + **3유도(X·Y·Z)를 PCA로 합친 신호 1개** (Ramírez 2017). 30초 조각마다 특징 28개 |
| 정제 신호 저장 | PCA로 합친 신호를 **float16 `.npy`**, 파일 이름 = 환자 ID (`P0001.npy`) |
| 증강 | 정제된 신호의 **앞 15시간(`P0001-1`), 뒤 15시간(`P0001-2`)**. 445명 모두 기록이 20시간 이상이라 전원 2개씩 → 890개 |
| 학습 시 규칙 | 증강본은 원본 환자와 같은 split에 둔다 |

현재 split(팀 파일 기준): train 280 (SCD 29) · val 74 (SCD 9) · test 91 (SCD 12).

## 3. 노트북 (모두 `notebooks/`, git main에 있음)

| 노트북 | 하는 일 | 결과 |
|---|---|---|
| `final_holter_ecg_1_cohort.ipynb` | ①~⑤ Swee 코호트와 4년 정답 → ⑥ 팀 분할 파일에서 심장 기기 환자를 빼고 445명 표 저장 | `results/final_holter_ecg/cohort_final.csv` (445명, `pid`, `split`, `SCD_4year_label`, `outcome_code`) |
| `final_holter_ecg_3_preprocessing.ipynb` | `cohort_final.csv`의 445명을 처리. 원본 정제 → 특징 CSV + `.npy` 저장 → 같은 정제 신호에서 앞·뒤 15시간을 잘라 증강본 CSV + `.npy` | `/mnt/c/data/music/SCD_Holter_v2_pca/segment_features/*.csv`, `/mnt/c/data/music/SCD_Holter_v2_pca/clean_signal/*.npy`, 처리 기록 `results/preprocess_swee/record_log.csv` |
| `final_holter_ecg_2_augmentation.ipynb` | **선택.** 원파형(정제 전)을 WFDB로 잘라 저장. `.npy`와 특징은 노트북 3이 이미 만들기 때문에, 드라이브 `ECG_tem`에 원파형도 올려야 할 때만 필요 | `AUG_DIR`에 `P0001-1.hea/.dat` 등 890쌍, 약 57 GB |

## 4. 집에서 실행 순서

### 0단계: 준비
1. `git pull` (최신 커밋: `8936393` 이후)
2. 데이터가 있는지 확인:
   - `/mnt/c/data/music/subject-info.csv`, `/mnt/c/data/music/RECORDS`
   - `/mnt/c/data/music/Holter_ECG/*.hea, *.dat` (원본 Holter, PhysioNet MUSIC 1.0.1)
   - `results/final_holter_ecg/final_split.csv` (git에 있음)
3. 파이썬 환경: `.venv`에 `requirements.txt` 설치. neurokit2 0.2.13 (pandas 2.x) 기준
4. 디스크: C:에 약 40 GB 이상 여유 (`.npy` 약 35 GB + CSV)

### 1단계: 노트북 1 실행 (처음부터 끝까지)
- ④ 출력: `환자 목록이 Swee와 같은가: True`
- ⑥ 출력 확인: `분할 파일: 508 명 · 기기 이식 제외 63 명`, 정답이 바뀐 환자 P0043·P0346·P0539, 표 합계 **445 / SCD 50**
- `results/final_holter_ecg/cohort_final.csv`가 생겼는지 확인
- 노트북 끝 📝 메모는 사용자가 채운다 (부탁하면 도와줌)

### 2단계: 노트북 3 실행
1. ① 실행 → 출력 `Holter 기록 936개 중 처리 대상 445명`. Holter 파일이 없는 환자가 있으면 assert로 멈춤 → 원본 데이터부터 확인
2. ②③ 함수 정의 실행
3. ④ 시험 (P0307 1명): 확인할 것
   - `status = ok`, `n_aug = 2`
   - `.npy` 3개 출력: `P0307.npy` 약 24시간, `P0307-1.npy`·`P0307-2.npy` 각 15.00시간, dtype `float16`
   - `seconds`(1명 처리 시간)와 `peak_mem_gb`(메모리) → 445명 예상 시간 = seconds × 445 ÷ `N_JOBS`. 증강본 특징까지 계산해서 이전보다 2배 넘게 걸림
   - RAM 16 GB 안에 들어가게 ①의 `N_JOBS`를 조정 (기본 4)
4. ④-2 그림: 합친 신호의 R파가 위로 향하는지 눈으로 확인
5. ⑤ 전체 실행. 중간에 끊겨도 다시 실행하면 `record_log.csv`에 없는 환자부터 이어서 함. 오류 난 환자는 기록에 남아 다시 하지 않음 (다시 하려면 그 줄을 지우고 실행)
6. ⑥ 결과 확인:
   - 처리 상태 전원 `ok`인지
   - `.npy`와 특징 CSV가 **1,335개씩**(445 + 890), "짝이 안 맞는 이름" 목록이 비어 있는지
   - `.npy` 총용량 약 35 GB
   - 조각 3개 미만 환자 수 (지우지 않고 표시만 함. 학습에서 뺄지는 학습 때 사용자가 정함. Swee는 뺐음)

### 3단계 (선택): 노트북 2
- 원파형 WFDB가 필요할 때만. **주의: ①의 `AUG_DIR = /home/aiuser/data/music/Holter_ECG_aug`는 학교 컴퓨터 경로다.** 집 컴퓨터 사용자 이름이 다르면 폴더를 만들 수 없어 오류가 남 → 집 경로로 바꿔야 함
- 예전에 730명 기준으로 시험한 파일(`P0906-1`, `-2`)이 남아 있으면 ⑧의 "계획에 없는 증강 파일"에 나옴

## 5. 구글 드라이브 업로드

팀 구글 드라이브의 `DATA` 폴더 (팀원이 공유한 폴더. 사용자 드라이브의 "공유 문서함"에서 찾을 수 있음). 안에 `clincal`, `ECG_num`, `ECG_tem/ECG_tem_clean`, `final_split.csv`가 있다.

| 드라이브 폴더 | 올릴 것 (집 컴퓨터 위치, 윈도우 경로) |
|---|---|
| `ECG_num` | `C:\data\music\SCD_Holter_v2_pca\segment_features\` 안의 CSV 1,335개 + `record_log.csv` |
| `ECG_tem/ECG_tem_clean` | `C:\data\music\SCD_Holter_v2_pca\clean_signal\` 안의 `.npy` 1,335개 |
| `ECG_tem` | 노트북 2를 돌렸을 때만 원파형 WFDB. **이 폴더의 용도("tem" 뜻)는 팀원에게 아직 확인 안 함** |
| `DATA` 바로 아래 | `labels_445.csv`, README (아래 내용) |

- 파일이 많고 크므로 **Google Drive 데스크톱 앱**으로 폴더째 끌어 올리는 방법을 권함. Claude의 드라이브 커넥터로는 대용량 파일을 올리기 어렵다
- 올리기 전 개수 확인: 각 폴더 1,335개

### 드라이브 README에 들어갈 내용 (사용자가 요청하면 Claude가 작성)
- 환자 445명 기준, 정답 정의(`SCD_4year_label`, 4년 = 1,460일), 팀 `SCD` 칸과 다른 3명
- 파일 이름 규칙: `P0001` 원본, `P0001-1` 앞 15시간, `P0001-2` 뒤 15시간. 하이픈 앞이 환자 ID
- `.npy`: 1차원 float16, 단위 mV, 200 Hz, PCA로 합친 신호, **기록 시작 30초 뒤부터** (앞 30초 보정 신호 제거)
- 특징 CSV: 30초 조각 1줄, 열 = `patient_id`, `window_idx`, `start_idx` + 특징 28개 (목록은 노트북 3 ①의 `FEATURES`)
- 증강본은 원본 환자와 같은 split에 넣을 것

## 6. 아직 정하지 않은 것 / 확인할 것

| 항목 | 내용 |
|---|---|
| 팀 분할 파일에서 빠진 198명 | Swee 조건을 만족하는데 팀 파일에 없는 환자 198명(4년 SCD 21명 포함). 기준을 팀원에게 확인 중. 사용자는 445명으로 가기로 함 |
| 새 split | 팀원이 `labels_445.csv`로 다시 나누면 `final_split.csv` 교체 → 노트북 1 ⑥ 다시 실행 (환자는 그대로라 전처리는 다시 안 해도 됨) |
| `ECG_tem` 용도 | 원파형인지 팀원에게 확인 필요 |
| 기록 경로 | 노트북 3 `LOG_PATH`가 아직 `results/preprocess_swee/record_log.csv` (다른 결과는 `final_holter_ecg/`). 동작에는 문제 없음 |
| 심장 기기 제외 근거 | Swee 논문은 이유를 안 씀 (부록 E 미확인) |

## 7. 사용자에게 물어서 받을 것

이 문서는 공개 저장소에 있어서 개인정보와 공유 링크를 넣지 않았다. 필요하면 사용자에게 직접 받는다.

| 필요한 것 | 언제 |
|---|---|
| 팀 구글 드라이브 `DATA` 폴더 위치 (링크 또는 경로) | 업로드 안내할 때 |
| 집 컴퓨터의 사용자 이름·경로 (`/home/<이름>/...`), 원본 Holter·`subject-info.csv` 위치 | 0단계 준비, 노트북 2 `AUG_DIR` 바꿀 때 |
| 코호트·증강 정리 페이지 링크 | 정리 페이지를 고쳐야 할 때 |
| 팀원 연락처나 팀원이 정한 기준 | 새 split, `ECG_tem` 용도, 빠진 198명 확인할 때 |

## 8. 참고 자료

- Swee 2026 논문: https://www.medrxiv.org/content/10.64898/2026.04.21.26351421v2.full · 코드: https://github.com/stswee/multimodal_aecg_clinicalfeatures
