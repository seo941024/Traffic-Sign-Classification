# 교통·도로 표지판 이미지 분류 — ANN / DNN / CNN 구조 비교

같은 표지판 데이터를 ANN, DNN, CNN으로 각각 학습시켜
**신경망 구조가 이미지 분류에 어떤 영향을 주는지** 분석하는 프로젝트.
PyTorch와 Keras 양쪽으로 구현해 교차 검증한다.

목표는 높은 정확도가 아니다. 다음 질문에 코드와 Tensor shape로 답하는 것이 목표다.

- Flatten을 하면 무엇을 잃는가?
- Convolution은 실제로 무엇을 계산하는가? Feature Map은 무엇인가?
- Pooling은 왜 쓰는가?
- 파라미터 수는 왜 모델마다 다르게 증가하는가?
- 같은 데이터에서 모델별 결과가 왜 달라졌는가?

---

## ⚠️ 시작 전 반드시 읽을 것

### 1. 이 PC에 Python이 설치되어 있지 않다

`python`은 Windows Store 스텁만 잡힌다. **이 저장소의 코드는 아직 한 번도 실행되지 않았다.**
문법 오류가 남아 있을 수 있으니 아래 순서로 확인해라.

Python **3.10 ~ 3.12** 설치 후:

```bash
pip install -r requirements.txt
```

GPU를 쓰려면 PyTorch는 CUDA 빌드로 별도 설치해야 한다:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 다운로드한 데이터 중 하나가 짝이 안 맞는다

| 세트 | 라벨 | 원천 | 교집합 |
|---|---|---|---|
| **c (수도권)** | 30,000 | 30,000 | **30,000** ✅ |
| d (수도권) | 15,000 | 없음 | ❌ |
| m (광역시) | 25,997 (1280×720) | 15,000 (**1920×1080**) | **0** ❌ |

**m 세트는 라벨과 원천의 해상도가 달라 파일명이 하나도 겹치지 않는다.**
쓰려면 `[원천]m_train_1280_720_daylight_1.tar`를 추가로 받아야 한다.

**다만 c 세트(6GB)만으로 1차 실험은 충분하다.**

자세한 내용: [docs/DATA_FINDINGS.md](docs/DATA_FINDINGS.md)

---

## 실행 순서

```bash
python scripts/00_check_env.py
```
GPU/CUDA 인식과 경로 확인. **제일 먼저.**

```bash
python scripts/90_model_report.py
```
**데이터 없이 지금 실행 가능.** 세 모델의 층별 shape 변화와 파라미터 분포를 출력하고,
PyTorch/Keras의 파라미터 수가 일치하는지 교차 검증한다.

```bash
python scripts/01_extract_tars.py --verify-only   # 매칭 확인 (권장: 먼저)
python scripts/01_extract_tars.py                 # D: 드라이브에 압축 해제
```

```bash
python scripts/02_build_crops.py --limit 500      # 소규모 검증 먼저
python scripts/02_build_crops.py                  # 전체
```

```bash
python scripts/03_verify_crops.py
```
**건너뛰지 마라.** bbox를 잘못 읽어도 코드는 에러 없이 돈다.
생성된 시트를 열어 표지판이 제대로 잘렸는지 **눈으로** 확인한다.

```bash
python scripts/04_build_manifest.py
```

```bash
python scripts/05_run_experiment.py --framework torch --epochs 3   # 빠른 검증
python scripts/05_run_experiment.py --framework both              # 본 실험
```

```bash
python scripts/06_make_figures.py
```

---

## 구조

```
config.py                     실험 조건의 단일 출처 — 모델별로 절대 다르게 두지 않는다
docs/
  DATA_FINDINGS.md            AI-Hub 데이터 실측 결과
  HANDOFF_REVISED.md          원본 핸드오프 PDF의 수정·보완본
scripts/
  00_check_env.py             GPU/CUDA/경로 점검
  01_extract_tars.py          tar 검증 + 압축 해제
  02_build_crops.py           bbox → 표지판 crop
  03_verify_crops.py          crop 시각 검증
  04_build_manifest.py        분할 확정 → CSV
  05_run_experiment.py        ANN/DNN/CNN 학습 및 비교
  06_make_figures.py          학습곡선 · Confusion Matrix · 파라미터 대비 정확도
  90_model_report.py          구조 리포트 (학습·데이터 불필요)
  91_inspect_annotation.py    annotation 구조 조사 도구
src/
  data/
    aihub.py                  AI-Hub annotation 파서
    manifest.py               train/val/test 분할을 CSV로 못박는다
    split.py                  그룹 단위 분할 (데이터 누수 방지)
  torch_impl/  models.py · dataset.py · train.py
  keras_impl/  models.py · dataset.py · train.py
  analysis/
    model_stats.py            파라미터 수 · Tensor shape 추적
    metrics.py                Confusion Matrix · Baseline · 형태/색상 오류 분해
    compare.py                세 모델 비교표
```

---

## 클래스 체계 — 원본 계획에서 변경됨

**AI-Hub에는 `stop`, `speed_30` 같은 의미 라벨이 없다.**
표지판의 **생김새**만 라벨링되어 있어서 `shape × color` 조합을 클래스로 쓴다.

| 클래스 | 표지판 유형 | ≥24px 장수 |
|---|---|---|
| `circle_red` | 규제표지 (속도제한·진입금지) | 8,153 |
| `triangle_yellow` | 주의표지 | 3,564 |
| `circle_blue` | 지시표지 | 2,378 |
| `triangle_red` | 규제표지(삼각) | 1,254 |
| `triangle_blue` | 지시표지(삼각) | 1,237 |
| `rectangle_blue` | 안내표지 | 872 |

**이 변경이 오히려 유리하다**: `circle` vs `triangle` vs `rectangle` 구분은
**윤곽선 판단**이고, 그건 Convolution의 얕은 층이 학습하는 것 그 자체다.
Flatten된 ANN/DNN은 픽셀 인접 관계를 모르므로 형태 판단에 근본적으로 불리하다.

→ **이 프로젝트의 핵심 논증**: 오류를 "형태를 틀림" vs "색을 틀림"으로 분해한다.
ANN이 색은 맞히는데 형태에서 무너진다면, "공간 정보를 못 쓴다"는 주장에 수치 증거가 붙는다.
(`src/analysis/metrics.py`의 `shape_vs_color_confusion`)

---

## 설계 결정과 근거

### 입력 32×32×3 (Flatten 3,072)

원본 계획은 64×64였으나, 실제 bbox **짧은 변 중앙값이 17px**이다.
17px를 64×64로 키우는 건 없는 정보를 늘려서 만드는 것이라 얻는 게 없다.
(GTSRB 등 공개 표지판 데이터셋도 32×32가 표준)

`MIN_BBOX_SIZE = 24` — 이보다 작으면 사람도 구분 못 한다. 필터가 **필수**다.

### manifest CSV로 "동일 조건"을 강제

PyTorch의 `ImageFolder`와 Keras의 `image_dataset_from_directory`는
각자 폴더를 훑어 **자기 나름대로 클래스 순서를 정한다.**
양쪽에서 따로 랜덤 분할하면 seed를 맞춰도 같다는 보장이 없다.

분할을 **딱 한 번만** 하고 CSV로 고정한 뒤 두 프레임워크는 읽기만 한다.
→ "같은 데이터를 봤다"가 **가정이 아니라 사실**이 된다.

### 그룹 단위 분할 (데이터 누수 방지)

**사진 한 장에서 표지판을 여러 개 자른다** (샘플 하나에 6개가 있었다).
그것들은 배경·조명·촬영조건이 완전히 같아서, train과 test에 나뉘면
모델이 표지판이 아니라 **촬영 조건**을 외워서 맞힐 수 있다.

crop 파일명을 `{원본stem}__{순번}.jpg`로 지어 원본 단위로 묶고,
여기에 연속 촬영 프레임 묶기(`GROUP_ID_BUCKET`)를 추가로 적용한다.

### Early Stopping (고정 epoch 대신)

DNN은 파라미터가 많아 빨리 과적합하고 CNN은 더 오래 개선된다.
30 epoch 고정이면 **DNN은 과적합 상태로, CNN은 덜 학습된 상태로** 비교하게 된다.
그건 구조의 차이가 아니라 멈추는 시점의 차이다.

→ 세 모델에 **동일한 patience**로 early stopping을 적용한다.
patience 값이 동일하다는 것이 "동일 조건"의 진짜 의미다.

### Keras와 PyTorch를 둘 다 쓰는 이유

같은 구조를 두 프레임워크로 짜면 **파라미터 수가 정확히 일치**해야 한다.
일치한다는 건 파라미터 수를 프레임워크가 아니라 **구조가 결정한다**는 확인이다.
`90_model_report.py`가 이 교차 검증을 자동으로 한다.

| | PyTorch | Keras |
|---|---|---|
| shape 확인 | forward hook으로 **실행 중 실제 shape** 추적 | 구성 시점에 확정 — `summary()` 기본 제공 |
| 학습 루프 | 직접 구현 (무엇을 하는지 다 보인다) | `model.fit()` + 콜백 |
| 축 순서 | channels-first `(3,32,32)` | channels-last `(32,32,3)` |
| 데이터 로딩 | `DataLoader` + `num_workers` | `tf.data` + `prefetch` |

---

## 세 모델의 구조

| | ANN | DNN | CNN |
|---|---|---|---|
| Flatten 위치 | 맨 앞 | 맨 앞 | **맨 뒤** |
| Hidden Dense | 1층 (256) | 4층 (1024→512→256→128) | 1층 (256) |
| Conv Block | 없음 | 없음 | 3개 (32→64→128) |
| 공간 정보 활용 | ✗ | ✗ | ○ |

**핵심 대비**: ANN과 DNN은 맨 앞에서 Flatten하고, CNN은 맨 뒤에서 한다.
DNN이 ANN보다 깊지만 입력이 이미 Flatten되어 있다는 제약은 동일하다.
**깊이를 늘리는 것으로는 잃어버린 공간 정보를 되찾을 수 없다.**

### 파라미터 (클래스 6개 기준, 손계산 — 실행하여 확인 필요)

| 모델 | 파라미터 | 최대 파라미터 층 |
|---|---|---|
| ANN | 약 788,000 | `Linear(3072→256)` 786,688 (99.8%) |
| DNN | 약 3,836,000 | `Linear(3072→1024)` 3,146,752 (82%) |
| **CNN** | **약 619,000** | `Linear(2048→256)` 524,544 (85%) |

CNN의 Conv 3개 합계는 **93,248** (전체의 15%)뿐이다.
Conv는 커널을 재사용하므로(weight sharing) 해상도가 커져도 파라미터가 안 늘고,
Dense는 입력이 커지면 그대로 비례해 늘어난다.

→ *"CNN은 파라미터가 적다"*보다
***"CNN은 Dense에 넣기 전에 Pooling으로 입력을 줄여놨다"***가 정확한 설명이다.

---

## 학습 시 반드시 지킬 것

모델 마지막 층에 **softmax를 넣지 않았다** (logits 출력):

- PyTorch: `nn.CrossEntropyLoss()` — 내부에서 log_softmax
- Keras: `SparseCategoricalCrossentropy(**from_logits=True**)`

**`from_logits`를 빠뜨리면 loss가 조용히 잘못 계산된다. 에러는 안 난다.**

seed는 **모델을 만들기 직전마다** 다시 고정한다. 루프 밖에서 한 번만 하면
ANN 학습이 난수를 소모해 DNN/CNN이 다른 초기화를 받는다.

`config.py`의 학습 하이퍼파라미터는 **모델별로 절대 바꾸지 않는다.**

---

## 평가

Accuracy 하나만 쓰지 않는다.

**Baseline을 반드시 함께 본다** — 없으면 정확도를 해석할 수 없다:
- 무작위 = 1/6 = **16.7%**
- 최빈 클래스 = 가장 많은 클래스 비율 → **이걸 못 넘으면 모델이 무의미하다**

그 외: Train/Val Loss · Accuracy · **Macro F1**(클래스 불균형 대응) ·
Confusion Matrix · 파라미터 수 · 학습 시간 · 추론 시간 · 모델 크기 ·
**형태/색상 오류 분해**

---

## 문서

- [docs/DATA_FINDINGS.md](docs/DATA_FINDINGS.md) — AI-Hub 데이터 실측 결과 (형식·함정·통계)
- [docs/HANDOFF_REVISED.md](docs/HANDOFF_REVISED.md) — 원본 핸드오프 PDF의 수정·보완본
