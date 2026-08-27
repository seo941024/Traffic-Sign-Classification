# HANDOFF — 다른 컴퓨터에서 바로 이어서 작업하기

**작성일**: 2026-08-27
**저장소**: https://github.com/seo941024/Traffic-Sign-Classification
**프로젝트**: 교통·도로 표지판 이미지 분류 — ANN / DNN / CNN 구조 비교

이 문서 하나로 새 PC에서 환경 구축부터 학습까지 재현할 수 있게 하는 것이 목적이다.

---

## 0. 현재 상태 요약

| 항목 | 상태 |
|---|---|
| 코드 전체 | ✅ 작성 완료 |
| **파이프라인 검증** | ✅ **합성 데이터로 전 과정 통과** (`99_smoke_test.py`) |
| PyTorch ↔ Keras 파라미터 일치 | ✅ 세 모델 모두 0 차이로 검증됨 |
| GPU 학습 코드 | ✅ CUDA 자동 감지 확인 (`torch.cuda.is_available()`) — 실제 학습 PC 사양과 무관하게 동작 |
| 원본 데이터 crop | ⏳ 미실행 (tar 압축 해제 필요) |
| 본 실험 학습 | ⏳ 미실행 |

**실제 AI-Hub 데이터로는 아직 아무것도 돌리지 않았다.** 합성 데이터로 "코드가 끝까지 도는지"만 확인했다.

---

## 1. 검증된 환경 (이 버전 조합으로 실제 동작 확인함)

### 1-1. 결정적 제약 — Python 버전

| Python | PyTorch | TensorFlow | 판정 |
|---|---|---|---|
| **3.12.13** | ✅ | ✅ | **이걸 써라** |
| 3.14.3 | ✅ (2.13.0) | ❌ **cp314 휠 없음** | 쓰지 마라 |

> 원본 핸드오프 PDF 33절이 *"Python 3.14 환경에서 일부 딥러닝 패키지가 호환되지 않을 가능성"*을
> 경고했는데, **실측으로 정확히 확인됐다.** TensorFlow는 Python 3.14용 휠을 배포하지 않는다.
>
> ```
> × No solution found when resolving dependencies:
>   ╰─▶ Because all versions of tensorflow have no wheels with a matching
>       Python ABI tag (e.g., `cp314`)
> ```

### 1-2. 검증된 정확한 버전

```
Python        3.12.13
torch         2.5.1+cu121      ← CUDA 12.1 빌드 (CPU 빌드 아님!)
torchvision   0.20.1+cu121
CUDA (torch)  12.1
cuDNN         90100
tensorflow    2.21.0
keras         3.15.1
numpy         2.5.2
pillow        12.3.0
matplotlib    3.11.1
scikit-learn  1.9.0
pandas        3.0.5
tqdm          4.70.0
```

전체 58개 패키지 목록: [`requirements.lock.txt`](requirements.lock.txt)

### 1-3. 하드웨어 관련 참고

**학습은 이 문서를 만든 PC가 아니라 별도 PC에서 진행한다. 코드는 특정 GPU를 가정하지 않는다.**

`get_device()`(`src/torch_impl/train.py`)가 `torch.cuda.is_available()`로 자동 감지하므로
어떤 NVIDIA GPU든 그대로 잡힌다. 위 1-2절 버전 검증에 쓴 장비는 CUDA 12.1 드라이버가 깔린
NVIDIA GPU였다는 것만 의미가 있고, 모델명 자체는 재현과 무관하다.

실제 학습 PC에서 확인할 것은 딱 하나:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`True`가 나오면 된다. `False`면 GPU 드라이버/CUDA 툴킷 버전을 그 PC에서 확인해라
(2-3절의 `+cu121` 인덱스 설치 여부부터 점검).

---

## 2. 새 PC 환경 구축

### 2-1. uv 사용 (권장 — 이 프로젝트가 실제로 쓴 방법)

`uv`는 Python 자체도 설치해주므로 Python을 따로 깔 필요가 없다.

```bash
uv python install 3.12
```

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv -r requirements.lock.txt
```

> ⚠️ **`requirements.lock.txt`에는 `torch==2.5.1+cu121`이 들어 있다.**
> 기본 PyPI에는 `+cu121` 빌드가 없으므로 아래처럼 PyTorch 인덱스를 함께 지정해야 한다.

```bash
uv pip install --python .venv torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2-2. 일반 venv 사용

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install tensorflow==2.21.0 numpy==2.5.2 pillow==12.3.0 matplotlib==3.11.1 scikit-learn==1.9.0 pandas==3.0.5 tqdm==4.70.0
```

### 2-3. ⚠️ GPU 관련 함정 두 개

**(1) 그냥 `pip install torch` 하면 CPU 전용이 깔린다**

실제로 이 프로젝트에서 처음에 `torch==2.13.0+cpu`가 설치돼서 `cuda available: False`가 나왔다.
반드시 `--index-url https://download.pytorch.org/whl/cu121`을 붙여라.

확인 방법:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`2.5.1+cu121 True`가 나와야 한다. `+cpu`가 보이면 잘못 깔린 것이다.

**(2) TensorFlow는 Windows 네이티브에서 GPU를 못 쓴다**

TF 2.11부터 Windows 네이티브 GPU 지원이 중단됐다. 실제로 이런 경고가 뜬다:

```
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows
for TensorFlow >= 2.11. Please use WSL2 or the TensorFlow-DirectML plugin.
```

**대응**: 이 프로젝트에서 Keras는 **구조 비교·교차 검증용**이므로 CPU로 충분하다.
**GPU가 필요한 본 학습은 PyTorch로 한다.** WSL2를 쓸 필요 없다.

단, 학습 시간을 비교할 때 **PyTorch(GPU) vs Keras(CPU)를 직접 비교하면 안 된다.**
프레임워크 차이가 아니라 하드웨어 차이다. 같은 프레임워크 안에서 세 모델을 비교해라.

### 2-4. VS Code 설정

`Ctrl+Shift+P` → `Python: Select Interpreter` → `.venv\Scripts\python.exe` 선택.

> 이전에 uv의 Python 3.14가 선택돼 있었을 수 있다. TensorFlow가 import 안 되면 이걸 먼저 확인해라.

---

## 3. 환경 검증 (구축 직후 반드시)

```bash
python scripts/00_check_env.py
```

확인할 것: `CUDA 사용 가능: True`, `클래스 수: 6`

```bash
python scripts/99_smoke_test.py
```

**합성 데이터로 전 파이프라인을 돌린다. 원본 데이터가 없어도 실행된다.**
마지막에 `전체 통과`가 나오면 환경이 정상이다.

이 테스트가 확인하는 것:
- annotation 파서가 **뒤집힌 box**를 정규화하는가
- 그룹 분할이 **누수 없이** 되는가 (`확인: 모든 group_id 가 한 split 에만 속한다`)
- PyTorch / Keras 학습 루프가 도는가
- 평가 지표·그림 생성이 되는가

```bash
python scripts/90_model_report.py
```

세 모델 구조와 파라미터를 출력하고, **PyTorch ↔ Keras 파라미터 수가 일치하는지 교차 검증**한다.
`일치`가 3개 나와야 한다.

---

## 4. 데이터 준비

### 4-1. 🔴 다운로드 상태 — 문제 있음

| 세트 | 라벨 | 원천 | 교집합 | 사용 |
|---|---|---|---|---|
| **c (수도권 1280×720)** | 30,000 | 30,000 | **30,000** | ✅ |
| d (수도권 1920×1080) | 15,000 | 없음 | — | ❌ |
| m (광역시 1280×720 라벨) | 25,997 | 15,000 (**1920×1080**) | **0** | ❌ |

**m 세트는 라벨과 원천의 해상도가 달라 파일명이 하나도 겹치지 않는다.**

- 라벨 범위 `i0437256` ~ `i0555071`
- 원천 범위 `i0036120` ~ `i0664417`

**추가 다운로드가 필요한 것** (선택 — c 세트만으로 1차 실험 충분):
- `[원천]m_train_1280_720_daylight_1.tar` ← m 라벨과 짝
- `[원천]d_train_1920_1080_daylight_1.tar` ← d 라벨과 짝

> **규칙**: 라벨과 원천은 `{접두사}_train_{해상도}_daylight_{번호}`가 **완전히 같아야** 짝이 맞는다.

### 4-2. tar 파일 위치

```
~/Downloads/
├── 신호등-도로표지판 인지 영상(수도권)/Training/
│   ├── [라벨]c_train_1280_720_daylight_1.tar     38 MB
│   ├── [라벨]d_train_1920_1080_daylight_1.tar    20 MB
│   └── [원천]c_train_1280_720_daylight_1.tar   6,147 MB   ← 이게 핵심
└── 신호등-도로표지판 인지 영상(광역시, 고속도로, 국도 등)/Training/
    ├── [라벨]m_train_1280_720_daylight_1.tar     31 MB
    └── [원천]m_train_1920_1080_daylight_1.tar  9,436 MB   ← 짝 안 맞음
```

새 PC로 옮길 때 **최소한 c 세트 2개(라벨 38MB + 원천 6.1GB)는 반드시 가져가야 한다.**

### 4-3. 경로 설정

`config.py`에서 새 PC의 경로에 맞게 수정:

```python
DOWNLOAD_ROOT = Path.home() / "Downloads"       # tar 위치
AIHUB_ROOT = Path("D:/AIHub_TrafficSign")       # 압축 해제 및 crop 작업 공간
```

**필요 디스크**: 원천 압축 해제 약 6GB + crop 결과 약 200MB → **여유 10GB 권장**

---

## 5. 실행 순서

```bash
python scripts/01_extract_tars.py --verify-only
```
라벨↔원천 매칭 확인만 한다. **6GB 풀기 전에 반드시 먼저.**

```bash
python scripts/01_extract_tars.py
```
D: 드라이브에 압축 해제. 6GB라 시간이 걸린다.

```bash
python scripts/02_build_crops.py --limit 500
```
**먼저 소규모로.** annotation 500개만 처리해서 crop이 제대로 되는지 본다.

```bash
python scripts/03_verify_crops.py
```
🔴 **절대 건너뛰지 마라.** `results/figures/crop_verification.png`를 **눈으로** 확인한다.

> bbox를 잘못 읽어도 코드는 **에러 없이 잘 돈다.** 엉뚱한 영역을 자르고 파일도 정상 저장되고
> 학습도 돌아간다. 정확도만 이상하게 낮게 나온다. 그때 원인 찾으려면 훨씬 오래 걸린다.
>
> 확인할 것: 표지판이 중앙에 있는가 / 테두리가 잘렸는가 / **사람이 봐서 클래스를 구분할 수 있는가**

```bash
python scripts/02_build_crops.py
```
검증 통과 후 전체 실행.

```bash
python scripts/04_build_manifest.py
```
train/val/test 분할을 CSV로 확정. 출력에서 확인:
- 클래스별 분포가 균형잡혔는가
- `확인: 모든 group_id 가 한 split 에만 속한다 (누수 없음)`

```bash
python scripts/05_run_experiment.py --framework torch --epochs 3
```
**먼저 3 epoch으로 빠르게** 전체가 도는지 확인.

```bash
python scripts/05_run_experiment.py --framework both
```
본 실험 (30 epoch, early stopping).

```bash
python scripts/06_make_figures.py
```
학습 곡선 · Confusion Matrix · 파라미터 대비 정확도 그림 생성.

---

## 6. 검증된 모델 구조 (실제 실행 출력)

입력 `32×32×3` → Flatten `3,072`

| Model | Params | Size(MB) | 최소 대비 | 최대 파라미터 층 |
|---|---|---|---|---|
| ann | 788,230 | 3.01 | 1.27x | `hidden.0` 786,688 |
| dnn | 3,836,550 | 14.64 | 6.19x | `hidden.0` 3,146,752 |
| **cnn** | **619,334** | **2.36** | **1.00x** | `head.0` 524,544 |

**PyTorch ↔ Keras 교차 검증: 세 모델 모두 차이 0 (일치)** ✅

CNN 내부 분해:
```
Conv  계열    93,248   (15.1%)
Dense 계열   526,086   (84.9%)
```

→ *"CNN은 파라미터가 적다"*보다 ***"CNN은 Dense에 넣기 전에 Pooling으로 입력을 줄여놨다"***가 정확한 설명.

---

## 7. 클래스 체계 (원본 PDF에서 변경됨)

🔴 **AI-Hub에는 `stop`, `speed_30` 같은 의미 라벨이 없다.**
표지판의 **생김새**만 라벨링되어 있어 `shape × color` 조합을 클래스로 쓴다.

| 클래스 | 표지판 유형 | ≥24px 장수 |
|---|---|---|
| `circle_red` | 규제표지 (속도제한·진입금지) | 8,153 |
| `triangle_yellow` | 주의표지 | 3,564 |
| `circle_blue` | 지시표지 | 2,378 |
| `triangle_red` | 규제표지(삼각) | 1,254 |
| `triangle_blue` | 지시표지(삼각) | 1,237 |
| `rectangle_blue` | 안내표지 | 872 |

**이게 오히려 유리하다**: circle/triangle/rectangle 구분은 **윤곽선 판단**이고,
그건 Convolution 얕은 층이 학습하는 것 그 자체다.

→ **핵심 논증**: 오류를 "형태를 틀림" vs "색을 틀림"으로 분해한다.
ANN이 색은 맞히는데 형태에서 무너지면 → *"공간 정보를 못 쓴다"*에 수치 증거가 붙는다.
(`src/analysis/metrics.py`의 `shape_vs_color_confusion`, 실행 시 자동 출력)

---

## 8. Annotation 형식과 함정

```json
{
  "annotation": [
    {"box":[708,177,719,188], "class":"traffic_sign", "shape":"circle", "color":"blue"}
  ],
  "image": {"filename":"s01027933.jpg", "imsize":[1280,720]}
}
```

`box` = `[x1, y1, x2, y2]` **픽셀 절대좌표** (정규화 아님)

### 🔴 조용히 망가지는 함정들 (전부 처리 완료)

1. **좌표가 뒤집힌 박스 존재** — `[315,308,287,335]`에서 x2 < x1
   → `min`/`max` 정규화 안 하면 빈 crop이 생기는데 **예외는 안 난다**
2. key 순서가 객체마다 다름 → 순서 의존 파싱 금지
3. `annotation`이 빈 배열인 파일 다수
4. `type` 키가 아예 없는 객체 많음 (circle_red 중 약 20%) → **분류에 쓸 수 없음**
5. `traffic_light`의 `type`은 `car`/`bus`/`bicycle` → `class`로 먼저 걸러야 함
6. `text` 타입 불일치 — `"0"`(문자열) vs `0`(정수)

전부 [`src/data/aihub.py`](src/data/aihub.py)에서 처리했고 `99_smoke_test.py`가 검증한다.

---

## 9. 이번 작업에서 발견하고 고친 버그

**스모크 테스트가 실제 데이터 누수 버그를 잡았다.**

- **증상**: `경고: 같은 group_id 가 여러 split 에 걸쳐 있다 (10개)`
- **원인**: 원본 사진 하나에 `circle_red`와 `triangle_yellow`가 같이 있으면 두 crop이
  **같은 group_id**를 갖는데, 분할이 **클래스별로 독립 실행**돼서 같은 사진이 train/test로 갈라졌다.
  막으려던 누수가 그대로 발생.
- **수정**: `split_by_group`을 **그룹 단위 전역 배정 + 탐욕적 클래스 균형**으로 재작성.
  큰 그룹부터 배정하며 목표 분포에서 가장 적게 벗어나는 split을 고른다.
- **결과**: 6클래스 × 24장 테스트에서 **16/4/4 완벽 균형 + 누수 0**

---

## 10. 학습 시 반드시 지킬 것

### 10-1. softmax를 넣지 않았다 (logits 출력)

| 프레임워크 | 반드시 |
|---|---|
| PyTorch | `nn.CrossEntropyLoss()` |
| Keras | `SparseCategoricalCrossentropy(**from_logits=True**)` |

**`from_logits`를 빠뜨리면 loss가 조용히 잘못 계산된다. 에러는 안 난다.**

### 10-2. seed는 모델 생성 직전마다

```python
for name in ("ann", "dnn", "cnn"):
    set_seed(config.SEED)        # ← 루프 안. 밖이 아니다
    model = build_model(name)
```

루프 밖에서 한 번만 하면 ANN 학습이 난수를 소모해 DNN/CNN이 다른 초기화를 받는다.

### 10-3. config.py의 학습 조건은 모델별로 절대 바꾸지 않는다

바꾸는 순간 "구조 때문에 생긴 차이"라고 말할 수 없게 된다.

```python
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
OPTIMIZER = "adam"
SEED = 42
EARLY_STOPPING_PATIENCE = 7    # 세 모델 동일
```

> **Early stopping을 쓰는 이유**: 고정 epoch이면 DNN은 과적합 상태로, CNN은 덜 학습된 상태로
> 비교하게 된다. 그건 구조 차이가 아니라 멈추는 시점 차이다.
> **patience 값이 동일하다는 것이 "동일 조건"의 진짜 의미다.**

---

## 11. 평가 시 볼 것

**Baseline 없이는 정확도를 해석할 수 없다** (자동 출력됨):
- 무작위 = 1/6 = **16.7%**
- 최빈 클래스 → **이걸 못 넘으면 모델이 무의미하다**

그 외: Train/Val Loss · Accuracy · **Macro F1** · Confusion Matrix ·
파라미터 수 · 학습 시간 · 추론 시간 · 모델 크기 · **형태/색상 오류 분해**

---

## 12. 남은 작업

- [ ] `01_extract_tars.py`로 c 세트 압축 해제
- [ ] `02_build_crops.py --limit 500` → `03_verify_crops.py` **눈으로 확인**
- [ ] crop 전체 생성 → `04_build_manifest.py`
- [ ] `GROUP_ID_BUCKET` 조정 — **휴리스틱이라 검증 필요**
      (AI-Hub ID 부여 규칙을 공식 문서로 확인한 게 아니다. 클래스별 분포 보고 조정)
- [ ] `05_run_experiment.py --framework torch --epochs 3` 빠른 검증
- [ ] 본 실험 → `06_make_figures.py`
- [ ] (선택) 파라미터 수를 맞춘 대조 실험 — `docs/HANDOFF_REVISED.md` B-7절
- [ ] (선택) traffic_light 39,888개로 클래스 4개 추가 — B-10절
- [ ] 포트폴리오 문서 작성

---

## 13. 관련 문서

| 문서 | 내용 |
|---|---|
| [README.md](README.md) | 프로젝트 개요·구조·설계 근거 |
| [docs/DATA_FINDINGS.md](docs/DATA_FINDINGS.md) | AI-Hub 데이터 실측 (형식·함정·통계 전체) |
| [docs/HANDOFF_REVISED.md](docs/HANDOFF_REVISED.md) | 원본 PDF의 수정·보완본 (12개 보완 항목) |
| [requirements.lock.txt](requirements.lock.txt) | 검증된 전체 패키지 버전 58개 |

---

## 14. 새 PC 체크리스트

```
[ ] git clone
[ ] uv python install 3.12
[ ] uv venv --python 3.12 .venv
[ ] torch를 cu121 인덱스로 설치 (+cpu 아닌지 확인!)
[ ] 나머지 패키지 설치
[ ] VS Code 인터프리터를 .venv로 지정
[ ] python scripts/00_check_env.py   → CUDA 사용 가능: True
[ ] python scripts/99_smoke_test.py  → 전체 통과
[ ] python scripts/90_model_report.py → 교차 검증 일치 3개
[ ] c 세트 tar 2개 복사 (라벨 38MB + 원천 6.1GB)
[ ] config.py의 AIHUB_ROOT 경로 수정
[ ] D: 드라이브 여유 10GB 확인
```
