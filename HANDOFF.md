# HANDOFF — 내일 이어서 작업하기

**저장소**: https://github.com/seo941024/Traffic-Sign-Classification
**프로젝트**: 교통 표지판 이미지 분류 — ANN / DNN / CNN 구조 비교 (Keras + PyTorch)

---

## 1. 지금 상태

코드는 다 짜여 있다. **학습은 아직 한 번도 안 돌렸다.**
학습은 학교 컴퓨터(RTX 3080 10GB)에서 할 예정.

### 파일 구성 (전부 11개)

```
config.py              모델 구조 설정 (두 프레임워크가 공유)
prepare_data.py        AI-Hub json → 클래스별 crop 폴더
train_keras.py         Keras 로 ANN/DNN/CNN 학습
train_torch.py         PyTorch 로 ANN/DNN/CNN 학습
requirements.txt

src/torch_impl/models.py    ANN, DNN, CNN (PyTorch)
src/keras_impl/models.py    ANN, DNN, CNN (Keras)
src/__init__.py
src/torch_impl/__init__.py
src/keras_impl/__init__.py
.gitignore
```

---

## 2. 내일 할 일 (순서대로)

### 2-1. 환경 세팅

```bash
git clone https://github.com/seo941024/Traffic-Sign-Classification.git
```

**Python 3.12 를 써라.** 3.13+ 는 TensorFlow 휠이 없어서 설치 자체가 안 된다.
(3.14 에서 실제로 확인함: `no wheels with a matching Python ABI tag (cp314)`)

torch 는 그냥 설치하면 **CPU 전용**이 깔린다. 반드시 인덱스를 지정해라:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install tensorflow pillow
```

설치 확인 — `True` 가 나와야 한다:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`+cpu` 라고 나오면 잘못 깔린 것이다. 지우고 위 인덱스로 다시 깔아라.

### 2-2. 데이터 압축 풀기

AI-Hub tar 2개가 필요하다 (라벨 + 원천). 아래 참고에서 **c 세트**만 쓴다.

압축을 풀면 이렇게 된다:
```
<어딘가>/annotations/c_train_1280_720_daylight_1/*.json   ← 라벨 30,000개
<어딘가>/raw/*.jpg                                        ← 원천 30,000장
```

### 2-3. Crop 만들기

```bash
python prepare_data.py --json-dir <json폴더> --image-dir <jpg폴더> --out crop
```

결과:
```
crop/
  circle_blue/     circle_red/       rectangle_blue/
  triangle_blue/   triangle_red/     triangle_yellow/
```

### 2-4. 🔴 crop 폴더를 눈으로 확인 (건너뛰지 마라)

**bbox 를 잘못 읽어도 에러가 안 난다.** 엉뚱한 영역을 자르고 파일도 정상 저장되고
학습도 돌아간다. 정확도만 이상하게 낮게 나온다. 그때 원인 찾으려면 훨씬 오래 걸린다.

폴더 열어서 확인할 것:
- 표지판이 이미지 중앙에 제대로 들어와 있는가
- 테두리가 잘리지 않았는가 (잘렸으면 `PADDING` 을 올려라)
- **사람이 봐서 클래스를 구분할 수 있는가** (뭉개졌으면 `MIN_SIZE` 를 올려라)
- `circle_red` 폴더에 정말 빨간 원형만 있는가

### 2-5. 학습

먼저 3 epoch 으로 빠르게 확인:

```bash
python train_keras.py --epochs 3
```

정상이면 본 실행:

```bash
python train_keras.py
```

```bash
python train_torch.py
```

---

## 3. 배치 크기와 epoch

3080 10GB 기준.

**batch 는 그 PC 에서 실제로 돌려보고 정한다.**
`train_keras.py` / `train_torch.py` 상단의 `BATCH_SIZE` 를 고치면 된다 (현재 32).

이론상으로는 이 모델들이 작아서(제일 큰 DNN 도 15MB, 입력 32×32) VRAM 여유가 많지만,
실제 속도는 데이터 로딩·드라이버·다른 프로세스 등에 따라 달라진다.
**과거에 그 PC 에서 잘 나왔던 값이 있으면 그걸 먼저 써라.**

**epoch 은 고정 숫자로 정하지 마라.** 코드에 early stopping 이 들어 있다
(val loss 가 5 epoch 연속 안 좋아지면 자동 중단). 기본 20 으로 두고
로그의 val loss 를 보면서 판단해라.

### ⚠️ GPU 가 느릴 때 — 공유 메모리 fallback

Windows CUDA 는 VRAM 이 넘치면 **에러를 안 내고 시스템 RAM 으로 넘긴다.**
PCIe 를 타므로 수십 배 느려지는데 OOM 이 안 떠서 원인을 못 찾고 며칠 날리기 쉽다.

**미리 꺼둬라:**
NVIDIA 제어판 → 3D 설정 관리 → **CUDA - Sysmem Fallback Policy** → **Prefer No System Fallback**

이러면 VRAM 초과 시 바로 OOM 에러가 떠서 즉시 알 수 있다. (드라이버 536.40+)

학습 중엔 작업관리자 → 성능 → GPU 에서 "전용 GPU 메모리" vs "공유 GPU 메모리" 를 보면
실시간으로 넘어가는지 확인된다.

---

## 4. 🔴 데이터 관련 — 반드시 알아야 할 것

### 4-1. 다운로드한 tar 중 하나는 짝이 안 맞는다

| 세트 | 라벨 | 원천 | 교집합 | 사용 |
|---|---|---|---|---|
| **c (수도권 1280×720)** | 30,000 | 30,000 | **30,000** | ✅ |
| d (수도권 1920×1080) | 15,000 | 없음 | — | ❌ |
| m (광역시) | 25,997 (**1280×720**) | 15,000 (**1920×1080**) | **0** | ❌ |

**m 세트는 라벨과 원천의 해상도가 달라서 파일명이 하나도 안 겹친다.**
- 라벨 범위 `i0437256` ~ `i0555071`
- 원천 범위 `i0036120` ~ `i0664417`

추가로 받으려면 `[원천]m_train_1280_720_daylight_1.tar` 가 필요하다.
**라벨과 원천은 `{접두사}_train_{해상도}_daylight_{번호}` 가 완전히 같아야 짝이 맞는다.**

**c 세트 하나(6GB)로 1차 실험은 충분하다.**

### 4-2. 클래스 — 원래 계획에서 바뀜

**AI-Hub 에는 `stop`, `speed_30` 같은 의미 라벨이 없다.**
표지판의 **생김새만** 라벨링되어 있다. `text` 필드가 있지만 값이 대부분 `0` 이라
속도 숫자로 쓸 수 없다.

그래서 `shape × color` 조합을 클래스로 쓴다:

| 클래스 | 표지판 유형 | 24px 이상 장수 |
|---|---|---|
| `circle_red` | 규제표지 (속도제한·진입금지) | 8,153 |
| `triangle_yellow` | 주의표지 | 3,564 |
| `circle_blue` | 지시표지 | 2,378 |
| `triangle_red` | 규제표지(삼각) | 1,254 |
| `triangle_blue` | 지시표지(삼각) | 1,237 |
| `rectangle_blue` | 안내표지 | 872 |

**이게 오히려 프로젝트에 유리하다.**
circle / triangle / rectangle 구분은 **윤곽선 판단**이고, 그건 Convolution 의
얕은 층이 학습하는 것 그 자체다. Flatten 된 ANN/DNN 은 픽셀 인접 관계를 모르므로
형태 판단에 근본적으로 불리하다.

→ 결과 분석할 때 **"형태를 틀렸나" vs "색을 틀렸나"** 로 나눠서 보면
ANN 이 색은 맞히는데 형태에서 무너지는 게 보일 것이다.
그게 "공간 정보를 못 쓴다"는 주장의 직접적인 증거가 된다.

### 4-3. bbox 가 매우 작다

`traffic_sign` bbox 짧은 변 분포 (c 세트, 54,500개):

| p25 | **p50** | p75 | p90 | p95 |
|---|---|---|---|---|
| 12px | **17px** | 26px | 39px | 48px |

**절반이 17px 이하다.** 주행 영상이라 표지판이 멀리 있다.

그래서 `MIN_SIZE = 24` 필터가 **필수**다. 없으면 8px 짜리 뭉개진 이미지가
학습 데이터의 절반을 차지해서, 모든 모델 성능이 같이 낮아지고
**구조 차이가 데이터 노이즈에 묻힌다.** 프로젝트 목적이 무너진다.

입력 크기를 32×32 로 잡은 것도 이 때문이다. 17px 짜리를 64×64 로 키우면
없는 정보를 늘려서 만드는 것이라 얻는 게 없다.
(GTSRB 등 공개 표지판 데이터셋도 32×32 가 표준)

### 4-4. Annotation 형식

```json
{
  "annotation": [
    {"box":[708,177,719,188], "class":"traffic_sign", "shape":"circle", "color":"blue"}
  ],
  "image": {"filename":"s01027933.jpg", "imsize":[1280,720]}
}
```

`box` = `[x1, y1, x2, y2]` **픽셀 절대좌표** (정규화 아님)
- 근거: `[708,177,719,188]` 을 `(x,y,w,h)` 로 읽으면 `x+w=1427` 로 폭 1280 을 넘는다

### 4-5. 🔴 조용히 망가지는 함정들 (`prepare_data.py` 에서 전부 처리함)

전부 실제 데이터에서 관찰된 것이고, **에러 없이 잘못된 결과를 만든다.**

1. **좌표가 뒤집힌 박스 존재** — `[315,308,287,335]` 에서 x2 < x1
   → `min`/`max` 정규화 안 하면 빈 crop 이 나오는데 **예외는 안 난다**
2. **key 순서가 객체마다 다르다** — `box` 먼저인 것도 `shape` 먼저인 것도 있음
   → 순서에 의존하는 파싱 금지
3. `annotation` 이 빈 배열인 파일이 많다 (표지판 없는 프레임)
4. **`type` 키가 아예 없는 객체가 많다** (circle_red 중 약 20%)
   → 그래서 `type` 을 분류에 안 쓰고 `shape`+`color` 를 쓴다
5. **`traffic_light` 의 `type` 은 `car`/`bus`/`bicycle`** — 표지판의 `type` 과 의미가 완전히 다름
   → `class` 로 먼저 걸러야 한다
6. `text` 타입 불일치 — `"0"`(문자열) vs `0`(정수)

### 4-6. 파일 섞기 (샘플링 편향 방지)

`prepare_data.py` 는 json 파일을 **seed 42 로 섞은 뒤** 처리한다.

안 섞으면 파일명이 촬영 순서(`s01000200`, `s01000201`...)라서:
- `circle_red`(27,958장) → 앞쪽 10% 구간에서 800장 다 채우고 멈춤
- `rectangle_blue`(2,904장) → 거의 전 구간을 훑어야 채워짐

결과적으로 **클래스마다 다른 주행 구간에서 뽑힌다.** 그러면 모델이 표지판이 아니라
그 구간의 배경/조명/날씨를 보고 클래스를 맞힐 수 있다. 정확도는 높게 나오는데
실제로는 아무것도 배운 게 아니다.

---

## 5. 모델 구조

입력 `32×32×3` → Flatten `3,072`

| Model | Params | 최대 파라미터 층 |
|---|---|---|
| ANN | 788,230 | `Linear(3072→256)` 786,688 |
| DNN | 3,836,550 | `Linear(3072→1024)` 3,146,752 |
| **CNN** | **619,334** | `Linear(2048→256)` 524,544 |

**PyTorch 와 Keras 파라미터 수가 정확히 일치하는 것을 확인했다** (세 모델 모두 차이 0).
같은 구조를 서로 다른 문법으로 쓴 것이라는 증거다.

CNN 내부:
```
Conv  계열    93,248   (15.1%)
Dense 계열   526,086   (84.9%)
```

→ *"CNN 은 파라미터가 적다"* 보다
***"CNN 은 Dense 에 넣기 전에 Pooling 으로 입력을 줄여놨다"*** 가 정확한 설명이다.

세 모델의 핵심 차이:

| | ANN | DNN | CNN |
|---|---|---|---|
| Flatten 위치 | 맨 앞 | 맨 앞 | **맨 뒤** |
| Hidden Dense | 1층 | 4층 | 1층 |
| Conv Block | 없음 | 없음 | 3개 |
| 공간 정보 활용 | ✗ | ✗ | ○ |

ANN 과 DNN 은 맨 앞에서 Flatten 하고 CNN 은 맨 뒤에서 한다.
DNN 이 ANN 보다 깊지만 **입력이 이미 Flatten 되어 있다는 제약은 동일하다.**
깊이를 늘리는 것으로는 잃어버린 공간 정보를 되찾을 수 없다.

---

## 6. 학습 시 주의

### 6-1. softmax 를 넣지 않았다 (logits 출력)

| 프레임워크 | 반드시 |
|---|---|
| PyTorch | `nn.CrossEntropyLoss()` |
| Keras | `SparseCategoricalCrossentropy(**from_logits=True**)` |

`from_logits` 를 빠뜨리면 loss 가 잘못 계산되는데 **에러는 안 나서 찾기 어렵다.**

### 6-2. seed 는 모델 만들기 직전마다

```python
for name in ("ann", "dnn", "cnn"):
    set_seed(SEED)        # ← 루프 안. 밖이 아니다
    model = build_model(name)
```

루프 밖에서 한 번만 하면 ANN 학습이 난수를 소모해서 DNN/CNN 이 다른 초기값을 받는다.
그러면 비교 조건이 어긋난다. (두 학습 스크립트에 이미 반영돼 있음)

### 6-3. 세 모델의 학습 조건은 절대 다르게 두지 마라

batch, lr, epoch, optimizer, seed 전부 동일해야 한다.
하나라도 모델별로 바꾸면 **"구조 때문에 생긴 차이"라고 말할 수 없게 된다.**

---

## 7. 결과 볼 때

**Accuracy 하나만 보면 안 된다.**

- **무작위 추측 = 1/6 = 16.7%** (두 스크립트가 자동으로 찍어준다)
- 이걸 못 넘으면 학습이 안 된 것

예상 시나리오와 대응:

| 상황 | 해석 |
|---|---|
| train loss↓ val loss↓ | 정상 |
| train loss↓ **val loss↑** | **과적합.** DNN 에서 제일 먼저 나타날 것이다 (파라미터가 제일 많아서) |
| 둘 다 높은 채 정체 | 과소적합 또는 lr 문제 |

**CNN 이 예상보다 나쁘면 — 이 순서로 확인해라:**
1. **데이터부터 의심해라.** crop 이 제대로 됐나? (2-4 절)
   모델 의심하기 전에 데이터를 본다. 대부분 여기서 나온다.
2. 두 프레임워크의 정규화가 같은가
3. `from_logits` 설정
4. 학습이 수렴했나 (early stopping 이 너무 일찍 잘랐을 수 있다)

**ANN/DNN 이 예상보다 좋다면** — 32×32 는 작은 이미지라 Flatten 의 불리함이
덜 드러날 수 있고, `shape×color` 는 **색만으로도 상당 부분 구분 가능**하다
(`circle_red` vs `circle_blue` 는 평균 색만으로 갈린다).
→ **형태가 다른 쌍만 따로 보면** CNN 의 우위가 드러날 것이다.

이건 실패가 아니라 **더 정밀한 분석의 출발점**이다.

---

## 8. 남은 작업 체크리스트

```
[ ] 학교 컴퓨터에 Python 3.12 + torch(cu121) + tensorflow 설치
[ ] torch.cuda.is_available() == True 확인
[ ] NVIDIA 제어판에서 Sysmem Fallback 끄기
[ ] AI-Hub c 세트 tar 2개 옮기기 (라벨 38MB + 원천 6.1GB)
[ ] tar 압축 해제
[ ] python prepare_data.py 실행
[ ] 🔴 crop 폴더 눈으로 확인
[ ] BATCH_SIZE 확인 (현재 32)
[ ] python train_keras.py --epochs 3  (빠른 확인)
[ ] python train_keras.py / python train_torch.py  (본 실행)
[ ] 결과 분석 — 형태 오류 vs 색상 오류 나눠서 보기
[ ] 포트폴리오 문서 작성
```

### 나중에 여유되면

- 신호등(`traffic_light`)이 데이터에 **39,888개** 있다. 상태(`red`/`green`/`left_arrow`)로
  클래스를 4개 더 만들 수 있다. 데이터셋 이름 자체가 "**신호등**-도로표지판 인지 영상"이다.
- 파라미터 수를 맞춘 대조 실험 — DNN 의 hidden units 를 줄여 CNN(619K)과 비슷하게 맞추고
  다시 돌린다. **파라미터가 같은데도 CNN 이 이긴다면 구조 때문이라는 결정적 증거**가 된다.

---

## 9. Data Augmentation 주의 (1차 실험에서는 쓰지 마라)

세 모델의 조건을 동일하게 유지하는 게 우선이고, augmentation 은 모델마다
다르게 작용할 수 있어 구조 비교를 흐린다. 나중에 확장할 때 참고:

| 기법 | 가능? | 이유 |
|---|---|---|
| **색상 변경 (Hue/Saturation)** | ❌ **절대 금지** | **색이 클래스 그 자체다.** red→blue 가 되면 라벨이 틀려진다 |
| Rotation | ⚠️ ±10° 이내 | 삼각형을 크게 돌리면 형태 인식이 무의미해진다 |
| Random Crop | ⚠️ 조금만 | 테두리가 잘리면 형태 정보가 사라진다 |
| Brightness / Contrast | ✅ | 조명 변화는 실제로 존재한다 |
| Horizontal Flip | ✅ | shape×color 에서는 의미가 안 바뀐다 |
