"""
실험 조건의 단일 출처(Single Source of Truth).

이 프로젝트의 목적은 "모델 구조의 차이"만으로 결과 차이를 설명하는 것이다.
따라서 데이터·전처리·학습 하이퍼파라미터는 여기 한 곳에서만 정의하고,
ANN / DNN / CNN, PyTorch / Keras 어느 쪽에서도 이 값을 그대로 가져다 쓴다.

여기 있는 값을 모델별로 다르게 두면 비교 실험이 무의미해진다.

── 2026-08-27 실제 AI-Hub 데이터 확인 후 전면 수정됨 ──
확인한 내용은 docs/DATA_FINDINGS.md 참고.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# AI-Hub 에서 받은 tar 파일이 있는 곳.
DOWNLOAD_ROOT = Path.home() / "Downloads"

# tar 를 풀고 crop 결과를 만들 작업 공간. 용량이 크므로 D: 드라이브를 쓴다.
AIHUB_ROOT = Path("D:/AIHub_TrafficSign")
RAW_DIR = AIHUB_ROOT / "raw"                  # 원천 jpg 압축 해제
ANNOTATION_DIR = AIHUB_ROOT / "annotations"   # 라벨 json 압축 해제
CROP_DIR = AIHUB_ROOT / "crop"                # bbox 로 잘라낸 표지판 (클래스별 폴더)

# manifest: 어떤 파일이 train/val/test 중 어디에 속하는지 적어둔 CSV.
# PyTorch 와 Keras 가 "완전히 같은 분할"을 쓰도록 강제하는 장치다.
MANIFEST_DIR = PROJECT_ROOT / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "dataset_manifest.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORT_DIR = RESULTS_DIR / "reports"
FIGURE_DIR = RESULTS_DIR / "figures"
HISTORY_DIR = RESULTS_DIR / "history"

# ---------------------------------------------------------------------------
# 데이터 소스  ─ 실제 확인 결과 반영
# ---------------------------------------------------------------------------
# 다운로드한 4개 tar 중 '라벨 ↔ 원천이 매칭되는' 것은 c 세트 하나뿐이다.
#   c (수도권 1280x720) : 원천 30,000장 ↔ 라벨 30,000개, 교집합 30,000  → 사용 가능
#   d (수도권 1920x1080): 라벨만 있고 원천 없음                          → 사용 불가
#   m (광역시)          : 라벨은 1280_720, 원천은 1920_1080 → 교집합 0   → 사용 불가
# 자세한 내용은 docs/DATA_FINDINGS.md 참고.
DATASET_KEY = "c_train_1280_720_daylight_1"
SOURCE_TAR = DOWNLOAD_ROOT / "신호등-도로표지판 인지 영상(수도권)" / "Training" / f"[원천]{DATASET_KEY}.tar"
LABEL_TAR = DOWNLOAD_ROOT / "신호등-도로표지판 인지 영상(수도권)" / "Training" / f"[라벨]{DATASET_KEY}.tar"

# ---------------------------------------------------------------------------
# 클래스 정의  ─ AI-Hub annotation 의 실제 값에서 유도
# ---------------------------------------------------------------------------
# 중요: AI-Hub 데이터에는 'stop', 'speed_30' 같은 표지판 의미 라벨이 없다.
# annotation 이 실제로 주는 것은 다음 속성들이다.
#     class : traffic_sign / traffic_light / traffic_information
#     shape : circle / triangle / rectangle
#     color : red / blue / yellow
#     type  : restriction / instruction / caution / pedestrian  (자주 누락됨)
#
# type 은 결측이 많아서(circle_red 중 약 20%) 분류 기준으로 쓰기 부적절하다.
# 반면 shape 와 color 는 traffic_sign 전 객체에 빠짐없이 존재한다.
# 따라서 분류 대상을 shape × color 조합으로 정의한다.
#
# 이 6개 클래스는 실제 도로표지판 체계와도 대응된다:
#     circle_red      규제표지 (속도제한, 진입금지, 주정차금지 ...)
#     circle_blue     지시표지 (지정방향, 전용차로 ...)
#     triangle_yellow 주의표지 (위험, 급커브 ...)
#     triangle_blue   지시표지 중 삼각형
#     triangle_red    규제표지 중 삼각형
#     rectangle_blue  지시/안내표지 중 사각형
CLASS_NAMES: list[str] = [
    "circle_blue",
    "circle_red",
    "rectangle_blue",
    "triangle_blue",
    "triangle_red",
    "triangle_yellow",
]
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Crop 조건  ─ bbox 크기 분포를 확인하고 정한 값
# ---------------------------------------------------------------------------
# 실제 bbox 짧은 변 분포 (c 세트, traffic_sign 54,500개):
#     p5=8px  p10=9px  p25=12px  p50=17px  p75=26px  p90=39px  p95=48px  p99=68px
# 즉 절반이 17px 이하다. 멀리 있는 표지판이 대부분이라 그렇다.
#
# 너무 작은 bbox 를 crop 하면 뭉개진 이미지가 되어 사람도 구분 못 한다.
# 그런 걸 학습에 넣으면 모델이 아니라 데이터가 성능을 망친다.
# 짧은 변 24px 이상만 쓰면 클래스별로 872~8,153장이 남아 1차 실험에 충분하다.
MIN_BBOX_SIZE = 24

# bbox 주변 여유. 표지판 테두리가 잘리지 않도록 약간 넓게 자른다.
CROP_PADDING_RATIO = 0.10

# 클래스당 최대 수집 장수. 클래스 불균형을 완화한다.
# circle_red 는 8,153장이나 되지만 rectangle_blue 는 872장뿐이라
# 그대로 쓰면 circle_red 만 맞혀도 정확도가 높게 나온다.
MAX_IMAGES_PER_CLASS = 800

# ---------------------------------------------------------------------------
# 입력 규격  ─ 세 모델 공통
# ---------------------------------------------------------------------------
# 원본 bbox 가 24~48px 대가 대부분이므로 32x32 로 맞춘다.
# 64x64 로 키우면 없는 정보를 늘려서 만드는 셈이라 얻는 게 없다.
# (GTSRB 등 공개 표지판 데이터셋도 32x32 를 표준으로 쓴다.)
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3

# ANN / DNN 은 이 값을 1차원으로 펴서(Flatten) 입력으로 받는다.
# 32 * 32 * 3 = 3,072  →  Dense Layer 의 파라미터가 폭증하는 원인.
# 이 숫자 자체가 ANN 과 CNN 의 구조적 차이를 설명하는 핵심 포인트다.
FLATTENED_INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS

# 정규화: [0,1] → (x-0.5)/0.5 → [-1,1]
# ImageNet 통계를 쓰지 않는 이유는 사전학습 모델을 쓰지 않기 때문이고,
# 두 프레임워크에서 동일하게 재현하기도 쉽기 때문이다.
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# ---------------------------------------------------------------------------
# 데이터 분할  ─ 세 모델 공통
# ---------------------------------------------------------------------------
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# 데이터 누수 방지.
# 같은 원본 이미지 한 장에서 표지판을 여러 개 자르는 경우가 흔하다.
# (샘플 파일 하나에 traffic_sign 이 6개 있었다.)
# 그것들이 train 과 test 에 나뉘어 들어가면 배경·조명·촬영조건이 동일한
# 사실상 같은 장면을 양쪽에서 보게 된다.
# 따라서 '원본 이미지 단위'로 묶어서 통째로 한쪽에만 배정한다.
GROUP_AWARE_SPLIT = True

# 더 강한 누수 방지: 연속 촬영 프레임까지 묶는다.
# 파일명이 s01000200 처럼 순번이라 ID 가 가까우면 같은 주행 구간일 가능성이 높다.
# ID 를 이 값으로 나눈 몫을 그룹으로 쓴다. 0 이면 원본 이미지 단위만 적용.
GROUP_ID_BUCKET = 1000

# ---------------------------------------------------------------------------
# 학습 하이퍼파라미터  ─ 세 모델 공통
# ---------------------------------------------------------------------------
# 아래 값은 모델별로 절대 바꾸지 않는다. 바꾸는 순간 "구조 때문에 생긴 차이"라고
# 말할 수 없게 된다.
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
OPTIMIZER = "adam"
LOSS = "cross_entropy"

SEED = 42
NUM_WORKERS = 4

# Early stopping: val loss 가 이 횟수만큼 연속으로 개선되지 않으면 중단.
# 세 모델에 동일하게 적용해야 학습 시간 비교가 공정하다.
EARLY_STOPPING_PATIENCE = 7

# ---------------------------------------------------------------------------
# 모델 구조 정의  ─ 여기만 모델마다 다르다
# ---------------------------------------------------------------------------
# ANN: 단순한 Fully Connected Network (Hidden Dense 1개)
ANN_HIDDEN_UNITS = (256,)

# DNN: Hidden Dense Layer 를 여러 개 쌓은 더 깊은 FC Network
DNN_HIDDEN_UNITS = (1024, 512, 256, 128)

# CNN: Convolution Block 을 쌓고 마지막에만 Dense 를 쓴다.
#   32 → 16 → 8 → 4  (3개 Block, 최종 feature map 4x4)
CNN_CONV_CHANNELS = (32, 64, 128)
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2
CNN_HEAD_UNITS = (256,)

DROPOUT = 0.5

MODEL_NAMES = ("ann", "dnn", "cnn")
FRAMEWORKS = ("torch", "keras")


def ensure_dirs() -> None:
    """산출물 디렉터리를 만들어 둔다."""
    for path in (MANIFEST_DIR, RESULTS_DIR, CHECKPOINT_DIR,
                 REPORT_DIR, FIGURE_DIR, HISTORY_DIR):
        path.mkdir(parents=True, exist_ok=True)


def input_shape(framework: str) -> tuple[int, ...]:
    """
    같은 이미지라도 프레임워크마다 텐서 축 순서가 다르다.

        PyTorch : (C, H, W)  — channels-first
        Keras   : (H, W, C)  — channels-last

    Flatten 결과의 '원소 순서'는 달라지지만 개수는 3,072 로 같다.
    파라미터 수가 정확히 일치해야 비교가 성립하므로 여기서 명시해 둔다.
    """
    if framework == "torch":
        return (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    if framework == "keras":
        return (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
    raise ValueError(f"알 수 없는 framework: {framework!r} (가능: {FRAMEWORKS})")
