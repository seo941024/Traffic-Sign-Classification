"""
환경 점검 — 학습을 시작하기 전에 제일 먼저 돌려볼 것.

핸드오프 32, 33절 대응.
Python 버전과 PyTorch/TensorFlow 의 CUDA 호환성이 어긋나면
"코드는 도는데 GPU 를 안 쓰는" 상태가 되고, 그러면 학습이 몇십 배 느려진다.
느려진 이유를 나중에 찾는 것보다 지금 확인하는 게 싸다.

사용법:
    python scripts/00_check_env.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def check_python() -> None:
    print("=" * 60)
    print("Python")
    print("=" * 60)
    print(f"  버전   : {platform.python_version()}")
    print(f"  실행파일: {sys.executable}")
    print(f"  OS     : {platform.platform()}")

    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 13):
        print(
            f"  경고: Python {major}.{minor} 는 딥러닝 패키지 지원이 아직 늦을 수 있다.\n"
            f"        PyTorch/TensorFlow 가 정상 설치된 환경인지 아래 결과로 확인해라."
        )
    print()


def check_torch() -> None:
    print("=" * 60)
    print("PyTorch")
    print("=" * 60)
    try:
        import torch
    except ImportError:
        print("  설치되지 않음 → pip install torch --index-url ...")
        print()
        return

    print(f"  버전          : {torch.__version__}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA 버전     : {torch.version.cuda}")
        print(f"  GPU 개수      : {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            total = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
            print(f"    [{index}] {name} ({total:.1f} GB)")
    else:
        print("  GPU 를 못 쓴다. CPU 로도 학습은 되지만 매우 느리다.")
        print("  torch 를 CPU 전용 빌드로 설치했을 가능성이 가장 크다.")
    print()


def check_tensorflow() -> None:
    print("=" * 60)
    print("TensorFlow / Keras")
    print("=" * 60)
    try:
        import tensorflow as tf
    except ImportError:
        print("  설치되지 않음 → pip install tensorflow")
        print()
        return

    print(f"  버전     : {tf.__version__}")
    try:
        import keras

        print(f"  Keras    : {keras.__version__}")
    except ImportError:
        pass

    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPU 인식 : {len(gpus)}개")
    for gpu in gpus:
        print(f"    {gpu.name}")

    if not gpus:
        print("  GPU 를 못 쓴다.")
        print("  참고: TensorFlow 2.11 부터 Windows 네이티브에서는 GPU 를 지원하지 않는다.")
        print("        Windows 에서 TF GPU 를 쓰려면 WSL2 를 쓰거나 tensorflow<2.11 을 써야 한다.")
        print("        PyTorch 는 Windows 네이티브에서 GPU 가 정상 동작한다.")
    print()


def check_paths() -> None:
    print("=" * 60)
    print("경로")
    print("=" * 60)
    targets = [
        ("AI-Hub 루트", config.AIHUB_ROOT),
        ("원본 (raw)", config.RAW_DIR),
        ("Annotation", config.ANNOTATION_DIR),
        ("Crop 결과", config.CROP_DIR),
        ("Manifest", config.MANIFEST_PATH),
    ]
    for label, path in targets:
        mark = "있음" if path.exists() else "없음"
        print(f"  [{mark}] {label:<12} {path}")
    print()


def check_config() -> None:
    print("=" * 60)
    print("실험 조건 (세 모델 공통)")
    print("=" * 60)
    print(f"  입력 크기      : {config.IMAGE_SIZE} x {config.IMAGE_SIZE} x {config.IMAGE_CHANNELS}")
    print(f"  Flatten 차원   : {config.FLATTENED_INPUT_DIM:,}  ← ANN/DNN 의 입력 특징 개수")
    print(f"  Batch size     : {config.BATCH_SIZE}")
    print(f"  Epochs         : {config.EPOCHS}")
    print(f"  Learning rate  : {config.LEARNING_RATE}")
    print(f"  Optimizer      : {config.OPTIMIZER}")
    print(f"  Seed           : {config.SEED}")
    print(f"  분할 비율      : {config.SPLIT_RATIOS}")
    print(f"  클래스 수      : {config.NUM_CLASSES}"
          f"{'  (아직 미확정 — Annotation 확인 후 config.py 에 채워라)' if config.NUM_CLASSES == 0 else ''}")
    print()


def main() -> None:
    check_python()
    check_torch()
    check_tensorflow()
    check_paths()
    check_config()
    print("점검 완료.")


if __name__ == "__main__":
    main()
