"""
Crop 된 이미지 폴더 → manifest CSV (train/val/test 분할 확정).

전제: crop 이 이미 끝나서 아래처럼 클래스별 폴더가 만들어져 있어야 한다.

    D:/AIHub_TrafficSign/crop/
      ├── circle_blue/
      ├── circle_red/
      ├── triangle_yellow/
      └── ...

이 스크립트는 분할을 딱 한 번 수행하고 그 결과를 CSV 로 못박는다.
이후 PyTorch 와 Keras 는 이 CSV 를 읽기만 한다.
→ "세 모델이 완전히 동일한 데이터를 썼다"가 가정이 아니라 사실이 된다. (핸드오프 19절)

사용법:
    python scripts/04_build_manifest.py
    python scripts/04_build_manifest.py --limit-per-class 30    # 파이프라인 검증용 소규모
    python scripts/04_build_manifest.py --strategy image        # 누수 대조군
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.data.manifest import Sample, summarize, write_manifest  # noqa: E402
from src.data.split import infer_group_id, split_by_group, split_by_image  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_samples(crop_dir: Path, limit_per_class: int | None) -> list[Sample]:
    """
    클래스별 폴더를 훑어 이미지를 모은다.

    class_id 는 폴더명을 정렬한 순서로 0부터 부여한다.
    정렬을 쓰는 이유: 파일시스템 순서는 환경에 따라 달라질 수 있는데,
    그러면 다시 돌릴 때마다 라벨이 바뀐다. 정렬하면 항상 같은 결과가 나온다.
    """
    if not crop_dir.exists():
        raise FileNotFoundError(
            f"crop 디렉터리가 없다: {crop_dir}\n"
            f"Annotation 확인 → bbox crop 을 먼저 끝내라. (핸드오프 39절)"
        )

    class_dirs = sorted(p for p in crop_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"{crop_dir} 안에 클래스 폴더가 없다.")

    samples: list[Sample] = []
    for class_id, class_dir in enumerate(class_dirs):
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit_per_class is not None:
            images = images[:limit_per_class]

        if not images:
            print(f"  경고: {class_dir.name} 에 이미지가 없다 — 건너뛴다.")
            continue

        for image_path in images:
            samples.append(
                Sample(
                    filepath=image_path.resolve(),
                    class_name=class_dir.name,
                    class_id=class_id,
                    # split 은 아직 미정. 다음 단계에서 채운다.
                    split="",
                    group_id=infer_group_id(image_path),
                )
            )
        print(f"  {class_dir.name:<20} {len(images):>6,}장")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="crop 이미지로 manifest 생성")
    parser.add_argument("--crop-dir", type=Path, default=config.CROP_DIR)
    parser.add_argument("--output", type=Path, default=config.MANIFEST_PATH)
    parser.add_argument(
        "--limit-per-class", type=int, default=None,
        help="클래스당 최대 이미지 수. 파이프라인 검증 시 30 정도로 시작해라. (핸드오프 29.2)",
    )
    parser.add_argument(
        "--strategy", choices=("group", "image"), default=None,
        help="group=촬영 단위로 분할(권장), image=이미지 단위 무작위(누수 대조군)",
    )
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    strategy = args.strategy or ("group" if config.GROUP_AWARE_SPLIT else "image")

    print("=" * 60)
    print("Manifest 생성")
    print("=" * 60)
    print(f"crop 디렉터리: {args.crop_dir}")
    print(f"분할 방식    : {strategy}")
    print(f"분할 비율    : {config.SPLIT_RATIOS}")
    print(f"seed         : {args.seed}")
    if args.limit_per_class:
        print(f"클래스당 상한: {args.limit_per_class}장")
    print()

    print("[클래스별 수집]")
    samples = collect_samples(args.crop_dir, args.limit_per_class)
    if not samples:
        print("\n수집된 이미지가 없다. 중단한다.")
        return

    group_count = len({s.group_id for s in samples})
    print(f"\n총 {len(samples):,}장, 그룹 {group_count:,}개")

    if strategy == "group":
        if group_count == len(samples):
            print(
                "\n경고: 그룹 수가 이미지 수와 같다 = 그룹이 전혀 묶이지 않았다.\n"
                "      infer_group_id() 가 실제 파일명 규칙과 안 맞는다는 뜻이다.\n"
                "      src/data/split.py 의 infer_group_id 를 실제 규칙에 맞게 고쳐라.\n"
                "      이대로 두면 누수 방지가 아무 일도 하지 않는다."
            )
        splits = split_by_group(samples, config.SPLIT_RATIOS, args.seed)
    else:
        print(
            "\n주의: 이미지 단위 분할은 데이터 누수에 취약하다.\n"
            "      같은 장면이 train 과 test 에 함께 들어갈 수 있다. (핸드오프 15절)"
        )
        splits = split_by_image(samples, config.SPLIT_RATIOS, args.seed)

    write_manifest(splits, args.output)
    print(f"\n저장: {args.output}")

    print("\n" + "=" * 60)
    print("분할 결과")
    print("=" * 60)
    print(summarize(args.output))

    class_names = sorted({s.class_name for s in splits})
    print("\n" + "=" * 60)
    print("config.py 에 반영할 것")
    print("=" * 60)
    print("CLASS_NAMES = [")
    for name in class_names:
        print(f'    "{name}",')
    print("]")
    print(f"\n→ NUM_CLASSES 는 {len(class_names)} 가 된다.")


if __name__ == "__main__":
    main()
