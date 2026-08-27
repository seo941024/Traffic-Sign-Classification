"""
Annotation 의 bbox 로 원본 이미지에서 표지판만 잘라내 클래스별 폴더로 저장한다.

    D:/AIHub_TrafficSign/crop/
      ├── circle_blue/
      ├── circle_red/
      ├── rectangle_blue/
      ├── triangle_blue/
      ├── triangle_red/
      └── triangle_yellow/

잘라낸 파일 이름은 "{원본stem}__{순번}.jpg" 로 짓는다.
원본 이름을 남겨두는 이유: 나중에 분할할 때 "같은 원본에서 나온 crop"을
하나로 묶어야 데이터 누수를 막을 수 있기 때문이다. (핸드오프 15절)

사용법:
    python scripts/02_build_crops.py --limit 500        # 먼저 소규모로 검증
    python scripts/02_build_crops.py                    # 전체
    python scripts/02_build_crops.py --dry-run          # 저장 없이 통계만
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.data.aihub import crop_box, parse_annotation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="bbox 기반 표지판 crop 생성")
    parser.add_argument("--annotation-dir", type=Path, default=config.ANNOTATION_DIR)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.CROP_DIR)
    parser.add_argument("--min-size", type=int, default=config.MIN_BBOX_SIZE,
                        help="bbox 짧은 변 최소 픽셀. 이보다 작으면 버린다.")
    parser.add_argument("--max-per-class", type=int, default=config.MAX_IMAGES_PER_CLASS,
                        help="클래스당 최대 장수. 클래스 불균형 완화용.")
    parser.add_argument("--limit", type=int, default=None,
                        help="처리할 annotation 파일 수 상한 (파이프라인 검증용)")
    parser.add_argument("--dry-run", action="store_true",
                        help="파일을 쓰지 않고 통계만 낸다")
    args = parser.parse_args()

    if not args.annotation_dir.exists():
        print(f"annotation 디렉터리가 없다: {args.annotation_dir}")
        print("먼저 python scripts/01_extract_tars.py 를 실행해라.")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        print("Pillow 가 필요하다: pip install pillow")
        sys.exit(1)

    json_files = sorted(args.annotation_dir.rglob("*.json"))
    if args.limit:
        json_files = json_files[: args.limit]

    if not json_files:
        print(f"json 을 못 찾았다: {args.annotation_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Crop 생성")
    print("=" * 60)
    print(f"annotation : {args.annotation_dir}  ({len(json_files):,}개)")
    print(f"원천 이미지: {args.raw_dir}")
    print(f"출력       : {args.out_dir}")
    print(f"최소 크기  : {args.min_size}px (짧은 변)")
    print(f"클래스 상한: {args.max_per_class}장")
    if args.dry_run:
        print("** dry-run: 파일을 쓰지 않는다 **")
    print()

    saved = Counter()          # 클래스별 저장 수
    rejected = Counter()       # 버린 이유별 집계
    seen_classes = Counter()   # 크기 필터 전 클래스 분포

    # 같은 원본 이미지를 반복해서 열지 않도록 캐시 없이 순차 처리한다.
    # (json 하나당 원본 하나이므로 자연스럽게 1회만 연다)
    for index, json_path in enumerate(json_files, start=1):
        if index % 2000 == 0:
            total = sum(saved.values())
            print(f"  {index:,}/{len(json_files):,} 처리  저장 {total:,}장")

        annotation = parse_annotation(json_path)
        if annotation is None:
            rejected["json 파싱 실패"] += 1
            continue
        if not annotation.objects:
            rejected["표지판 없음"] += 1
            continue

        # 크기 필터를 통과하고, 아직 상한에 안 걸린 객체만 남긴다.
        targets = []
        for obj in annotation.objects:
            seen_classes[obj.class_name] += 1
            if obj.class_name not in config.CLASS_NAMES:
                rejected["대상 클래스 아님"] += 1
                continue
            if obj.short_side < args.min_size:
                rejected[f"너무 작음 (<{args.min_size}px)"] += 1
                continue
            if saved[obj.class_name] >= args.max_per_class:
                rejected["클래스 상한 도달"] += 1
                continue
            targets.append(obj)

        if not targets:
            continue

        image_path = args.raw_dir / annotation.filename
        if not image_path.exists():
            # 하위 폴더에 들어가 있을 수 있으니 한 번 더 찾아본다.
            candidates = list(args.raw_dir.rglob(annotation.filename))
            if not candidates:
                rejected["원본 이미지 없음"] += 1
                continue
            image_path = candidates[0]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size

                for order, obj in enumerate(targets):
                    if saved[obj.class_name] >= args.max_per_class:
                        continue

                    box = crop_box(obj, width, height, config.CROP_PADDING_RATIO)
                    patch = image.crop(box)
                    if patch.width < 1 or patch.height < 1:
                        rejected["crop 결과 비어 있음"] += 1
                        continue

                    if not args.dry_run:
                        class_dir = args.out_dir / obj.class_name
                        class_dir.mkdir(parents=True, exist_ok=True)
                        out_path = class_dir / f"{annotation.stem}__{order}.jpg"
                        patch.save(out_path, quality=95)

                    saved[obj.class_name] += 1
        except OSError:
            rejected["이미지 열기 실패"] += 1
            continue

        if args.max_per_class and all(
            saved[name] >= args.max_per_class for name in config.CLASS_NAMES
        ):
            print(f"\n  모든 클래스가 상한({args.max_per_class})에 도달 — 조기 종료")
            break

    # ---------------------------------------------------------------- 결과
    print("\n" + "=" * 60)
    print("결과")
    print("=" * 60)

    print("\n[크기 필터 적용 전 — 전체 표지판 분포]")
    for name, count in seen_classes.most_common():
        mark = "" if name in config.CLASS_NAMES else "  (대상 아님)"
        print(f"  {name:<18} {count:>8,}{mark}")

    print(f"\n[저장된 crop]  총 {sum(saved.values()):,}장")
    for name in config.CLASS_NAMES:
        print(f"  {name:<18} {saved[name]:>8,}")

    print("\n[제외 사유]")
    for reason, count in rejected.most_common():
        print(f"  {reason:<26} {count:>8,}")

    empty = [name for name in config.CLASS_NAMES if saved[name] == 0]
    if empty:
        print(f"\n경고: 아래 클래스가 0장이다 — {empty}")
        print("      --min-size 를 낮추거나 클래스 목록을 조정해라.")

    counts = [saved[name] for name in config.CLASS_NAMES if saved[name] > 0]
    if counts and max(counts) / min(counts) > 3:
        print(f"\n경고: 클래스 불균형이 크다 (최대/최소 = {max(counts) / min(counts):.1f}배)")
        print("      정확도만 보면 다수 클래스에 속아넘어간다. Macro F1 을 같이 봐라.")

    if args.dry_run:
        print("\ndry-run 이라 파일은 저장하지 않았다.")
    else:
        print(f"\n저장 위치: {args.out_dir}")
        print("\n다음: crop 결과를 **눈으로 직접 확인해라.**")
        print("      python scripts/03_verify_crops.py")
        print("      bbox 형식을 잘못 읽어도 코드는 에러 없이 잘 돈다.")


if __name__ == "__main__":
    main()
