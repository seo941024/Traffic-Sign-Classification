"""
Crop 결과를 눈으로 확인한다.

이 단계를 건너뛰면 안 되는 이유:
    bbox 좌표 형식을 잘못 읽어도 코드는 **에러 없이 잘 돈다.**
    (x1,y1,x2,y2) 를 (x,y,w,h) 로 착각해도 crop 은 그냥 엉뚱한 영역을 자르고
    파일도 정상적으로 저장된다. 학습도 돌아간다. 정확도만 이상하게 낮게 나온다.
    그 시점에 원인을 찾으려면 훨씬 오래 걸린다.

    "모델이 이상하다"고 의심하기 전에 "데이터가 맞나"를 먼저 확인해라.

클래스별로 샘플을 모아 한 장의 대조 시트(contact sheet)로 만든다.
저장된 png 를 열어서 다음을 확인해라:
    - 표지판이 화면 중앙에 제대로 들어와 있는가?
    - 테두리가 잘리지 않았는가?
    - circle_red 폴더에 정말 빨간 원형 표지판만 있는가?
    - 사람이 봐서 클래스를 구분할 수 있는가? (뭉개져 있으면 min-size 를 올려라)

사용법:
    python scripts/03_verify_crops.py
    python scripts/03_verify_crops.py --per-class 12 --cell 96
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser(description="crop 결과 시각 검증")
    parser.add_argument("--crop-dir", type=Path, default=config.CROP_DIR)
    parser.add_argument("--per-class", type=int, default=10, help="클래스당 샘플 수")
    parser.add_argument("--cell", type=int, default=96, help="셀 한 변 픽셀")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow 가 필요하다: pip install pillow")
        sys.exit(1)

    if not args.crop_dir.exists():
        print(f"crop 디렉터리가 없다: {args.crop_dir}")
        print("먼저 python scripts/02_build_crops.py 를 실행해라.")
        sys.exit(1)

    class_dirs = sorted(p for p in args.crop_dir.iterdir() if p.is_dir())
    if not class_dirs:
        print(f"{args.crop_dir} 안에 클래스 폴더가 없다.")
        sys.exit(1)

    rng = random.Random(args.seed)
    cell = args.cell
    label_width = 150
    padding = 4

    rows = []
    print("=" * 60)
    print("Crop 검증")
    print("=" * 60)

    for class_dir in class_dirs:
        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            print(f"  {class_dir.name:<18} 0장 — 건너뛴다")
            continue

        # 원본 크기 통계도 같이 본다. 너무 작으면 학습에 쓸 수 없다.
        sizes = []
        for path in images[:200]:
            try:
                with Image.open(path) as image:
                    sizes.append(min(image.size))
            except OSError:
                continue
        median = sorted(sizes)[len(sizes) // 2] if sizes else 0

        print(f"  {class_dir.name:<18} {len(images):>6,}장   짧은변 중앙값 {median}px")
        rows.append((class_dir.name, rng.sample(images, min(args.per_class, len(images)))))

    if not rows:
        print("\n표시할 이미지가 없다.")
        sys.exit(1)

    sheet_width = label_width + args.per_class * (cell + padding) + padding
    sheet_height = len(rows) * (cell + padding) + padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    for row_index, (class_name, samples) in enumerate(rows):
        y = padding + row_index * (cell + padding)
        draw.text((padding, y + cell // 2 - 6), class_name, fill=(20, 20, 20))

        for col_index, path in enumerate(samples):
            x = label_width + col_index * (cell + padding)
            try:
                with Image.open(path) as image:
                    # 원본 비율을 유지해서 붙인다.
                    # 강제로 정사각형으로 늘리면 실제 crop 이 찌그러졌는지 안 보인다.
                    thumb = image.convert("RGB")
                    thumb.thumbnail((cell, cell), Image.LANCZOS)
                    offset_x = x + (cell - thumb.width) // 2
                    offset_y = y + (cell - thumb.height) // 2
                    sheet.paste(thumb, (offset_x, offset_y))
            except OSError:
                draw.rectangle([x, y, x + cell, y + cell], outline=(200, 0, 0))

    config.ensure_dirs()
    output = config.FIGURE_DIR / "crop_verification.png"
    sheet.save(output)

    print(f"\n저장: {output}")
    print("""
이 이미지를 열어서 확인해라:
  1. 표지판이 중앙에 제대로 들어와 있는가?
     → 아니면 bbox 좌표 해석이 틀렸다. src/data/aihub.py 의 _normalize_box 를 봐라.
  2. 테두리가 잘렸는가?
     → config.CROP_PADDING_RATIO 를 올려라.
  3. 사람이 봐서 클래스를 구분할 수 있는가?
     → 뭉개져 있으면 config.MIN_BBOX_SIZE 를 올려라. 사람이 못 하면 모델도 못 한다.
  4. 폴더 이름과 내용이 일치하는가? (circle_red 에 파란 표지판이 섞여 있지 않은가)

여기서 문제가 없다고 확인한 뒤에 다음으로 넘어가라:
  python scripts/04_build_manifest.py
""")


if __name__ == "__main__":
    main()
