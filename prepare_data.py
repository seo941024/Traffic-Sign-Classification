"""
AI-Hub json 에서 표지판만 잘라내 클래스별 폴더로 저장한다.

    crop/
      circle_red/    s01027933_0.jpg ...
      circle_blue/   ...

실행:
    python prepare_data.py --json-dir <json폴더> --image-dir <jpg폴더> --out crop
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

import config

MIN_SIZE = 24        # bbox 짧은 변이 이보다 작으면 버린다 (뭉개져서 못 알아본다)
MAX_PER_CLASS = 800  # 클래스 불균형 완화
PADDING = 0.1        # 테두리 안 잘리게 여유


def load_boxes(json_path):
    """json 하나에서 (파일명, [(x1,y1,x2,y2,클래스), ...]) 를 뽑는다."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, []

    filename = (data.get("image") or {}).get("filename")
    if not filename:
        return None, []

    boxes = []
    for obj in data.get("annotation") or []:
        # traffic_light / traffic_information 은 제외
        if obj.get("class") != "traffic_sign":
            continue

        shape = str(obj.get("shape") or "").lower()
        color = str(obj.get("color") or "").lower()
        name = f"{shape}_{color}"
        if name not in config.CLASS_NAMES:
            continue

        box = obj.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        # 좌표가 뒤집힌 경우가 실제로 있다: [315,308,287,335]
        # min/max 로 정규화 안 하면 빈 이미지가 나오는데 에러는 안 난다.
        a, b, c, d = (int(float(v)) for v in box)
        x1, x2 = min(a, c), max(a, c)
        y1, y2 = min(b, d), max(b, d)

        if min(x2 - x1, y2 - y1) < MIN_SIZE:
            continue

        boxes.append((x1, y1, x2, y2, name))

    return filename, boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("crop"))
    args = parser.parse_args()

    # 원본 jpg 를 매번 찾지 않도록 미리 인덱스를 만든다.
    index = {p.name: p for p in args.image_dir.rglob("*.jpg")}
    print(f"원본 이미지 {len(index):,}장")

    json_files = sorted(args.json_dir.rglob("*.json"))
    print(f"annotation {len(json_files):,}개\n")

    saved = Counter()

    for i, json_path in enumerate(json_files, 1):
        if i % 2000 == 0:
            print(f"  {i:,}/{len(json_files):,}  저장 {sum(saved.values()):,}장")

        filename, boxes = load_boxes(json_path)
        if not boxes:
            continue

        # 아직 상한에 안 걸린 것만
        boxes = [b for b in boxes if saved[b[4]] < MAX_PER_CLASS]
        if not boxes:
            continue

        image_path = index.get(filename)
        if image_path is None:
            continue

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                W, H = image.size

                for n, (x1, y1, x2, y2, name) in enumerate(boxes):
                    px = int((x2 - x1) * PADDING)
                    py = int((y2 - y1) * PADDING)
                    patch = image.crop((
                        max(0, x1 - px), max(0, y1 - py),
                        min(W, x2 + px), min(H, y2 + py),
                    ))

                    out_dir = args.out / name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    patch.save(out_dir / f"{Path(filename).stem}_{n}.jpg", quality=95)
                    saved[name] += 1
        except OSError:
            continue

        if all(saved[c] >= MAX_PER_CLASS for c in config.CLASS_NAMES):
            print("\n  모든 클래스가 상한 도달, 종료")
            break

    print(f"\n총 {sum(saved.values()):,}장")
    for name in config.CLASS_NAMES:
        print(f"  {name:<18} {saved[name]:>6,}")
    print(f"\n저장 위치: {args.out}")
    print("\n반드시 폴더를 열어서 제대로 잘렸는지 눈으로 확인해라.")
    print("bbox 를 잘못 읽어도 에러 없이 그냥 엉뚱한 영역이 저장된다.")


if __name__ == "__main__":
    main()
