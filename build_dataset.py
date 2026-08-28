"""
표지판코드분류 crop 데이터에서 상위 N개 코드를 골라 학습용 crop/ 폴더를 만든다.

    표지판코드분류crop데이터{1,2}/.../result_*/<코드>/*.jpg
        ↓  (짧은 변 >= MIN_SIZE 필터, 코드당 CAP 장)
    crop/<코드>/*.jpg

train/val 분할은 train_keras.py / train_torch.py 의 로더가 알아서 한다.

실행:
    python build_dataset.py --src "D:/신호등-도로표지판 인지 영상(수도권)/Training"
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

MIN_SIZE = 24          # 짧은 변이 이보다 작으면 버린다 (뭉개져서 못 알아본다)
CAP = 1000             # 코드(클래스)당 최대 장수
TOP_N = 10             # 상위 몇 개 코드를 쓸지
SEED = 42
EXCLUDE = {"224", "blank"}   # 224=여러 속도제한이 한 폴더, blank=판독불가


def find_class_roots(src: Path):
    """result_* 안의 <코드> 폴더들을 코드별로 모은다: {코드: [경로, ...]}."""
    roots: dict[str, list[Path]] = {}
    for crop_set in src.glob("표지판코드분류crop데이터*"):
        for result_dir in crop_set.rglob("result_*"):
            if not result_dir.is_dir():
                continue
            for code_dir in result_dir.iterdir():
                if code_dir.is_dir():
                    roots.setdefault(code_dir.name, []).append(code_dir)
    return roots


def big_enough(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            w, h = im.size
        return min(w, h) >= MIN_SIZE
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True,
                        help="Training 폴더 (표지판코드분류crop데이터* 가 그 안에 있음)")
    parser.add_argument("--out", type=Path, default=Path("crop"))
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--cap", type=int, default=CAP)
    args = parser.parse_args()

    roots = find_class_roots(args.src)
    print(f"코드 {len(roots):,}개 발견", flush=True)

    # 1) 원본 파일 개수(listdir, 헤더 안 읽음)로 후보 코드를 고른다.
    raw_files: dict[str, list[Path]] = {}
    for code, dirs in roots.items():
        if code in EXCLUDE:
            continue
        fs = []
        for d in dirs:
            for f in d.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    fs.append(f)
        raw_files[code] = fs

    raw_counts = Counter({c: len(fs) for c, fs in raw_files.items()})
    # 필터로 좀 줄어들 것을 감안해 후보를 넉넉히 잡고, 아래서 실제 확보량으로 재정렬.
    candidates = [c for c, _ in raw_counts.most_common(args.top_n + 8)]
    print("후보:", candidates, flush=True)

    # 2) 후보에 대해서만 헤더를 읽어 >=MIN_SIZE 파일을 CAP*3 개까지 모은다.
    rng = random.Random(SEED)
    per_code: dict[str, list[Path]] = {}
    for code in candidates:
        fs = raw_files[code][:]
        rng.shuffle(fs)
        kept = []
        for f in fs:
            if big_enough(f):
                kept.append(f)
            if len(kept) >= args.cap * 3:
                break
        per_code[code] = kept
        print(f"  {code:>8}  원본 {len(raw_files[code]):>7,}  확보 {len(kept):>5,}", flush=True)

    counts = Counter({c: len(fs) for c, fs in per_code.items()})
    chosen = [c for c, _ in counts.most_common(args.top_n)]

    print("\n=== 최종 선택 (>=%dpx) ===" % MIN_SIZE)
    for c, n in counts.most_common():
        mark = "  <== 선택" if c in chosen else ""
        print(f"  {c:>8} {min(n, args.cap):>7,}{mark}")

    if args.out.exists():
        shutil.rmtree(args.out)

    total = 0
    for code in chosen:
        files = per_code[code][: args.cap]
        dst_dir = args.out / code
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_path in files:
            # result_1 / result_2 에 같은 파일명이 있을 수 있어 접두사를 붙인다.
            tag = src_path.parent.parent.name  # result_N
            shutil.copy2(src_path, dst_dir / f"{tag}_{src_path.name}")
        total += len(files)
        print(f"  {code:>8}  {len(files):>5,}장 복사")

    print(f"\n총 {total:,}장 → {args.out}")
    print("반드시 crop/ 폴더를 열어 클래스별로 제대로 들어갔는지 눈으로 확인해라.")


if __name__ == "__main__":
    main()
