"""
다운로드한 tar 를 D: 드라이브에 풀고, 라벨↔원천 매칭을 검증한다.

먼저 검증부터 하는 이유:
    실제로 확인해보니 다운로드한 4개 tar 중 매칭되는 건 c 세트 하나뿐이었다.
    m(광역시)은 라벨이 1280x720, 원천이 1920x1080 이라 교집합이 0이다.
    이걸 모르고 6GB 를 풀고 나서 발견하면 시간이 아깝다.

사용법:
    python scripts/01_extract_tars.py --verify-only     # 매칭만 확인 (압축 해제 안 함)
    python scripts/01_extract_tars.py                   # 검증 후 압축 해제
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def list_members(tar_path: Path, suffix: str) -> set[str]:
    """tar 안의 특정 확장자 파일 이름(stem)을 모은다. 압축을 풀지는 않는다."""
    stems: set[str] = set()
    with tarfile.open(tar_path, "r") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith(suffix):
                stems.add(Path(member.name).stem)
    return stems


def verify(label_tar: Path, source_tar: Path) -> tuple[set[str], bool]:
    """라벨과 원천의 파일명이 실제로 매칭되는지 확인한다."""
    print(f"라벨 tar : {label_tar.name}")
    print(f"원천 tar : {source_tar.name}")

    for path in (label_tar, source_tar):
        if not path.exists():
            print(f"\n파일이 없다: {path}")
            return set(), False

    print("\n목록 읽는 중... (원천 tar 는 수 GB 라 몇 분 걸린다)")

    start = time.time()
    labels = list_members(label_tar, ".json")
    print(f"  라벨 {len(labels):,}개  ({time.time() - start:.0f}s)")

    start = time.time()
    sources = list_members(source_tar, ".jpg")
    print(f"  원천 {len(sources):,}개  ({time.time() - start:.0f}s)")

    matched = labels & sources
    print(f"\n교집합: {len(matched):,}개")

    if not matched:
        print("""
매칭되는 파일이 하나도 없다. 이 조합으로는 crop 을 만들 수 없다.

원인: 라벨 tar 와 원천 tar 가 서로 다른 해상도 세트다.
      (예: 라벨은 1280_720, 원천은 1920_1080)

해결: AI-Hub 에서 같은 해상도끼리 짝이 맞는 tar 를 받아라.
      [라벨]X_train_AAAA_BBBB_daylight_N.tar
      [원천]X_train_AAAA_BBBB_daylight_N.tar
      ↑ X 와 AAAA_BBBB 가 둘 다 같아야 한다.
""")
        return matched, False

    coverage = len(matched) / max(len(labels), 1) * 100
    print(f"라벨 기준 커버리지: {coverage:.1f}%")
    return matched, True


def extract(tar_path: Path, dest: Path, description: str) -> None:
    """tar 를 dest 에 푼다. 이미 풀려 있으면 건너뛴다."""
    dest.mkdir(parents=True, exist_ok=True)

    existing = sum(1 for _ in dest.rglob("*") if _.is_file())
    if existing > 0:
        print(f"  {description}: 이미 {existing:,}개 있음 — 건너뛴다.")
        print(f"    다시 풀려면 {dest} 를 지워라.")
        return

    print(f"  {description} 푸는 중 → {dest}")
    start = time.time()
    with tarfile.open(tar_path, "r") as archive:
        # Python 3.12+ 에서 tar 추출 보안 필터를 명시하지 않으면 경고가 뜬다.
        try:
            archive.extractall(dest, filter="data")
        except TypeError:
            archive.extractall(dest)
    print(f"    완료 ({time.time() - start:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Hub tar 검증 및 압축 해제")
    parser.add_argument("--label-tar", type=Path, default=config.LABEL_TAR)
    parser.add_argument("--source-tar", type=Path, default=config.SOURCE_TAR)
    parser.add_argument("--verify-only", action="store_true",
                        help="매칭 확인만 하고 압축은 풀지 않는다")
    args = parser.parse_args()

    print("=" * 60)
    print("AI-Hub tar 검증")
    print("=" * 60)

    matched, ok = verify(args.label_tar, args.source_tar)
    if not ok:
        sys.exit(1)

    if args.verify_only:
        print("\n검증만 수행했다. 압축을 풀려면 --verify-only 없이 다시 실행해라.")
        return

    print("\n" + "=" * 60)
    print("압축 해제")
    print("=" * 60)
    print("원천 이미지는 6GB 다. 디스크 여유를 확인해라.")

    extract(args.label_tar, config.ANNOTATION_DIR, "라벨")
    extract(args.source_tar, config.RAW_DIR, "원천")

    print("\n다음: python scripts/02_build_crops.py")


if __name__ == "__main__":
    main()
