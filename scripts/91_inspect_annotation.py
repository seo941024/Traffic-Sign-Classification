"""
AI-Hub Annotation 구조 조사 — 핸드오프 39절의 '가장 먼저 해야 할 작업'.

핸드오프는 이렇게 못박고 있다:
    "Annotation 구조를 확인하기 전에 Crop 코드를 임의로 작성하지 않는다."

그래서 이 스크립트는 아무것도 자르지 않는다. 읽고 보고만 한다.
형식(JSON/XML/CSV)도 미리 가정하지 않고, 실제 파일을 열어서 뭐가 들어 있는지 보여준다.

이 스크립트로 답해야 할 질문 (핸드오프 39절 체크리스트):
    [ ] Annotation 파일 형식
    [ ] Annotation 내부 구조
    [ ] 이미지 파일명 연결 방식
    [ ] Bounding Box 좌표 형식   ← (x,y,w,h) 인지 (x1,y1,x2,y2) 인지, 정규화 여부
    [ ] 클래스 이름/ID
    [ ] 클래스 개수
    [ ] 원본 이미지와 Annotation 의 관계

사용법:
    python scripts/91_inspect_annotation.py
    python scripts/91_inspect_annotation.py --dir D:/AIHub_TrafficSign/annotations --samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# bbox 로 보이는 키를 찾을 때 쓰는 힌트. 대소문자 무시하고 비교한다.
BBOX_HINTS = ("bbox", "box", "x", "y", "w", "h", "left", "top", "right", "bottom",
              "xmin", "ymin", "xmax", "ymax", "width", "height", "points", "region")
CLASS_HINTS = ("class", "label", "category", "name", "type", "code", "sign")
IMAGE_HINTS = ("image", "file", "filename", "path", "img", "source")

MAX_DEPTH = 4          # 너무 깊이 들어가면 출력이 폭발한다
MAX_LIST_PREVIEW = 2   # 리스트는 앞의 몇 개만 본다


def find_annotation_files(root: Path) -> dict[str, list[Path]]:
    """확장자별로 annotation 후보 파일을 모은다."""
    if not root.exists():
        return {}

    by_extension: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        extension = path.suffix.lower()
        if extension in (".json", ".xml", ".csv", ".txt", ".yaml", ".yml"):
            by_extension.setdefault(extension, []).append(path)
    return by_extension


def describe_value(value, depth: int = 0) -> str:
    """값 하나를 타입과 예시로 요약한다. 실제 값을 통째로 찍지는 않는다."""
    indent = "  " * depth

    if isinstance(value, dict):
        if depth >= MAX_DEPTH:
            return f"dict ({len(value)} keys) ...더 깊음"
        lines = [f"dict ({len(value)} keys)"]
        for key, item in value.items():
            lines.append(f"{indent}  {key}: {describe_value(item, depth + 1)}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return "list (비어 있음)"
        if depth >= MAX_DEPTH:
            return f"list ({len(value)} items) ...더 깊음"
        lines = [f"list ({len(value)} items)"]
        for index, item in enumerate(value[:MAX_LIST_PREVIEW]):
            lines.append(f"{indent}  [{index}]: {describe_value(item, depth + 1)}")
        if len(value) > MAX_LIST_PREVIEW:
            lines.append(f"{indent}  ... 외 {len(value) - MAX_LIST_PREVIEW}개")
        return "\n".join(lines)

    if isinstance(value, str):
        preview = value if len(value) <= 60 else value[:57] + "..."
        return f"str  = {preview!r}"

    return f"{type(value).__name__}  = {value!r}"


def collect_keys(value, prefix: str = "", found: set[str] | None = None) -> set[str]:
    """중첩 구조 전체를 훑어 'a.b.c' 형태의 키 경로를 모은다."""
    found = found if found is not None else set()

    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.add(path)
            collect_keys(item, path, found)
    elif isinstance(value, list) and value:
        collect_keys(value[0], f"{prefix}[]", found)

    return found


def guess_roles(keys: set[str]) -> dict[str, list[str]]:
    """키 이름만 보고 bbox / class / image 후보를 추려낸다. 어디까지나 추정이다."""

    def matches(hints: tuple[str, ...]) -> list[str]:
        result = []
        for key in sorted(keys):
            leaf = key.split(".")[-1].replace("[]", "").lower()
            if any(hint == leaf or hint in leaf for hint in hints):
                result.append(key)
        return result

    return {
        "bounding box 후보": matches(BBOX_HINTS),
        "class / label 후보": matches(CLASS_HINTS),
        "image 참조 후보": matches(IMAGE_HINTS),
    }


def inspect_json(path: Path) -> set[str]:
    print(f"\n{'-' * 70}")
    print(f"파일: {path}")
    print(f"크기: {path.stat().st_size / 1024:.1f} KB")
    print("-" * 70)

    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except UnicodeDecodeError:
        # AI-Hub 한국어 데이터는 cp949 로 저장된 경우가 있다.
        with path.open(encoding="cp949") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as error:
        print(f"  JSON 파싱 실패: {error}")
        print("  → JSON Lines 형식이거나 인코딩 문제일 수 있다. 파일 앞부분을 직접 열어봐라.")
        return set()

    print("\n[구조]")
    print(describe_value(data))

    keys = collect_keys(data)
    print(f"\n[전체 키 경로]  ({len(keys)}개)")
    for key in sorted(keys):
        print(f"  {key}")

    print("\n[역할 추정]")
    for role, candidates in guess_roles(keys).items():
        print(f"  {role}: {candidates if candidates else '(못 찾음)'}")

    return keys


def inspect_xml(path: Path) -> None:
    import xml.etree.ElementTree as ET

    print(f"\n{'-' * 70}")
    print(f"파일: {path}")
    print("-" * 70)

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        print(f"  XML 파싱 실패: {error}")
        return

    print(f"  루트 태그: <{root.tag}>")

    tags: Counter[str] = Counter()

    def walk(element, depth: int = 0) -> None:
        tags[element.tag] += 1
        if depth < MAX_DEPTH:
            text = (element.text or "").strip()
            preview = f" = {text[:40]!r}" if text else ""
            print(f"  {'  ' * depth}<{element.tag}>{preview}")
            for child in list(element)[:MAX_LIST_PREVIEW + 1]:
                walk(child, depth + 1)

    walk(root)
    print(f"\n  태그 빈도: {dict(tags.most_common(15))}")


def inspect_tabular(path: Path, rows: int = 5) -> None:
    print(f"\n{'-' * 70}")
    print(f"파일: {path}")
    print("-" * 70)
    try:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= rows:
                    break
                print(f"  {index}: {line.rstrip()}")
    except UnicodeDecodeError:
        with path.open(encoding="cp949") as handle:
            for index, line in enumerate(handle):
                if index >= rows:
                    break
                print(f"  {index}: {line.rstrip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Hub Annotation 구조 조사 (읽기 전용)")
    parser.add_argument("--dir", type=Path, default=config.ANNOTATION_DIR,
                        help="annotation 이 있는 디렉터리")
    parser.add_argument("--samples", type=int, default=3,
                        help="확장자마다 몇 개 파일을 열어볼지")
    args = parser.parse_args()

    print("=" * 70)
    print("AI-Hub Annotation 구조 조사")
    print("=" * 70)
    print(f"대상 디렉터리: {args.dir}")

    if not args.dir.exists():
        print("\n디렉터리가 없다.")
        print("데이터 다운로드가 끝나면 config.py 의 ANNOTATION_DIR 을 실제 경로로 맞추고 다시 실행해라.")
        print(f"  현재 설정값: {config.ANNOTATION_DIR}")
        return

    by_extension = find_annotation_files(args.dir)
    if not by_extension:
        print("\nannotation 으로 보이는 파일을 못 찾았다.")
        print("압축이 아직 안 풀렸거나 하위 폴더 구조가 예상과 다를 수 있다.")
        return

    print("\n[발견한 파일]")
    for extension, paths in sorted(by_extension.items()):
        total_mb = sum(p.stat().st_size for p in paths) / (1024 ** 2)
        print(f"  {extension:<6} {len(paths):>7,}개   합계 {total_mb:>10.1f} MB")

    all_keys: set[str] = set()

    for extension, paths in sorted(by_extension.items()):
        print(f"\n\n{'=' * 70}")
        print(f"{extension} 샘플 (최대 {args.samples}개)")
        print("=" * 70)

        for path in paths[: args.samples]:
            if extension == ".json":
                all_keys |= inspect_json(path)
            elif extension == ".xml":
                inspect_xml(path)
            else:
                inspect_tabular(path)

    if all_keys:
        print(f"\n\n{'=' * 70}")
        print("종합 — 확인한 JSON 파일 전체의 키 경로")
        print("=" * 70)
        for role, candidates in guess_roles(all_keys).items():
            print(f"\n{role}:")
            for candidate in candidates:
                print(f"  {candidate}")

    print(f"\n\n{'=' * 70}")
    print("다음 할 일")
    print("=" * 70)
    print("""
  1. 위 출력에서 bbox 좌표 형식을 직접 확인해라.
     (x, y, w, h) 인가, (x1, y1, x2, y2) 인가?
     0~1 로 정규화된 값인가, 픽셀 값인가?
     좌표계 원점이 좌상단인가?
     → 이걸 잘못 읽으면 crop 이 엉뚱한 데를 자르는데, 코드는 에러 없이 잘 돈다.
       그래서 반드시 crop 결과를 눈으로 봐야 한다.

  2. 클래스 체계를 확인해라. 이름인가 코드인가? 몇 종류인가?
     → config.py 의 CLASS_NAMES 에 실제 클래스명을 채워라.
       임의로 지어내지 말고 annotation 에 있는 값을 그대로 써라.

  3. annotation 파일과 이미지 파일이 어떻게 연결되는지 확인해라.
     파일명이 1:1 로 대응하는가? 한 annotation 이 여러 이미지를 담는가?

  4. 그 다음에야 crop 코드를 짠다. 그 전에는 짜지 않는다.
""")


if __name__ == "__main__":
    main()
