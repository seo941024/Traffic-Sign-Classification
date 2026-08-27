"""
Manifest — "어떤 파일이 train/val/test 중 어디에 속하는가"를 적어둔 CSV 한 장.

왜 이런 걸 만드나:
    핸드오프 19절의 핵심 원칙은 "세 모델이 완전히 동일한 분할 데이터를 쓴다"이다.
    그런데 PyTorch 와 Keras 는 각자 폴더를 훑어 데이터를 읽는 기본 도구가 다르고
    (ImageFolder vs image_dataset_from_directory),
    파일 정렬 순서나 셔플 방식이 미묘하게 다를 수 있다.
    두 쪽에서 각각 "랜덤 분할"을 하면 seed 를 맞춰도 같은 결과가 나온다는 보장이 없다.

    그래서 분할을 딱 한 번만 수행해서 CSV 로 못박고,
    PyTorch 와 Keras 는 그 CSV 를 읽기만 한다.
    이러면 "두 프레임워크가 같은 데이터를 봤다"가 가정이 아니라 사실이 된다.

CSV 스키마:
    filepath   crop 된 이미지의 절대경로
    class_name 클래스 이름 (폴더명 기준, 예: circle_red)
    class_id   0부터 시작하는 정수 라벨. class_name 을 정렬한 순서로 부여한다.
    split      train / val / test
    group_id   데이터 누수 방지용 그룹 키 (같은 촬영/시퀀스에서 나온 것끼리 같은 값)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FIELDS = ("filepath", "class_name", "class_id", "split", "group_id")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    """이미지 한 장에 대한 기록."""

    filepath: Path
    class_name: str
    class_id: int
    split: str
    group_id: str


def write_manifest(samples: list[Sample], path: Path) -> None:
    """분할 결과를 CSV 로 고정한다. 이 파일이 실험의 기준점이 된다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "filepath": str(sample.filepath),
                    "class_name": sample.class_name,
                    "class_id": sample.class_id,
                    "split": sample.split,
                    "group_id": sample.group_id,
                }
            )


def read_manifest(path: Path, split: str | None = None) -> list[Sample]:
    """
    CSV 를 읽어 Sample 목록으로 돌려준다.
    split 을 주면 해당 분할만 걸러서 준다.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"manifest 가 없다: {path}\n"
            f"scripts/04_build_manifest.py 를 먼저 실행해서 분할을 확정해라."
        )
    if split is not None and split not in SPLITS:
        raise ValueError(f"알 수 없는 split: {split!r} (가능: {SPLITS})")

    samples: list[Sample] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["split"] != split:
                continue
            samples.append(
                Sample(
                    filepath=Path(row["filepath"]),
                    class_name=row["class_name"],
                    class_id=int(row["class_id"]),
                    split=row["split"],
                    group_id=row["group_id"],
                )
            )
    return samples


def class_names_from_manifest(path: Path) -> list[str]:
    """
    class_id 순서대로 정렬된 클래스 이름 목록.

    두 프레임워크가 **같은 정수 라벨**을 쓰게 만드는 지점이다.
    이게 어긋나면 Confusion Matrix 의 축이 서로 달라져서 비교가 무의미해진다.
    """
    mapping: dict[int, str] = {}
    for sample in read_manifest(path):
        existing = mapping.get(sample.class_id)
        if existing is not None and existing != sample.class_name:
            raise ValueError(
                f"class_id {sample.class_id} 에 이름이 두 개 붙어 있다: "
                f"{existing!r}, {sample.class_name!r} — manifest 가 깨졌다."
            )
        mapping[sample.class_id] = sample.class_name
    return [mapping[key] for key in sorted(mapping)]


def summarize(path: Path) -> str:
    """
    분할이 제대로 됐는지 사람 눈으로 확인하는 표.

    반드시 볼 것:
      - 클래스별 이미지 수가 심하게 치우쳐 있지 않은가 (클래스 불균형)
      - 어떤 클래스가 val 이나 test 에서 0장이 되지는 않았는가
        (그룹 단위로 분할하면 소수 클래스에서 실제로 일어난다)
    """
    samples = read_manifest(path)
    if not samples:
        return "manifest 가 비어 있다."

    counts: dict[str, dict[str, int]] = {}
    for sample in samples:
        counts.setdefault(sample.class_name, {}).setdefault(sample.split, 0)
        counts[sample.class_name][sample.split] += 1

    header = f"{'class':<18} {'train':>7} {'val':>7} {'test':>7} {'합계':>8}"
    lines = [header, "-" * len(header)]
    totals = {split: 0 for split in SPLITS}

    for class_name in sorted(counts):
        row = counts[class_name]
        per_split = [row.get(split, 0) for split in SPLITS]
        for split, value in zip(SPLITS, per_split):
            totals[split] += value
        lines.append(
            f"{class_name:<18} " + " ".join(f"{v:>7}" for v in per_split)
            + f" {sum(per_split):>8}"
        )

    lines.append("-" * len(header))
    lines.append(
        f"{'합계':<18} " + " ".join(f"{totals[s]:>7}" for s in SPLITS)
        + f" {sum(totals.values()):>8}"
    )

    empty = [
        (class_name, split)
        for class_name, row in counts.items()
        for split in SPLITS
        if row.get(split, 0) == 0
    ]
    if empty:
        lines.append("")
        lines.append("경고: 아래 (클래스, split) 조합이 비어 있다 —")
        for class_name, split in empty:
            lines.append(f"  - {class_name} / {split}")
        lines.append("  그룹 단위 분할에서 소수 클래스에 흔히 생긴다.")
        lines.append("  해당 클래스의 데이터를 더 모으거나 클래스 목록에서 빼는 걸 고려해라.")

    groups_per_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for sample in samples:
        groups_per_split[sample.split].add(sample.group_id)

    overlap = (
        groups_per_split["train"] & groups_per_split["val"]
        | groups_per_split["train"] & groups_per_split["test"]
        | groups_per_split["val"] & groups_per_split["test"]
    )
    lines.append("")
    if overlap:
        lines.append(
            f"경고: 같은 group_id 가 여러 split 에 걸쳐 있다 ({len(overlap)}개). "
            "데이터 누수 가능성이 있다."
        )
    else:
        lines.append("확인: 모든 group_id 가 한 split 에만 속한다 (누수 없음).")

    return "\n".join(lines)
