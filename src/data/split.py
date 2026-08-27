"""
Train / Validation / Test 분할.

핸드오프 15절이 지적한 문제가 이 파일의 존재 이유다.

    교통표지판 데이터는 주행 영상에서 뽑은 프레임이 많다.
    같은 표지판을 1초 간격으로 찍으면 거의 똑같은 이미지 수십 장이 나온다.
    이걸 파일명 기준으로 무작위 셔플하면
    거의 동일한 장면이 train 과 test 에 동시에 들어간다.
    그러면 모델은 '외운 것'을 맞히면서 성능이 부풀려진다.

해결: 그룹 단위 분할.
    같은 촬영/시퀀스에서 나온 이미지들을 하나의 그룹으로 묶고,
    그룹을 통째로 train 또는 val 또는 test 중 한 곳에만 배정한다.
    개별 이미지가 아니라 그룹을 섞는 것이 핵심이다.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .manifest import SPLITS, Sample


def infer_group_id(filepath: Path, bucket: int | None = None) -> str:
    """
    crop 파일 경로에서 '같은 촬영 단위'를 나타내는 키를 뽑는다.

    crop 파일명 규칙 (scripts/02_build_crops.py 가 만든다):
        {원본stem}__{순번}.jpg      예: s01027933__2.jpg

    두 단계로 묶는다.

    1) 원본 이미지 단위
       "__" 앞부분이 원본 stem 이다. 사진 한 장에서 표지판을 여러 개 자르므로
       (실제로 한 파일에 6개가 있었다) 그것들은 반드시 같은 그룹이어야 한다.
       배경·조명·촬영조건이 똑같은 것들이 train 과 test 에 나뉘면 누수다.

    2) 주행 구간 단위 (bucket > 0 일 때)
       파일명이 s01027933 처럼 접두사+순번이다. ID 가 가까우면 같은 주행 구간에서
       연속 촬영됐을 가능성이 높다. 순번을 bucket 으로 나눈 몫으로 묶는다.
           bucket=1000 → s01027933 → "s01_27"

       주의: 이건 휴리스틱이다. AI-Hub 가 ID 를 어떻게 부여했는지 공식 문서로
       확인한 게 아니다. bucket 을 키우면 누수에는 안전하지만 그룹이 커져서
       클래스별 분할이 치우칠 수 있다. 04_build_manifest.py 출력의
       클래스별 train/val/test 분포를 보고 조정해라.
    """
    from src.data.aihub import group_id_for

    if bucket is None:
        import config

        bucket = config.GROUP_ID_BUCKET

    stem = filepath.stem
    # crop 파일명에서 "__순번" 을 떼어 원본 stem 을 복원한다.
    original_stem = stem.split("__")[0] if "__" in stem else stem

    if bucket <= 0:
        return original_stem
    return group_id_for(original_stem, bucket)


def split_by_group(
    samples: list[Sample],
    ratios: dict[str, float],
    seed: int,
) -> list[Sample]:
    """
    그룹을 통째로 배정하는 방식으로 분할한다.

    **중요: 그룹은 클래스를 가로질러 존재한다.**
    원본 사진 한 장에 circle_red 와 triangle_yellow 가 같이 찍혀 있으면,
    두 crop 은 서로 다른 클래스 폴더로 가지만 group_id 는 같다.

    따라서 클래스별로 따로 분할하면 안 된다.
    그러면 같은 사진에서 나온 crop 이 클래스에 따라 train 과 test 로 갈라져서
    막으려던 누수가 그대로 발생한다.
    (배경·조명·촬영조건이 동일한 이미지가 양쪽에 들어간다.)

    그래서 **그룹 단위로 전역 배정**한다. 대신 클래스 균형을 잃지 않도록,
    각 그룹을 넣었을 때 목표 분포에서 가장 적게 벗어나는 split 을 고르는
    탐욕적(greedy) 방식을 쓴다.

    큰 그룹부터 배정하는 이유:
        큰 덩어리를 먼저 놓아야 나중에 작은 것들로 미세 조정할 수 있다.
        작은 것부터 넣으면 마지막에 큰 덩어리가 남아 균형이 크게 깨진다.
        (greedy number partitioning 의 표준 접근)

    반환값은 split 이 채워진 새 Sample 목록이다. 입력은 변경하지 않는다.
    """
    _validate_ratios(ratios)
    rng = random.Random(seed)

    # 그룹 → 그 그룹에 속한 샘플들 (클래스 구분 없이)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)

    class_names = sorted({s.class_name for s in samples})
    class_index = {name: i for i, name in enumerate(class_names)}

    # 클래스별 전체 개수 → split 별 목표 개수
    class_totals = [0] * len(class_names)
    for sample in samples:
        class_totals[class_index[sample.class_name]] += 1

    targets = {
        split: [total * ratios[split] for total in class_totals] for split in SPLITS
    }
    current = {split: [0] * len(class_names) for split in SPLITS}

    # 정렬 후 셔플 → seed 만으로 재현 가능. 그다음 크기 내림차순으로 안정 정렬.
    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    group_ids.sort(key=lambda gid: len(groups[gid]), reverse=True)

    assignment: dict[str, str] = {}

    for group_id in group_ids:
        members = groups[group_id]

        # 이 그룹이 각 클래스를 몇 개 담고 있는지
        delta = [0] * len(class_names)
        for sample in members:
            delta[class_index[sample.class_name]] += 1

        # 각 split 에 넣어봤을 때의 '목표 대비 초과량' 을 계산해 가장 작은 쪽을 고른다.
        best_split = None
        best_cost = None
        for split in SPLITS:
            cost = 0.0
            for i in range(len(class_names)):
                after = current[split][i] + delta[i]
                target = targets[split][i]
                # 목표를 넘어선 만큼만 벌점을 준다. 목표에 못 미치는 건 괜찮다
                # (다른 그룹이 채우면 되므로).
                overflow = after - target
                if overflow > 0:
                    cost += overflow ** 2
                else:
                    # 아직 여유가 있으면 오히려 채워주는 게 좋으므로 약한 보상을 준다.
                    cost -= min(delta[i], target - current[split][i]) * 0.5
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_split = split

        assignment[group_id] = best_split
        for i in range(len(class_names)):
            current[best_split][i] += delta[i]

    result: list[Sample] = []
    for group_id, members in groups.items():
        split = assignment[group_id]
        for sample in members:
            result.append(
                Sample(
                    filepath=sample.filepath,
                    class_name=sample.class_name,
                    class_id=sample.class_id,
                    split=split,
                    group_id=sample.group_id,
                )
            )
    return result


def split_by_image(
    samples: list[Sample],
    ratios: dict[str, float],
    seed: int,
) -> list[Sample]:
    """
    이미지 단위 무작위 분할.

    주의: 이 방식은 데이터 누수에 취약하다.
    그룹 정보를 신뢰할 수 없을 때의 대조군으로만 써라.
    두 방식의 test accuracy 를 비교하면 누수가 성능을 얼마나 부풀리는지
    직접 측정할 수 있다 — 그 자체로 좋은 분석 재료다.
    """
    _validate_ratios(ratios)
    rng = random.Random(seed)

    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.class_name].append(sample)

    result: list[Sample] = []
    for class_name in sorted(by_class):
        items = sorted(by_class[class_name], key=lambda s: str(s.filepath))
        rng.shuffle(items)

        total = len(items)
        train_end = int(total * ratios["train"])
        val_end = train_end + int(total * ratios["val"])

        for index, sample in enumerate(items):
            if index < train_end:
                split = "train"
            elif index < val_end:
                split = "val"
            else:
                split = "test"
            result.append(
                Sample(
                    filepath=sample.filepath,
                    class_name=sample.class_name,
                    class_id=sample.class_id,
                    split=split,
                    group_id=sample.group_id,
                )
            )
    return result


def _validate_ratios(ratios: dict[str, float]) -> None:
    missing = set(SPLITS) - set(ratios)
    if missing:
        raise ValueError(f"분할 비율에 {sorted(missing)} 가 없다.")
    total = sum(ratios[split] for split in SPLITS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"분할 비율의 합이 1.0 이 아니다: {total}")
