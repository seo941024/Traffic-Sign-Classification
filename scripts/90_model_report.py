"""
모델 구조 리포트 — 학습 없이, 데이터 없이 지금 당장 실행 가능.

이 프로젝트가 답해야 할 질문의 절반은 학습 전에 이미 답이 나와 있다.

    "각 Layer 를 지날 때 Tensor 모양이 어떻게 변하는가?"
    "파라미터 수는 왜 모델마다 다르게 증가하는가?"
    "Flatten 을 하면 무엇이 문제인가?"

이건 전부 구조가 결정하는 것이지 데이터가 결정하는 게 아니다.
그래서 AI-Hub 다운로드를 기다리는 동안에도 이 스크립트는 돌아간다.

사용법:
    python scripts/90_model_report.py
    python scripts/90_model_report.py --classes 10 --framework torch
    python scripts/90_model_report.py --save          # results/reports/ 에 저장
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.analysis.model_stats import (  # noqa: E402
    ModelStat,
    collect_keras_stats,
    collect_torch_stats,
    render_comparison,
    render_stat_table,
)


def report_torch(num_classes: int) -> list[ModelStat]:
    try:
        from src.torch_impl.models import build_model
    except ImportError as error:
        print(f"PyTorch 를 못 불러왔다: {error}\n")
        return []

    stats: list[ModelStat] = []
    for name in config.MODEL_NAMES:
        model = build_model(name, num_classes=num_classes)
        stat = collect_torch_stats(model, config.input_shape("torch"), name)
        stats.append(stat)
        print(render_stat_table(stat))
        print()
    return stats


def report_keras(num_classes: int) -> list[ModelStat]:
    try:
        from src.keras_impl.models import build_model
    except ImportError as error:
        print(f"TensorFlow/Keras 를 못 불러왔다: {error}\n")
        return []

    stats: list[ModelStat] = []
    for name in config.MODEL_NAMES:
        model = build_model(name, num_classes=num_classes)
        stat = collect_keras_stats(model, name)
        stats.append(stat)
        print(render_stat_table(stat))
        print()

        # Keras 의 강점: 같은 정보를 표준 형식으로도 보여준다.
        print(f"--- Keras 기본 summary(): {name.upper()} ---")
        model.summary()
        print()
    return stats


def cross_framework_check(torch_stats: list[ModelStat], keras_stats: list[ModelStat]) -> None:
    """
    두 프레임워크의 파라미터 수가 일치하는지 확인한다.

    일치해야 정상이다. 파라미터 수는 프레임워크가 아니라 구조가 결정하기 때문이다.
    어긋난다면 두 구현의 층 구성이 실제로 다르다는 뜻이고,
    그 상태로 성능을 비교하면 결론이 틀린다.
    """
    if not torch_stats or not keras_stats:
        return

    print("=" * 70)
    print("프레임워크 교차 검증 — 두 구현이 정말 같은 모델인가?")
    print("=" * 70)

    by_name_torch = {stat.name: stat for stat in torch_stats}
    by_name_keras = {stat.name: stat for stat in keras_stats}

    header = f"{'Model':<8} {'PyTorch':>14} {'Keras':>14} {'차이':>12}   판정"
    print(header)
    print("-" * len(header))

    all_match = True
    for name in config.MODEL_NAMES:
        torch_stat = by_name_torch.get(name)
        keras_stat = by_name_keras.get(name)
        if not torch_stat or not keras_stat:
            continue

        difference = torch_stat.total_params - keras_stat.total_params
        matched = difference == 0
        all_match &= matched
        print(
            f"{name:<8} {torch_stat.total_params:>14,} {keras_stat.total_params:>14,} "
            f"{difference:>12,}   {'일치' if matched else '불일치'}"
        )

    print()
    if all_match:
        print("두 구현의 파라미터 수가 모두 일치한다. 같은 구조를 서로 다른 문법으로 쓴 것이다.")
        print("따라서 두 프레임워크의 결과 차이는 구조가 아니라 초기화/연산 순서 등에서 온다.")
    else:
        print("불일치가 있다. 두 구현의 층 구성이 실제로 다르다.")
        print("이 상태로 비교하면 결론이 틀린다. 위 표에서 차이 나는 모델을 먼저 맞춰라.")
    print()


def explain(stats: list[ModelStat]) -> None:
    """숫자만 찍고 끝내지 않고, 그 숫자가 무슨 뜻인지 붙여준다."""
    if not stats:
        return

    by_name = {stat.name: stat for stat in stats}
    print("=" * 70)
    print("이 숫자가 말하는 것")
    print("=" * 70)

    ann, dnn, cnn = (by_name.get(name) for name in ("ann", "dnn", "cnn"))

    print(f"""
1. Flatten 이 만드는 입력 차원
   {config.IMAGE_SIZE} x {config.IMAGE_SIZE} x {config.IMAGE_CHANNELS} = {config.FLATTENED_INPUT_DIM:,} 개의 입력 특징.
   ANN/DNN 은 이 {config.FLATTENED_INPUT_DIM:,} 개를 첫 Dense Layer 에 전부 연결한다.
   Dense 의 파라미터는 (입력수 x 출력수) 이므로 입력이 크면 그대로 비례해 커진다.
""")

    if ann and dnn:
        ratio = dnn.total_params / ann.total_params
        print(f"""2. ANN → DNN: 층을 늘리면 파라미터가 어떻게 되나
   ANN {ann.total_params:>12,}  (Hidden Dense {len(config.ANN_HIDDEN_UNITS)}층)
   DNN {dnn.total_params:>12,}  (Hidden Dense {len(config.DNN_HIDDEN_UNITS)}층)  → {ratio:.1f}배

   주의할 점: 층을 4배로 늘려서 파라미터가 는 게 아니다.
   첫 Dense 의 폭이 {config.ANN_HIDDEN_UNITS[0]} → {config.DNN_HIDDEN_UNITS[0]} 으로 넓어진 것이 지배적이다.
   '깊이'보다 '첫 층의 폭'이 파라미터를 결정한다.
""")

    if cnn and dnn:
        ratio = dnn.total_params / cnn.total_params
        by_kind = cnn.params_by_kind()
        conv_params = sum(v for k, v in by_kind.items() if "Conv" in k)
        dense_params = sum(v for k, v in by_kind.items() if k in ("Linear", "Dense"))
        print(f"""3. CNN 은 왜 파라미터가 적은가
   DNN {dnn.total_params:>12,}
   CNN {cnn.total_params:>12,}  → DNN 이 CNN 의 {ratio:.1f}배

   CNN 내부를 뜯어보면:
     Conv  계열 {conv_params:>12,}   ({conv_params / cnn.total_params * 100:.1f}%)
     Dense 계열 {dense_params:>12,}   ({dense_params / cnn.total_params * 100:.1f}%)

   Conv 는 커널 하나를 이미지 전체에 재사용한다(weight sharing).
   3x3 커널은 입력 해상도가 64든 256이든 파라미터가 9개(x채널)로 같다.
   반면 Dense 는 입력이 커지면 파라미터가 그대로 비례해 늘어난다.
   이것이 두 층의 파라미터 증가 방식이 근본적으로 다른 이유다.

   그리고 CNN 조차 파라미터 대부분은 Dense 에 있다.
   차이는 Pooling 으로 {config.IMAGE_SIZE}x{config.IMAGE_SIZE} 를 미리 줄여놓고 Flatten 했다는 것뿐이다.
   "CNN 은 파라미터가 적다"보다
   "CNN 은 Dense 에 넣기 전에 입력을 줄여놨다"가 정확한 설명이다.
""")

    print(f"""4. 파라미터가 많다고 성능이 좋은 게 아니다
   위 숫자대로면 DNN 이 가장 크고 CNN 이 가장 작다.
   그런데 이미지 분류 성능은 보통 CNN 이 가장 좋다.

   파라미터를 어디에 쓰느냐가 개수보다 중요하다는 뜻이다.
   DNN 은 "어느 픽셀과 어느 픽셀이 관련 있는지"를 데이터로부터 처음부터 배워야 한다.
   CNN 은 "가까운 픽셀끼리 관련 있다"를 구조에 이미 넣어두고 시작한다.
   이 사전 지식이 CNN 이 적은 파라미터로 더 잘하는 이유다.

   → 실제로 그런지는 학습을 돌려봐야 안다.
     예상과 다르게 나오면 그 원인을 분석하는 게 이 프로젝트의 목적이다. (핸드오프 24절)
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="ANN/DNN/CNN 구조 리포트 (학습 없음)")
    parser.add_argument(
        "--classes", type=int, default=None,
        help="클래스 수. 생략하면 config.NUM_CLASSES, 그것도 0이면 10으로 가정한다.",
    )
    parser.add_argument("--framework", choices=("torch", "keras", "both"), default="both")
    parser.add_argument("--save", action="store_true", help="results/reports/ 에 저장")
    args = parser.parse_args()

    num_classes = args.classes or config.NUM_CLASSES or len(config.CLASS_NAMES)
    assumed = args.classes is None and config.NUM_CLASSES == 0

    print("=" * 70)
    print("ANN / DNN / CNN 구조 리포트")
    print("=" * 70)
    print(f"입력 크기   : {config.IMAGE_SIZE} x {config.IMAGE_SIZE} x {config.IMAGE_CHANNELS}")
    print(f"Flatten 차원: {config.FLATTENED_INPUT_DIM:,}")
    print(f"클래스 수   : {num_classes}"
          + ("  (가정값 — Annotation 확인 후 config.CLASS_NAMES 를 채워라)" if assumed else ""))
    print()

    torch_stats: list[ModelStat] = []
    keras_stats: list[ModelStat] = []

    if args.framework in ("torch", "both"):
        print("#" * 70)
        print("# PyTorch — forward hook 으로 실행 중 실제 shape 를 추적")
        print("#" * 70)
        print()
        torch_stats = report_torch(num_classes)
        if torch_stats:
            print(render_comparison(torch_stats))
            print()

    if args.framework in ("keras", "both"):
        print("#" * 70)
        print("# Keras — 층 구성 시점에 확정된 shape 를 조회")
        print("#" * 70)
        print()
        keras_stats = report_keras(num_classes)
        if keras_stats:
            print(render_comparison(keras_stats))
            print()

    if args.framework == "both":
        cross_framework_check(torch_stats, keras_stats)

    explain(torch_stats or keras_stats)

    if args.save:
        config.ensure_dirs()
        output = config.REPORT_DIR / "model_structure_report.txt"
        lines = []
        for stat in torch_stats + keras_stats:
            lines.append(render_stat_table(stat))
            lines.append("")
        if torch_stats:
            lines.append(render_comparison(torch_stats))
        if keras_stats:
            lines.append(render_comparison(keras_stats))
        output.write_text("\n".join(lines), encoding="utf-8")
        print(f"저장: {output}")


if __name__ == "__main__":
    main()
