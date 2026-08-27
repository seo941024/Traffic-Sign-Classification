"""
세 모델의 결과를 한 표로 모으는 곳.

핸드오프 21절이 요구하는 항목을 전부 한 자리에 놓는다.
    Accuracy / Loss / Confusion Matrix / Parameter 수 / 학습 시간 / 추론 시간 / Model size

정확도만 나열하면 "CNN 이 제일 높다"로 끝나버린다.
파라미터 수와 학습 시간을 같이 놓아야
"CNN 은 파라미터가 제일 적은데도 제일 높다"는 진짜 결론이 보인다.
그게 이 프로젝트가 하려는 이야기다.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .metrics import EvalResult

COMPARISON_FIELDS = (
    "model", "framework", "accuracy", "macro_f1", "total_params",
    "model_size_mb", "train_seconds", "inference_ms_per_image",
)


def render_table(results: list[EvalResult]) -> str:
    """비교표. 파라미터 대비 성능을 함께 보여준다."""
    if not results:
        return "(비교할 결과가 없다)"

    header = (
        f"{'Model':<7} {'FW':<7} {'Acc':>8} {'MacroF1':>9} {'Params':>13} "
        f"{'Size(MB)':>9} {'학습(s)':>9} {'추론(ms)':>9}"
    )
    lines = ["모델 비교 (동일 데이터 / 동일 학습 조건)", header, "-" * len(header)]

    for result in results:
        params = f"{result.total_params:,}" if result.total_params else "-"
        size = f"{result.model_size_mb:.2f}" if result.model_size_mb else "-"
        train = f"{result.train_seconds:.1f}" if result.train_seconds else "-"
        infer = (
            f"{result.inference_ms_per_image:.3f}"
            if result.inference_ms_per_image else "-"
        )
        lines.append(
            f"{result.model_name:<7} {result.framework:<7} {result.accuracy:>8.4f} "
            f"{result.macro_f1:>9.4f} {params:>13} {size:>9} {train:>9} {infer:>9}"
        )

    lines.append("-" * len(header))
    lines.extend(_interpret(results))
    return "\n".join(lines)


def _interpret(results: list[EvalResult]) -> list[str]:
    """표를 읽는 법. 결론을 대신 내려주지는 않는다 — 어디를 보라고만 알려준다."""
    lines = ["", "읽는 법:"]

    best = max(results, key=lambda r: r.accuracy)
    lines.append(f"  - 정확도 1위: {best.model_name.upper()} ({best.accuracy:.4f})")

    with_params = [r for r in results if r.total_params]
    if with_params:
        smallest = min(with_params, key=lambda r: r.total_params)
        lines.append(
            f"  - 파라미터 최소: {smallest.model_name.upper()} ({smallest.total_params:,})"
        )
        if best.total_params and smallest.model_name == best.model_name:
            lines.append(
                "  - 파라미터가 가장 적은 모델이 정확도도 1위다."
                " 파라미터 개수가 성능을 결정하지 않는다는 직접적인 증거다."
            )
        elif best.total_params and smallest.total_params:
            ratio = best.total_params / smallest.total_params
            lines.append(
                f"    1위 모델은 최소 모델의 {ratio:.1f}배 파라미터를 쓴다."
                " 그만큼의 값어치를 했는지 판단해라."
            )

    lines.append(
        "  - Confusion Matrix 를 반드시 같이 봐라. 특히 형태가 다른 쌍(circle↔triangle)과 색만 다른 쌍(circle_red↔circle_blue)을 구분해서 봐라."
        " 어떤 모델이 어떤 쌍에서 무너지는지가 구조 차이를 가장 잘 보여준다."
    )
    lines.append(
        "  - 예상과 다른 결과가 나왔다면 그게 더 좋은 분석 재료다. (핸드오프 24절)"
    )
    return lines


def save_csv(results: list[EvalResult], path: Path) -> None:
    """포트폴리오 문서에 표로 붙이기 좋게 CSV 로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result.model_name,
                    "framework": result.framework,
                    "accuracy": round(result.accuracy, 6),
                    "macro_f1": round(result.macro_f1, 6),
                    "total_params": result.total_params or "",
                    "model_size_mb": (
                        round(result.model_size_mb, 4) if result.model_size_mb else ""
                    ),
                    "train_seconds": (
                        round(result.train_seconds, 2) if result.train_seconds else ""
                    ),
                    "inference_ms_per_image": (
                        round(result.inference_ms_per_image, 4)
                        if result.inference_ms_per_image else ""
                    ),
                }
            )
