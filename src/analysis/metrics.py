"""
평가 지표.

핸드오프 21절: "Accuracy 하나만 사용하지 않는다."

Accuracy 하나로는 알 수 없는 것:
    클래스 불균형이 있으면 다수 클래스만 맞혀도 Accuracy 가 높게 나온다.
    어떤 클래스를 어떤 클래스로 착각하는지도 안 보인다.

특히 이 프로젝트에서 중요한 것은 Confusion Matrix 다.
    circle_red ↔ circle_blue  — 형태는 같고 색만 다르다. 픽셀 평균색으로도 구분 가능.
    circle_* ↔ triangle_*     — 색은 같고 형태가 다르다. 공간 정보가 필요.
ANN 이 이런 쌍에서 특히 많이 틀린다면, 그건 "공간 정보를 못 쓴다"는 주장의 직접적인 증거가 된다.
학습이 끝난 뒤 반드시 이 쌍들을 확인해라.

이 모듈은 프레임워크에 의존하지 않는다.
PyTorch 든 Keras 든 예측을 numpy 배열로 뽑아서 넣으면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalResult:
    """모델 하나의 평가 결과."""

    model_name: str
    framework: str
    accuracy: float
    macro_f1: float
    confusion: np.ndarray          # (num_classes, num_classes), [정답][예측]
    class_names: list[str]
    per_class_accuracy: dict[str, float]

    # 핸드오프 21절이 요구하는 나머지 항목들. 학습 스크립트가 채운다.
    total_params: int | None = None
    train_seconds: float | None = None
    inference_ms_per_image: float | None = None
    model_size_mb: float | None = None


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    행 = 실제 클래스, 열 = 예측 클래스.

    대각선이 맞힌 것이고, 대각선을 벗어난 값이 혼동이다.
    (i, j) 가 크다는 건 "실제로는 i 인데 j 라고 답했다"가 많다는 뜻이다.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[int(true_label), int(predicted_label)] += 1
    return matrix


def per_class_metrics(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """
    Confusion Matrix 에서 클래스별 precision / recall / f1 을 뽑는다.

        precision = 이 클래스라고 답한 것 중 실제로 맞은 비율  (열 기준)
        recall    = 실제 이 클래스인 것 중 맞힌 비율            (행 기준)

    표지판에서 둘의 의미가 다르다:
        recall 이 낮다    = 이 표지판을 자주 놓친다
        precision 이 낮다 = 다른 걸 이 표지판이라고 자주 착각한다
    """
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_total = matrix.sum(axis=0).astype(np.float64)
    actual_total = matrix.sum(axis=1).astype(np.float64)

    # 등장하지 않은 클래스는 0으로 나누게 되므로 divide 로 안전하게 처리한다.
    precision = np.divide(
        true_positive, predicted_total,
        out=np.zeros_like(true_positive), where=predicted_total > 0,
    )
    recall = np.divide(
        true_positive, actual_total,
        out=np.zeros_like(true_positive), where=actual_total > 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator,
        out=np.zeros_like(precision), where=denominator > 0,
    )
    return {"precision": precision, "recall": recall, "f1": f1, "support": actual_total}


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    framework: str,
) -> EvalResult:
    """예측 결과를 EvalResult 로 정리한다."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"길이 불일치: y_true {len(y_true)}, y_pred {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("평가할 샘플이 없다.")

    num_classes = len(class_names)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    metrics = per_class_metrics(matrix)

    accuracy = float(np.trace(matrix) / matrix.sum())

    # macro F1 은 클래스별 F1 의 단순 평균이다.
    # 클래스 크기를 무시하므로 소수 클래스의 성능이 그대로 반영된다.
    # 클래스 불균형이 있을 때 accuracy 보다 정직한 지표다.
    macro_f1 = float(np.mean(metrics["f1"]))

    per_class_accuracy = {
        name: float(metrics["recall"][index]) for index, name in enumerate(class_names)
    }

    return EvalResult(
        model_name=model_name,
        framework=framework,
        accuracy=accuracy,
        macro_f1=macro_f1,
        confusion=matrix,
        class_names=class_names,
        per_class_accuracy=per_class_accuracy,
    )


def top_confusions(result: EvalResult, top_k: int = 10) -> list[tuple[str, str, int]]:
    """
    가장 많이 헷갈린 (실제, 예측) 쌍을 뽑는다. 대각선은 제외한다.

    이 프로젝트에서 가장 중요한 출력이다.
    ANN / DNN / CNN 세 모델의 결과를 나란히 놓고 봐라.
    ANN 에서만 circle ↔ triangle 혼동이 크게 나온다면
    "공간 정보를 못 쓴다"는 설명에 실제 증거가 붙는다.
    """
    pairs: list[tuple[str, str, int]] = []
    matrix = result.confusion
    for true_index in range(len(result.class_names)):
        for predicted_index in range(len(result.class_names)):
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            if count > 0:
                pairs.append(
                    (
                        result.class_names[true_index],
                        result.class_names[predicted_index],
                        count,
                    )
                )
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:top_k]


def baselines(y_true: np.ndarray, num_classes: int) -> dict[str, float]:
    """
    비교 기준선. 이게 없으면 정확도 숫자를 해석할 수 없다.

        무작위      = 1 / 클래스수
        최빈 클래스 = 가장 많은 클래스의 비율

    **최빈 클래스 baseline 을 못 넘는 모델은 아무것도 학습하지 못한 것이다.**
    "무작위가 16.7%인데 CNN 이 92%" 가
    "CNN 이 92%" 보다 훨씬 강한 진술이다.
    """
    counts = np.bincount(y_true.astype(int), minlength=num_classes)
    total = counts.sum()
    return {
        "random": 1.0 / num_classes,
        "majority": float(counts.max() / total) if total else 0.0,
    }


def shape_vs_color_confusion(result: EvalResult) -> dict[str, float]:
    """
    혼동을 '형태가 다른 것끼리'와 '색만 다른 것끼리'로 나눈다.

    이 프로젝트의 핵심 논증이 여기서 나온다.

    클래스 이름이 "{shape}_{color}" 형식이므로 틀린 예측을 세 종류로 가른다.
        shape_error : 형태를 틀림  (circle → triangle)
        color_error : 색을 틀림    (circle_red → circle_blue)
        both_error  : 둘 다 틀림

    가설:
        ANN/DNN 은 픽셀 평균색만으로도 색 구분은 어느 정도 한다.
        하지만 윤곽선(형태)은 인접 픽셀 관계를 알아야 하는데
        Flatten 이 그 정보를 없앴으므로 형태에서 무너질 것이다.

        → ANN 의 shape_error 비율이 CNN 보다 크게 높다면
          "공간 정보를 활용하지 못한다"는 주장에 직접적인 수치 증거가 된다.

    예상과 다르게 나와도 좋다. 그 경우 원인을 분석하는 것이 이 프로젝트의 목적이다.
    """
    matrix = result.confusion
    names = result.class_names

    # "circle_red" → ("circle", "red"). 형식이 다르면 분석을 건너뛴다.
    parsed = []
    for name in names:
        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            return {}
        parsed.append((parts[0], parts[1]))

    shape_error = color_error = both_error = 0
    for true_index in range(len(names)):
        for pred_index in range(len(names)):
            if true_index == pred_index:
                continue
            count = int(matrix[true_index, pred_index])
            if not count:
                continue

            true_shape, true_color = parsed[true_index]
            pred_shape, pred_color = parsed[pred_index]

            if true_shape != pred_shape and true_color != pred_color:
                both_error += count
            elif true_shape != pred_shape:
                shape_error += count
            else:
                color_error += count

    total_errors = shape_error + color_error + both_error
    if not total_errors:
        return {"shape_error": 0.0, "color_error": 0.0, "both_error": 0.0, "total_errors": 0}

    return {
        "shape_error": shape_error / total_errors,
        "color_error": color_error / total_errors,
        "both_error": both_error / total_errors,
        "total_errors": total_errors,
    }


def render_report(result: EvalResult) -> str:
    """클래스별 지표와 상위 혼동 쌍을 사람이 읽을 수 있게 만든다."""
    metrics = per_class_metrics(result.confusion)

    lines = [
        f"[{result.model_name.upper()}]  framework={result.framework}",
        f"  Accuracy : {result.accuracy:.4f}",
        f"  Macro F1 : {result.macro_f1:.4f}",
        "",
    ]

    header = f"  {'class':<18} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>9}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for index, name in enumerate(result.class_names):
        lines.append(
            f"  {name:<18} {metrics['precision'][index]:>10.4f} "
            f"{metrics['recall'][index]:>10.4f} {metrics['f1'][index]:>10.4f} "
            f"{int(metrics['support'][index]):>9,}"
        )

    confusions = top_confusions(result)
    if confusions:
        lines.append("")
        lines.append("  가장 많이 헷갈린 쌍 (실제 → 예측):")
        for actual, predicted, count in confusions:
            lines.append(f"    {actual:<18} → {predicted:<18} {count:>6,}회")

    breakdown = shape_vs_color_confusion(result)
    if breakdown and breakdown.get("total_errors"):
        lines.append("")
        lines.append(f"  오류 유형 분해 (총 {int(breakdown['total_errors']):,}건):")
        lines.append(f"    형태를 틀림   {breakdown['shape_error']:>7.1%}"
                     "   ← 공간 정보가 필요한 판단")
        lines.append(f"    색을 틀림     {breakdown['color_error']:>7.1%}"
                     "   ← 픽셀 평균색으로도 가능한 판단")
        lines.append(f"    둘 다 틀림    {breakdown['both_error']:>7.1%}")

    return "\n".join(lines)
