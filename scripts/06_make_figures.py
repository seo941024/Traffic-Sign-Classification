"""
포트폴리오용 그림 생성 — 학습 결과 JSON 에서 그림을 다시 그린다.

05_run_experiment.py 가 저장한 JSON 만 있으면 되므로 학습을 다시 돌릴 필요가 없다.
그림 스타일만 고치고 싶을 때 몇 시간짜리 학습을 반복하지 않아도 된다.

만드는 그림:
    1. 학습 곡선 (세 모델의 train/val loss 를 한 장에)
       → 과적합 여부를 본다. DNN 이 train loss 는 낮은데 val loss 가 올라간다면
         파라미터가 많아 외우고 있다는 뜻이다.
    2. Confusion Matrix (모델별)
       → speed 계열, 방향 계열 혼동을 확인한다.
    3. 파라미터 수 대비 정확도
       → 이 프로젝트의 결론을 한 장으로 보여주는 그림이다.

사용법:
    python scripts/06_make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


def load_runs() -> list[dict]:
    """저장된 실행 결과를 모두 읽는다."""
    if not config.HISTORY_DIR.exists():
        return []
    runs = []
    for path in sorted(config.HISTORY_DIR.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"  건너뜀 (파싱 실패): {path.name}")
    return runs


def plot_learning_curves(runs: list[dict], plt) -> None:
    """세 모델의 학습 곡선을 프레임워크별로 한 장에 겹쳐 그린다."""
    frameworks = sorted({run["framework"] for run in runs})

    for framework in frameworks:
        subset = [r for r in runs if r["framework"] == framework]
        if not subset:
            continue

        figure, axes = plt.subplots(1, 2, figsize=(13, 5))

        for run in subset:
            history = run.get("history") or {}
            label = run["model"].upper()

            train_loss = history.get("train_loss") or []
            val_loss = history.get("val_loss") or []
            train_acc = history.get("train_acc") or []
            val_acc = history.get("val_acc") or []

            epochs = range(1, len(train_loss) + 1)
            line, = axes[0].plot(epochs, train_loss, label=f"{label} train")
            axes[0].plot(
                range(1, len(val_loss) + 1), val_loss,
                linestyle="--", color=line.get_color(), label=f"{label} val",
            )

            line2, = axes[1].plot(range(1, len(train_acc) + 1), train_acc, label=f"{label} train")
            axes[1].plot(
                range(1, len(val_acc) + 1), val_acc,
                linestyle="--", color=line2.get_color(), label=f"{label} val",
            )

        axes[0].set_title(f"Loss ({framework})")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].set_title(f"Accuracy ({framework})")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        figure.suptitle(
            "실선 = train, 점선 = val   |   "
            "둘의 간격이 벌어지면 과적합",
            fontsize=9,
        )
        figure.tight_layout()

        output = config.FIGURE_DIR / f"learning_curves_{framework}.png"
        figure.savefig(output, dpi=150)
        plt.close(figure)
        print(f"  저장: {output}")


def plot_confusion(runs: list[dict], plt, np) -> None:
    """모델별 Confusion Matrix. 행 기준으로 정규화해서 클래스 크기 차이를 없앤다."""
    for run in runs:
        confusion = np.array(run.get("confusion") or [])
        if confusion.size == 0:
            continue

        class_names = run.get("class_names") or []
        row_totals = confusion.sum(axis=1, keepdims=True)
        # 등장하지 않은 클래스는 0으로 나누게 되므로 안전하게 처리한다.
        normalized = np.divide(
            confusion, row_totals,
            out=np.zeros(confusion.shape, dtype=float), where=row_totals > 0,
        )

        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)

        axis.set_xticks(range(len(class_names)))
        axis.set_yticks(range(len(class_names)))
        axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        axis.set_yticklabels(class_names, fontsize=8)
        axis.set_xlabel("예측")
        axis.set_ylabel("실제")
        axis.set_title(
            f"{run['model'].upper()} ({run['framework']})  "
            f"acc={run.get('accuracy', 0):.3f}"
        )

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                value = normalized[i, j]
                if value > 0.005:
                    axis.text(
                        j, i, f"{value:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if value > 0.5 else "black",
                    )

        figure.colorbar(image, ax=axis, fraction=0.046)
        figure.tight_layout()

        output = config.FIGURE_DIR / f"confusion_{run['framework']}_{run['model']}.png"
        figure.savefig(output, dpi=150)
        plt.close(figure)
        print(f"  저장: {output}")


def plot_params_vs_accuracy(runs: list[dict], plt) -> None:
    """
    파라미터 수 대비 정확도 — 이 프로젝트의 결론을 한 장으로 보여주는 그림.

    CNN 이 왼쪽 위(파라미터 적고 정확도 높음)에 있으면
    "파라미터 개수가 성능을 결정하지 않는다"는 주장이 그림 하나로 설명된다.
    """
    usable = [r for r in runs if r.get("total_params")]
    if not usable:
        return

    figure, axis = plt.subplots(figsize=(8, 6))
    markers = {"torch": "o", "keras": "s"}

    for run in usable:
        axis.scatter(
            run["total_params"], run["accuracy"],
            s=140, alpha=0.75,
            marker=markers.get(run["framework"], "o"),
            label=f"{run['model'].upper()} ({run['framework']})",
        )
        axis.annotate(
            run["model"].upper(),
            (run["total_params"], run["accuracy"]),
            textcoords="offset points", xytext=(8, 6), fontsize=9,
        )

    axis.set_xscale("log")
    axis.set_xlabel("파라미터 수 (log scale)")
    axis.set_ylabel("Test Accuracy")
    axis.set_title("파라미터 수 대비 정확도\n왼쪽 위에 있을수록 효율적인 구조")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()

    output = config.FIGURE_DIR / "params_vs_accuracy.png"
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"  저장: {output}")


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # 화면 없이 파일로만 저장
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib 과 numpy 가 필요하다: pip install matplotlib numpy")
        sys.exit(1)

    # 한글 라벨이 깨지지 않도록 폰트를 지정한다.
    # Windows 에 기본 설치된 맑은 고딕을 쓴다. 없으면 경고만 뜨고 진행된다.
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    config.ensure_dirs()

    runs = load_runs()
    if not runs:
        print(f"결과 JSON 이 없다: {config.HISTORY_DIR}")
        print("먼저 scripts/05_run_experiment.py 를 실행해라.")
        sys.exit(1)

    print("=" * 60)
    print(f"그림 생성 ({len(runs)}개 실행 결과)")
    print("=" * 60)

    plot_learning_curves(runs, plt)
    plot_confusion(runs, plt, np)
    plot_params_vs_accuracy(runs, plt)

    print(f"\n출력 폴더: {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
