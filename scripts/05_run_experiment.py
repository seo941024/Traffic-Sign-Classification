"""
메인 실험 — ANN / DNN / CNN 을 동일 조건에서 학습시키고 비교한다.

이 스크립트 하나가 핸드오프 23절의 흐름 전체를 수행한다.
    동일 데이터셋 준비 → ANN → DNN → CNN → 동일 조건 비교 → 결과 분석

핵심: 세 모델에 바뀌는 건 모델 구조뿐이다.
    같은 manifest, 같은 split, 같은 batch/lr/epoch/optimizer/loss/seed.
    그래야 결과 차이를 "구조 때문"이라고 말할 수 있다.

사용법:
    python scripts/05_run_experiment.py --framework torch
    python scripts/05_run_experiment.py --framework keras
    python scripts/05_run_experiment.py --framework both
    python scripts/05_run_experiment.py --framework torch --epochs 3   # 빠른 검증
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.analysis.compare import render_table, save_csv  # noqa: E402
from src.analysis.metrics import evaluate, render_report  # noqa: E402
from src.data.manifest import class_names_from_manifest  # noqa: E402


def save_run(result, history: dict, framework: str) -> None:
    """
    학습 곡선과 Confusion Matrix 를 함께 저장한다.

    Confusion Matrix 를 여기서 같이 저장해두면 나중에 그림을 다시 그릴 때
    학습을 처음부터 돌릴 필요가 없다. 학습은 오래 걸리므로 중요하다.
    """
    payload = {
        "model": result.model_name,
        "framework": framework,
        "class_names": result.class_names,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "total_params": result.total_params,
        "model_size_mb": result.model_size_mb,
        "train_seconds": result.train_seconds,
        "inference_ms_per_image": result.inference_ms_per_image,
        "confusion": result.confusion.tolist(),
        "history": history,
    }
    path = config.HISTORY_DIR / f"{framework}_{result.model_name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_torch(class_names: list[str], epochs: int) -> list:
    import torch

    from src.torch_impl.dataset import build_dataloader
    from src.torch_impl.models import build_model
    from src.torch_impl.train import (
        count_parameters, get_device, predict, set_seed, train_model,
    )

    print("#" * 70)
    print("# PyTorch")
    print("#" * 70)

    device = get_device()
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_loader = build_dataloader("train")
    val_loader = build_dataloader("val")
    test_loader = build_dataloader("test")
    print(f"  train {len(train_loader.dataset):,} / "
          f"val {len(val_loader.dataset):,} / test {len(test_loader.dataset):,}\n")

    results = []
    for name in config.MODEL_NAMES:
        # 모델마다 seed 를 다시 고정한다.
        # 안 그러면 앞 모델의 학습이 난수 상태를 바꿔놔서
        # 뒤 모델이 다른 초기화를 받게 되고, 비교 조건이 어긋난다.
        set_seed(config.SEED)

        model = build_model(name, num_classes=len(class_names))
        total_params, size_mb = count_parameters(model)
        print(f"[{name.upper()}] 파라미터 {total_params:,} ({size_mb:.2f} MB)")

        model, history, train_seconds = train_model(
            model, name, train_loader, val_loader,
            epochs=epochs,
            device=device,
            checkpoint_path=config.CHECKPOINT_DIR / f"torch_{name}.pth",
        )

        y_true, y_pred, inference_ms = predict(model, test_loader, device)
        result = evaluate(y_true, y_pred, class_names, name, "torch")
        result.total_params = total_params
        result.model_size_mb = size_mb
        result.train_seconds = train_seconds
        result.inference_ms_per_image = inference_ms
        results.append(result)

        print(f"\n{render_report(result)}\n")
        save_run(result, history.as_dict(), "torch")

    return results


def run_keras(class_names: list[str], epochs: int) -> list:
    from src.keras_impl.dataset import build_dataset
    from src.keras_impl.models import build_model
    from src.keras_impl.train import (
        check_gpu, count_parameters, predict, set_seed, train_model,
    )

    print("#" * 70)
    print("# Keras")
    print("#" * 70)
    check_gpu()

    train_dataset = build_dataset("train")
    val_dataset = build_dataset("val")
    test_dataset = build_dataset("test")

    results = []
    for name in config.MODEL_NAMES:
        set_seed(config.SEED)

        model = build_model(name, num_classes=len(class_names))
        total_params, size_mb = count_parameters(model)
        print(f"\n[{name.upper()}] 파라미터 {total_params:,} ({size_mb:.2f} MB)")

        model, history, train_seconds = train_model(
            model, name, train_dataset, val_dataset,
            epochs=epochs,
            checkpoint_path=config.CHECKPOINT_DIR / f"keras_{name}.keras",
        )

        y_true, y_pred, inference_ms = predict(model, test_dataset)
        result = evaluate(y_true, y_pred, class_names, name, "keras")
        result.total_params = total_params
        result.model_size_mb = size_mb
        result.train_seconds = train_seconds
        result.inference_ms_per_image = inference_ms
        results.append(result)

        print(f"\n{render_report(result)}\n")
        save_run(result, history, "keras")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ANN/DNN/CNN 동일 조건 비교 실험")
    parser.add_argument("--framework", choices=("torch", "keras", "both"), default="torch")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = parser.parse_args()

    config.ensure_dirs()

    if not config.MANIFEST_PATH.exists():
        print(f"manifest 가 없다: {config.MANIFEST_PATH}")
        print("먼저 scripts/04_build_manifest.py 를 실행해라.")
        sys.exit(1)

    class_names = class_names_from_manifest(config.MANIFEST_PATH)

    print("=" * 70)
    print("ANN / DNN / CNN 비교 실험")
    print("=" * 70)
    print(f"클래스 {len(class_names)}개: {class_names}")
    print(f"입력 {config.IMAGE_SIZE}x{config.IMAGE_SIZE}x{config.IMAGE_CHANNELS} "
          f"(Flatten {config.FLATTENED_INPUT_DIM:,})")
    print(f"batch {config.BATCH_SIZE} / lr {config.LEARNING_RATE} / "
          f"epochs {args.epochs} / seed {config.SEED}")
    print("위 조건은 세 모델에 동일하게 적용된다. 바뀌는 것은 모델 구조뿐이다.\n")

    results = []
    if args.framework in ("torch", "both"):
        results += run_torch(class_names, args.epochs)
    if args.framework in ("keras", "both"):
        results += run_keras(class_names, args.epochs)

    print("=" * 70)
    print(render_table(results))
    print("=" * 70)

    # Baseline 없이는 위 정확도가 좋은 건지 나쁜 건지 판단할 수 없다.
    if results:
        import numpy as np

        from src.analysis.metrics import baselines

        # confusion 의 행 합 = 클래스별 실제 개수 → 여기서 y_true 분포를 복원한다.
        counts = results[0].confusion.sum(axis=1)
        y_true = np.repeat(np.arange(len(class_names)), counts)
        base = baselines(y_true, len(class_names))

        print("\n비교 기준선 (Test set)")
        print(f"  무작위 추측     {base['random']:.4f}")
        print(f"  최빈 클래스     {base['majority']:.4f}   "
              "← 이걸 못 넘는 모델은 아무것도 학습하지 못한 것이다")
        best = max(results, key=lambda r: r.accuracy)
        gain = best.accuracy - base["majority"]
        print(f"  최고 모델       {best.accuracy:.4f} ({best.model_name.upper()})"
              f"   최빈 대비 +{gain:.4f}")

    output = config.REPORT_DIR / "comparison.csv"
    save_csv(results, output)
    print(f"\n저장: {output}")
    print(f"학습 곡선: {config.HISTORY_DIR}")
    print("\n다음: python scripts/06_make_figures.py")


if __name__ == "__main__":
    main()
