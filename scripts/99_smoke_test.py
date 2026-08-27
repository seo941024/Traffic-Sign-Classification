"""
파이프라인 전체 점검 — 합성 데이터로 코드가 도는지 확인한다.

AI-Hub 원본이 없어도, 압축을 풀지 않아도 실행된다.
가짜 표지판 이미지를 만들어서 manifest → 학습 → 평가 → 그림까지 전 과정을 돌려본다.

무엇을 확인하나:
    - 모든 모듈이 import 되는가
    - annotation 파서가 실제 형식(뒤집힌 box 포함)을 제대로 처리하는가
    - manifest 생성과 그룹 분할이 동작하는가
    - PyTorch / Keras 양쪽 학습 루프가 도는가
    - 평가 지표와 그림 생성이 동작하는가

무엇을 확인하지 못하나:
    - **모델이 실제로 잘 학습되는지는 알 수 없다.** 데이터가 가짜이기 때문이다.
      정확도 숫자에 의미를 두지 마라. "에러 없이 끝까지 도는가"만 본다.

사용법:
    python scripts/99_smoke_test.py
    python scripts/99_smoke_test.py --keep     # 생성한 임시 파일을 지우지 않는다
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# 합성 표지판을 그릴 때 쓰는 색. 실제 클래스 이름과 맞춘다.
COLOR_RGB = {"red": (200, 30, 30), "blue": (30, 60, 200), "yellow": (230, 200, 40)}


def make_fake_crops(root: Path, per_class: int) -> None:
    """
    클래스 이름대로 생긴 가짜 표지판을 그린다.
    circle_red 면 빨간 원, triangle_yellow 면 노란 삼각형.

    학습이 실제로 되는지 최소한이라도 보려고 클래스마다 실제로 다르게 그린다.
    (전부 같은 노이즈로 만들면 loss 가 안 내려가서 코드 문제인지 구분이 안 된다.)

    파일명은 실제 crop 규칙과 같게 짓는다: {원본stem}__{순번}.jpg
    그래야 그룹 분할 로직까지 함께 검증된다.
    """
    import random

    from PIL import Image, ImageDraw

    rng = random.Random(config.SEED)
    size = 48

    for class_name in config.CLASS_NAMES:
        shape, color = class_name.rsplit("_", 1)
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for index in range(per_class):
            image = Image.new("RGB", (size, size), (120, 120, 120))
            draw = ImageDraw.Draw(image)
            fill = COLOR_RGB[color]
            jitter = rng.randint(-3, 3)
            box = [6 + jitter, 6 + jitter, size - 6 + jitter, size - 6 + jitter]

            if shape == "circle":
                draw.ellipse(box, fill=fill, outline=(255, 255, 255), width=3)
            elif shape == "triangle":
                draw.polygon(
                    [(size // 2 + jitter, 6), (6, size - 6 + jitter),
                     (size - 6, size - 6 + jitter)],
                    fill=fill, outline=(255, 255, 255),
                )
            else:
                draw.rectangle(box, fill=fill, outline=(255, 255, 255), width=3)

            # 원본 이미지 하나당 표지판 2개 → 그룹 분할 로직을 실제로 태운다.
            source_stem = f"s01{index // 2:06d}"
            image.save(class_dir / f"{source_stem}__{index % 2}.jpg", quality=95)

    print(f"  합성 crop 생성: {len(config.CLASS_NAMES)}클래스 x {per_class}장")


def test_annotation_parser(tmp: Path) -> bool:
    """
    실제 데이터에서 관찰된 까다로운 케이스를 파서가 처리하는지 확인한다.
    특히 좌표가 뒤집힌 box 를 제대로 정규화하는지가 중요하다.
    """
    from src.data.aihub import parse_annotation

    sample = {
        "annotation": [
            # 정상
            {"box": [708, 177, 719, 188], "class": "traffic_sign",
             "shape": "circle", "color": "blue"},
            # 좌표가 뒤집힌 케이스 (실제 i0553098.json 에서 발견)
            {"shape": "triangle", "color": "yellow", "kind": "normal",
             "box": [315, 308, 287, 335], "text": 0, "type": "caution",
             "class": "traffic_sign"},
            # traffic_light — 걸러져야 한다
            {"light_count": "3", "box": [538, 160, 560, 167],
             "attribute": [{"red": "off", "green": "on"}],
             "type": "car", "class": "traffic_light", "direction": "horizontal"},
            # 깨진 box — 걸러져야 한다
            {"box": [1, 2], "class": "traffic_sign", "shape": "circle", "color": "red"},
        ],
        "image": {"filename": "s01027933.jpg", "imsize": [1280, 720]},
    }

    path = tmp / "s01027933.json"
    path.write_text(json.dumps(sample), encoding="utf-8")

    annotation = parse_annotation(path)
    if annotation is None:
        print("  실패: 파싱이 None 을 돌려줬다")
        return False

    ok = True
    if len(annotation.objects) != 2:
        print(f"  실패: traffic_sign 2개를 기대했는데 {len(annotation.objects)}개")
        ok = False

    flipped = next((o for o in annotation.objects if o.shape == "triangle"), None)
    if flipped is None:
        print("  실패: 뒤집힌 box 객체를 못 찾았다")
        ok = False
    elif flipped.x1 != 287 or flipped.x2 != 315:
        print(f"  실패: 뒤집힌 box 정규화 오류 x1={flipped.x1} x2={flipped.x2} (기대 287/315)")
        ok = False
    elif flipped.width != 28 or flipped.height != 27:
        print(f"  실패: 크기 계산 오류 w={flipped.width} h={flipped.height}")
        ok = False

    if ok:
        print("  annotation 파서 OK (뒤집힌 box 정규화, traffic_light 제외, 깨진 box 제외)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="파이프라인 전체 점검 (합성 데이터)")
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--keep", action="store_true", help="임시 파일을 지우지 않는다")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="trafficsign_smoke_"))
    crop_dir = tmp / "crop"
    manifest_path = tmp / "manifest.csv"

    failures: list[str] = []

    try:
        print("=" * 60)
        print("1. Annotation 파서")
        print("=" * 60)
        if not test_annotation_parser(tmp):
            failures.append("annotation 파서")

        print("\n" + "=" * 60)
        print("2. 합성 crop 생성")
        print("=" * 60)
        make_fake_crops(crop_dir, args.per_class)

        print("\n" + "=" * 60)
        print("3. Manifest 생성 및 그룹 분할")
        print("=" * 60)
        from src.data.manifest import Sample, summarize, write_manifest
        from src.data.split import infer_group_id, split_by_group

        samples = []
        for class_id, class_name in enumerate(config.CLASS_NAMES):
            for image_path in sorted((crop_dir / class_name).iterdir()):
                samples.append(
                    Sample(filepath=image_path.resolve(), class_name=class_name,
                           class_id=class_id, split="",
                           group_id=infer_group_id(image_path, bucket=0))
                )
        splits = split_by_group(samples, config.SPLIT_RATIOS, config.SEED)
        write_manifest(splits, manifest_path)
        print(summarize(manifest_path))

        # ---- 학습은 임시 manifest 를 보도록 config 를 잠시 갈아끼운다 ----
        original_manifest = config.MANIFEST_PATH
        original_workers = config.NUM_WORKERS
        config.MANIFEST_PATH = manifest_path
        config.NUM_WORKERS = 0  # Windows 에서 멀티프로세싱 워커는 스모크 테스트에 불필요

        print("\n" + "=" * 60)
        print("4. PyTorch 학습 루프")
        print("=" * 60)
        try:
            from src.analysis.metrics import baselines, evaluate, render_report
            from src.torch_impl.dataset import build_dataloader
            from src.torch_impl.models import build_model
            from src.torch_impl.train import predict, set_seed, train_model

            set_seed(config.SEED)
            model = build_model("cnn", num_classes=len(config.CLASS_NAMES))
            model, _, seconds = train_model(
                model, "cnn",
                build_dataloader("train", manifest_path),
                build_dataloader("val", manifest_path),
                epochs=args.epochs,
            )
            y_true, y_pred, ms = predict(model, build_dataloader("test", manifest_path))
            result = evaluate(y_true, y_pred, config.CLASS_NAMES, "cnn", "torch")
            print(f"\n{render_report(result)}")
            base = baselines(y_true, len(config.CLASS_NAMES))
            print(f"\n  baseline 무작위 {base['random']:.3f} / 최빈 {base['majority']:.3f}")
            print(f"  학습 {seconds:.1f}s, 추론 {ms:.3f}ms/장")
            print("  PyTorch 파이프라인 OK")
        except Exception as error:  # noqa: BLE001
            print(f"  실패: {type(error).__name__}: {error}")
            import traceback
            traceback.print_exc()
            failures.append("PyTorch 학습")

        print("\n" + "=" * 60)
        print("5. Keras 학습 루프")
        print("=" * 60)
        try:
            from src.analysis.metrics import evaluate as k_evaluate
            from src.keras_impl.dataset import build_dataset
            from src.keras_impl.models import build_model as k_build
            from src.keras_impl.train import predict as k_predict
            from src.keras_impl.train import set_seed as k_set_seed
            from src.keras_impl.train import train_model as k_train

            k_set_seed(config.SEED)
            model = k_build("cnn", num_classes=len(config.CLASS_NAMES))
            model, _, seconds = k_train(
                model, "cnn",
                build_dataset("train", manifest_path),
                build_dataset("val", manifest_path),
                epochs=args.epochs,
            )
            y_true, y_pred, ms = k_predict(model, build_dataset("test", manifest_path))
            result_k = k_evaluate(y_true, y_pred, config.CLASS_NAMES, "cnn", "keras")
            print(f"  accuracy {result_k.accuracy:.3f}, 학습 {seconds:.1f}s, 추론 {ms:.3f}ms/장")
            print("  Keras 파이프라인 OK")
        except Exception as error:  # noqa: BLE001
            print(f"  실패: {type(error).__name__}: {error}")
            import traceback
            traceback.print_exc()
            failures.append("Keras 학습")

        config.MANIFEST_PATH = original_manifest
        config.NUM_WORKERS = original_workers

        print("\n" + "=" * 60)
        print("6. 그림 생성")
        print("=" * 60)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            from src.analysis.model_stats import collect_torch_stats
            from src.torch_impl.models import build_model as t_build

            stat = collect_torch_stats(
                t_build("cnn", num_classes=len(config.CLASS_NAMES)),
                config.input_shape("torch"), "cnn",
            )
            print(f"  shape 추적 OK — {len(stat.layers)}개 층, {stat.total_params:,} 파라미터")

            figure, axis = plt.subplots()
            axis.imshow(np.eye(len(config.CLASS_NAMES)))
            figure.savefig(tmp / "test.png")
            plt.close(figure)
            print("  matplotlib 저장 OK")
        except Exception as error:  # noqa: BLE001
            print(f"  실패: {type(error).__name__}: {error}")
            failures.append("그림 생성")

    finally:
        if args.keep:
            print(f"\n임시 파일 유지: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"실패한 단계: {failures}")
        sys.exit(1)
    print("전체 통과 — 파이프라인이 끝까지 동작한다.")
    print("주의: 합성 데이터이므로 정확도 숫자에는 의미가 없다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
