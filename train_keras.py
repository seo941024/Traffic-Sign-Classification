"""
Keras 로 ANN / DNN / CNN 을 학습시키고 비교한다.

실행:
    python train_keras.py                    # 세 모델 전부
    python train_keras.py --model cnn        # 하나만
    python train_keras.py --epochs 3         # 빠른 확인
"""

import argparse
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
from src.keras_impl.models import build_model

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
SEED = 42


def load_data(data_dir):
    """crop 폴더를 train/val 로 나눠서 읽는다."""
    size = (config.IMAGE_SIZE, config.IMAGE_SIZE)

    train = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=size, batch_size=BATCH_SIZE,
    )
    val = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=size, batch_size=BATCH_SIZE,
    )

    class_names = train.class_names

    # [0,255] -> [-1,1]
    normalize = keras.layers.Rescaling(1.0 / 127.5, offset=-1)
    train = train.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)
    val = val.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)

    return train, val, class_names


def run(name, train_ds, val_ds, num_classes, epochs):
    keras.utils.set_random_seed(SEED)

    model = build_model(name, num_classes=num_classes)

    # 모델 마지막 층이 softmax 없이 logits 을 그대로 내보내므로 from_logits=True.
    # 빠뜨리면 loss 가 잘못 계산되는데 에러는 안 나서 찾기 어렵다.
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    print(f"\n{'=' * 55}")
    print(f"{name.upper()}  -  파라미터 {model.count_params():,}")
    print("=" * 55)

    start = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )],
        verbose=2,
    )
    seconds = time.time() - start

    return {
        "name": name,
        "params": model.count_params(),
        "acc": max(history.history["val_accuracy"]),
        "seconds": seconds,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="crop")
    parser.add_argument("--model", choices=["ann", "dnn", "cnn"], default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    train_ds, val_ds, class_names = load_data(args.data_dir)
    print(f"\n클래스 {len(class_names)}개: {class_names}")

    targets = [args.model] if args.model else ["ann", "dnn", "cnn"]
    results = [run(n, train_ds, val_ds, len(class_names), args.epochs) for n in targets]

    print(f"\n{'=' * 55}")
    print(f"{'Model':<6} {'Params':>12} {'Val Acc':>9} {'학습(s)':>9}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<6} {r['params']:>12,} {r['acc']:>9.4f} {r['seconds']:>9.1f}")
    print("=" * 55)
    print(f"\n무작위 추측 = {1 / len(class_names):.3f}  (이걸 못 넘으면 학습이 안 된 것)")


if __name__ == "__main__":
    main()
