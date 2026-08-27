"""
Keras 학습 루프.

PyTorch 쪽 (src/torch_impl/train.py) 과 **완전히 같은 조건**을 쓴다.
    같은 optimizer(Adam), 같은 lr, 같은 batch, 같은 epoch, 같은 early stopping patience.

Keras 를 쓰는 이유:
    model.fit() 하나로 학습 루프·검증·콜백이 다 처리된다.
    PyTorch 에서 직접 짠 _run_epoch 이 Keras 에서는 프레임워크가 제공하는 부분이다.
    같은 실험을 두 방식으로 해보면 "학습 루프가 실제로 무엇을 하는지"가 명확해진다.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

import config


def set_seed(seed: int = config.SEED) -> None:
    """재현성 고정. TF/Keras/numpy/python random 을 한 번에 처리한다."""
    keras.utils.set_random_seed(seed)


def check_gpu() -> bool:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("  경고: TensorFlow 가 GPU 를 못 쓴다. (Windows 네이티브는 TF 2.11+ 에서 미지원)")
        print("        Keras 는 구조 비교용이므로 CPU 로 진행해도 된다.")
        return False
    print(f"  GPU {len(gpus)}개 사용")
    return True


def compile_model(model: keras.Model, learning_rate: float = config.LEARNING_RATE) -> keras.Model:
    """
    모델을 컴파일한다.

    from_logits=True 가 **필수**다.
    모델 마지막 층에 softmax 를 넣지 않고 logits 을 그대로 내보내기 때문이다.
    (PyTorch 의 nn.CrossEntropyLoss 와 동일한 조건을 맞추기 위해서다.)

    이걸 빠뜨리면 loss 가 조용히 잘못 계산된다. 에러는 안 난다.
    학습이 되긴 하는데 성능이 이상하게 낮게 나오고, 원인을 찾기 어렵다.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: keras.Model,
    model_name: str,
    train_dataset,
    val_dataset,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    checkpoint_path: Path | None = None,
) -> tuple[keras.Model, dict, float]:
    """
    모델 하나를 학습시킨다.

    반환: (최적 가중치가 복원된 모델, history dict, 총 학습 시간(초))

    restore_best_weights=True 로 PyTorch 쪽의 best_state 복원과 동작을 맞춘다.
    """
    compile_model(model, learning_rate)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )

    print(f"  [{model_name}] 학습 시작")
    start = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    total_seconds = time.time() - start

    # PyTorch 쪽 TrainHistory 와 키 이름을 맞춰둔다 (비교/시각화 코드를 공유하려고).
    normalized = {
        "train_loss": [float(v) for v in history.history.get("loss", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "train_acc": [float(v) for v in history.history.get("accuracy", [])],
        "val_acc": [float(v) for v in history.history.get("val_accuracy", [])],
        "epoch_seconds": [],
    }
    return model, normalized, total_seconds


def predict(model: keras.Model, dataset) -> tuple[np.ndarray, np.ndarray, float]:
    """
    테스트 셋 예측. (y_true, y_pred, 이미지당 추론시간ms) 를 돌려준다.

    dataset 은 shuffle 되지 않은 상태여야 한다.
    (src/keras_impl/dataset.py 는 train 에서만 shuffle 을 켠다.)
    """
    y_true: list[np.ndarray] = []
    for _, labels in dataset:
        y_true.append(labels.numpy())

    start = time.time()
    logits = model.predict(dataset, verbose=0)
    elapsed_ms = (time.time() - start) * 1000

    y_true_array = np.concatenate(y_true)
    y_pred = np.argmax(logits, axis=1)

    return y_true_array, y_pred, elapsed_ms / max(len(y_true_array), 1)


def count_parameters(model: keras.Model) -> tuple[int, float]:
    """(파라미터 수, float32 기준 MB)"""
    total = int(
        sum(w.numpy().size for w in model.trainable_weights)
        + sum(w.numpy().size for w in model.non_trainable_weights)
    )
    return total, total * 4 / (1024 ** 2)
