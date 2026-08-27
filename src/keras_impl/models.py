"""
Keras 로 구현한 ANN / DNN / CNN.

src/torch_impl/models.py 와 **같은 구조, 같은 파라미터 수**를 목표로 한다.
같은 구조를 두 프레임워크로 짜보면 다음이 드러난다.

  - 프레임워크가 달라도 파라미터 수는 정확히 같다
    → 파라미터 수는 프레임워크가 아니라 '구조'가 결정한다는 확인.
  - 그런데 코드 생김새와 텐서 축 순서는 다르다
    → 같은 수학을 표현하는 방식이 여럿이라는 것.

Keras 를 쓰는 이유:
    Sequential 로 층을 나열하면 그 자체가 구조 설명서가 된다.
    무엇보다 model.summary() 가 층별 출력 shape 와 파라미터 수를 표로 바로 찍어준다.
    PyTorch 에서는 hook 을 걸어야 얻는 정보를 Keras 는 기본 제공한다.

축 순서 주의:
    Keras 는 channels-last (H, W, C) = (32, 32, 3)
    PyTorch 는 channels-first (C, H, W) = (3, 32, 32)
    Flatten 결과의 '원소 순서'는 다르지만 개수는 3,072 로 같다.
    학습에는 영향이 없다. 모델은 어차피 입력 순서를 데이터로부터 배운다.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

import config


def build_ann(
    num_classes: int,
    input_shape: tuple[int, int, int] | None = None,
    hidden_units: tuple[int, ...] = config.ANN_HIDDEN_UNITS,
    dropout: float = config.DROPOUT,
) -> keras.Model:
    """
    ANN — Flatten 후 Dense 한 층.

        (None, 32, 32, 3)
            ↓ Flatten
        (None, 3072)      ← 공간 정보가 사라지는 지점
            ↓ Dense + ReLU
        (None, 256)
            ↓ Dense
        (None, num_classes)

    summary() 를 찍어보면 Flatten 직후 Dense 의 Param # 가
    3072 * 256 + 256 = 786,688 로 나온다. 모델 전체의 대부분이다.
    """
    input_shape = input_shape or config.input_shape("keras")

    model = keras.Sequential(name="ANN")
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Flatten(name="flatten"))
    for i, units in enumerate(hidden_units, start=1):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i}"))
    model.add(layers.Dropout(dropout, name="dropout"))
    model.add(layers.Dense(num_classes, name="logits"))
    return model


def build_dnn(
    num_classes: int,
    input_shape: tuple[int, int, int] | None = None,
    hidden_units: tuple[int, ...] = config.DNN_HIDDEN_UNITS,
    dropout: float = config.DROPOUT,
) -> keras.Model:
    """
    DNN — ANN 과 같은 방식이되 Dense 를 더 깊게 쌓았다.

        (None, 32, 32, 3)
            ↓ Flatten
        (None, 3072)
            ↓ Dense + ReLU  x 4
        (None, 1024) → (None, 512) → (None, 256) → (None, 128)
            ↓ Dense
        (None, num_classes)

    ANN 과 나란히 summary() 를 비교하면
    "층을 4배로 늘렸을 때 파라미터가 어디서 얼마나 늘었는지"가 바로 보인다.
    답: 늘어난 층들 때문이 아니라 첫 Dense 의 폭이 256 → 1024 로 커진 것이 지배적이다.
    """
    input_shape = input_shape or config.input_shape("keras")

    model = keras.Sequential(name="DNN")
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Flatten(name="flatten"))
    for i, units in enumerate(hidden_units, start=1):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i}"))
    model.add(layers.Dropout(dropout, name="dropout"))
    model.add(layers.Dense(num_classes, name="logits"))
    return model


def build_cnn(
    num_classes: int,
    input_shape: tuple[int, int, int] | None = None,
    conv_channels: tuple[int, ...] = config.CNN_CONV_CHANNELS,
    kernel_size: int = config.CNN_KERNEL_SIZE,
    pool_size: int = config.CNN_POOL_SIZE,
    head_units: tuple[int, ...] = config.CNN_HEAD_UNITS,
    dropout: float = config.DROPOUT,
) -> keras.Model:
    """
    CNN — Conv 로 공간 구조를 유지하며 특징을 뽑고, 마지막에만 Flatten 한다.

        (None, 32, 32, 3)
            ↓ Conv2D(32) + MaxPool
        (None, 16, 16, 32)
            ↓ Conv2D(64) + MaxPool
        (None, 8, 8, 64)
            ↓ Conv2D(128) + MaxPool
        (None, 4, 4, 128)
            ↓ Flatten
        (None, 2048)
            ↓ Dense
        (None, 256)
            ↓ Dense
        (None, num_classes)

    summary() 에서 반드시 확인할 것:
        Conv2D 층들의 Param # 를 다 더해도 93,248 개다 (전체의 15%).
        Conv2D(3->32, 3x3) = 3*3*3*32 + 32 = 896
        커널이 이미지 전체에 재사용되므로(weight sharing) 해상도가 커져도
        파라미터가 늘지 않는다. Dense 는 입력이 커지면 그대로 비례해 늘어난다.
        이 대비가 ANN/DNN 과 CNN 의 파라미터 증가 방식이 다른 이유 전부다.
    """
    input_shape = input_shape or config.input_shape("keras")

    model = keras.Sequential(name="CNN")
    model.add(keras.Input(shape=input_shape))
    for i, channels in enumerate(conv_channels, start=1):
        model.add(
            layers.Conv2D(
                channels,
                kernel_size,
                padding="same",   # Conv 자체는 해상도를 바꾸지 않는다
                activation="relu",
                name=f"conv_{i}",
            )
        )
        model.add(layers.MaxPooling2D(pool_size, name=f"pool_{i}"))
    model.add(layers.Flatten(name="flatten"))
    for i, units in enumerate(head_units, start=1):
        model.add(layers.Dense(units, activation="relu", name=f"dense_{i}"))
    model.add(layers.Dropout(dropout, name="dropout"))
    model.add(layers.Dense(num_classes, name="logits"))
    return model


_BUILDERS = {"ann": build_ann, "dnn": build_dnn, "cnn": build_cnn}


def build_model(name: str, num_classes: int) -> keras.Model:
    """'ann' / 'dnn' / 'cnn' 이름으로 Keras 모델을 만든다."""
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"알 수 없는 모델: {name!r} (가능: {sorted(_BUILDERS)})")
    return _BUILDERS[key](num_classes=num_classes)
