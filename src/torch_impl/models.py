"""
PyTorch 로 구현한 ANN / DNN / CNN.

세 모델은 같은 입력 (3, 32, 32) 을 받아 같은 개수의 클래스를 출력한다.
다른 것은 "중간에서 무엇을 하는가" 뿐이다.

PyTorch 를 쓰는 이유:
    forward() 에 데이터 흐름이 그대로 코드로 적혀 있다.
    어느 줄에서 텐서 모양이 어떻게 변하는지 눈으로 따라갈 수 있고,
    forward hook 을 걸면 실행 중 실제 shape 를 그대로 뽑아낼 수 있다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config


class ANN(nn.Module):
    """
    ANN — 가장 단순한 Fully Connected Network.

    핵심: 이미지를 Flatten 해서 1차원 벡터로 만든 뒤 Dense Layer 에 넣는다.

        (B, 3, 32, 32)
            ↓ Flatten
        (B, 3072)        ← 여기서 '공간 정보'가 사라진다
            ↓ Linear
        (B, 256)
            ↓ Linear
        (B, num_classes)

    Flatten 이 하는 일은 단순히 "줄 세우기"다.
    (0,0) 픽셀과 (0,1) 픽셀은 원래 서로 옆에 붙어 있었지만,
    Flatten 이후에는 그냥 벡터의 0번, 1번 원소일 뿐이다.
    벡터의 0번과 64번이 원래 세로로 인접했다는 사실도 모델은 알지 못한다.
    Dense Layer 는 모든 입력을 대등하게 취급하므로 이 인접 관계를 데이터로부터
    처음부터 다시 배워야 한다. 이것이 ANN 이 이미지에 불리한 이유다.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = config.FLATTENED_INPUT_DIM,
        hidden_units: tuple[int, ...] = config.ANN_HIDDEN_UNITS,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        layers: list[nn.Module] = []
        in_features = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU(inplace=True))
            in_features = units
        self.hidden = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)      # (B, 3, 32, 32) -> (B, 3072)
        x = self.hidden(x)       # (B, 3072)     -> (B, 256)
        x = self.dropout(x)
        return self.classifier(x)  # (B, 256) -> (B, num_classes)


class DNN(nn.Module):
    """
    DNN — ANN 과 구조는 같고 Hidden Dense Layer 만 더 깊게 쌓았다.

        (B, 3, 32, 32)
            ↓ Flatten
        (B, 3072)
            ↓ Linear + ReLU  x 4
        (B, 1024) → (B, 512) → (B, 256) → (B, 128)
            ↓ Linear
        (B, num_classes)

    ANN 과의 유일한 차이는 층의 개수다. 입력 처리 방식은 완전히 같다.
    즉 Layer 를 깊게 쌓으면 더 복잡한 비선형 관계를 표현할 수는 있지만,
    입력이 이미 Flatten 되어 있다는 근본적인 제약은 그대로다.
    깊이를 늘리는 것으로는 잃어버린 공간 정보를 되찾을 수 없다.

    파라미터 관점에서 볼 것:
        첫 Linear 하나가 3072 x 1024 ≈ 315만 개를 차지한다.
        모델 전체 파라미터의 대부분이 "Flatten 직후 첫 층"에 몰린다.
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = config.FLATTENED_INPUT_DIM,
        hidden_units: tuple[int, ...] = config.DNN_HIDDEN_UNITS,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        layers: list[nn.Module] = []
        in_features = input_dim
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU(inplace=True))
            in_features = units
        self.hidden = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.dropout(x)
        return self.classifier(x)


class CNN(nn.Module):
    """
    CNN — Convolution 으로 공간 구조를 유지한 채 특징을 추출한다.

        (B, 3, 32, 32)
            ↓ Conv(3->32) + ReLU + MaxPool
        (B, 32, 16, 16)
            ↓ Conv(32->64) + ReLU + MaxPool
        (B, 64, 8, 8)
            ↓ Conv(64->128) + ReLU + MaxPool
        (B, 128, 4, 4)
            ↓ Flatten          ← Flatten 은 '마지막에' 한 번만 한다
        (B, 2048)
            ↓ Linear
        (B, 256)
            ↓ Linear
        (B, num_classes)

    Convolution 이 실제로 계산하는 것:
        3x3 커널 하나가 이미지 위를 훑으면서, 매 위치마다
        (커널 9칸 x 입력 채널) 과의 원소별 곱을 전부 더한다. 즉 국소 가중합이다.
        같은 커널을 이미지 전체에 재사용하므로
          - 파라미터가 위치 수에 비례해 늘지 않고 (weight sharing)
          - 왼쪽 위에서 배운 '세로 모서리 찾기'가 오른쪽 아래에서도 그대로 통한다.

    Feature Map:
        커널 하나가 이미지 전체를 훑고 나온 2차원 출력이 feature map 1장이다.
        Conv(3->32) 는 서로 다른 커널 32개를 쓰므로 feature map 32장이 나온다.
        각 장은 "이 커널이 반응하는 패턴이 이미지의 어느 위치에 있는가"의 지도다.
        얕은 층은 선·모서리·색 변화 같은 단순한 것에, 깊은 층은 그것들이 조합된
        원형 테두리·삼각형·숫자 형태 같은 것에 반응한다.

    Pooling 을 쓰는 이유:
        1) 해상도를 절반으로 줄여 이후 연산량과 파라미터를 크게 줄인다.
        2) 다음 Conv 의 커널이 원본 기준으로 더 넓은 영역을 보게 된다(수용 영역 확대).
        3) 표지판이 몇 픽셀 밀려 찍혀도 최댓값은 잘 안 변하므로 작은 위치 변화에 둔감해진다.

    파라미터 관점에서 볼 것:
        Conv 층 3개를 다 합쳐도 파라미터는 93,248 개뿐이다.
        반면 Flatten 직후 Linear(2048 -> 256) 하나가 약 52만 개다.
        CNN 조차도 파라미터 대부분은 Dense 에 있다 — 다만 Pooling 으로 4x4 까지
        줄여놨기 때문에 ANN/DNN 보다 훨씬 작다.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = config.IMAGE_CHANNELS,
        image_size: int = config.IMAGE_SIZE,
        conv_channels: tuple[int, ...] = config.CNN_CONV_CHANNELS,
        kernel_size: int = config.CNN_KERNEL_SIZE,
        pool_size: int = config.CNN_POOL_SIZE,
        head_units: tuple[int, ...] = config.CNN_HEAD_UNITS,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()

        blocks: list[nn.Module] = []
        channels = in_channels
        for out_channels in conv_channels:
            blocks.append(
                nn.Conv2d(
                    channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # 'same' — Conv 자체는 해상도를 바꾸지 않는다
                )
            )
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(pool_size))  # 해상도를 절반으로 줄이는 건 여기뿐
            channels = out_channels
        self.features = nn.Sequential(*blocks)

        # Conv 는 padding='same' 이라 해상도를 유지하고, MaxPool 만 절반으로 줄인다.
        # 따라서 Block 을 n 개 지나면 해상도는 image_size / (pool_size ** n).
        spatial = image_size // (pool_size ** len(conv_channels))
        if spatial < 1:
            raise ValueError(
                f"Conv Block 이 너무 많다: {image_size}px 를 {len(conv_channels)}번 "
                f"{pool_size}배 줄이면 해상도가 0 이 된다. "
                f"CNN_CONV_CHANNELS 를 줄이거나 IMAGE_SIZE 를 키워라."
            )
        self.flatten_dim = channels * spatial * spatial

        self.flatten = nn.Flatten()

        layers: list[nn.Module] = []
        in_features = self.flatten_dim
        for units in head_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU(inplace=True))
            in_features = units
        self.head = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)     # (B, 3, 32, 32) -> (B, 128, 4, 4)
        x = self.flatten(x)      # (B, 128, 4, 4) -> (B, 2048)
        x = self.head(x)         # (B, 2048)      -> (B, 256)
        x = self.dropout(x)
        return self.classifier(x)


# 이름으로 모델을 만들 수 있게 해두면, 비교 스크립트에서 세 모델을 반복문으로 돌릴 수 있다.
_BUILDERS = {"ann": ANN, "dnn": DNN, "cnn": CNN}


def build_model(name: str, num_classes: int) -> nn.Module:
    """'ann' / 'dnn' / 'cnn' 이름으로 PyTorch 모델을 만든다."""
    key = name.lower()
    if key not in _BUILDERS:
        raise ValueError(f"알 수 없는 모델: {name!r} (가능: {sorted(_BUILDERS)})")
    return _BUILDERS[key](num_classes=num_classes)
