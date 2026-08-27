"""
PyTorch 데이터 로딩.

ImageFolder 를 쓰지 않고 manifest CSV 를 직접 읽는다.
ImageFolder 는 폴더를 훑어 자기 나름대로 클래스 순서를 정하는데,
그러면 Keras 쪽과 정수 라벨이 어긋날 수 있다.
manifest 에 적힌 class_id 를 그대로 쓰면 그런 사고가 원천적으로 없다.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config
from src.data.manifest import Sample, read_manifest


def build_transform(train: bool = False) -> transforms.Compose:
    """
    전처리. Keras 쪽 (src/keras_impl/dataset.py) 과 **정확히 같은 연산**이어야 한다.

        Resize(32, 32) → ToTensor([0,1]) → Normalize(mean .5, std .5) → [-1, 1]

    train 인자는 지금은 아무것도 하지 않는다.
    Data Augmentation 은 핸드오프 34절의 '확장 3' 이며 1차 실험 범위 밖이다.
    지금 augmentation 을 넣으면 세 모델의 학습 조건이 달라져서
    구조 차이를 관찰한다는 목적이 흐려진다.

    나중에 augmentation 을 넣을 때 주의:
        좌회전/우회전 표지판에 Horizontal Flip 을 걸면 클래스 의미가 뒤집힌다.
        표지판은 flip 을 무조건 적용하면 안 되는 대표적인 도메인이다.
    """
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),  # (H,W,C) uint8 → (C,H,W) float [0,1]
            transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD),
        ]
    )


class TrafficSignDataset(Dataset):
    """manifest 의 한 split 을 그대로 읽어오는 Dataset."""

    def __init__(self, samples: list[Sample], transform: transforms.Compose | None = None):
        if not samples:
            raise ValueError("빈 샘플 목록으로 Dataset 을 만들 수 없다.")
        self.samples = samples
        self.transform = transform or build_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        # convert("RGB") 는 필수다. 흑백이나 팔레트 PNG 가 섞이면
        # 채널 수가 달라져 배치가 만들어지지 않는다.
        with Image.open(sample.filepath) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.class_id


def build_dataloader(
    split: str,
    manifest_path: Path | None = None,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> DataLoader:
    """
    split 이름으로 DataLoader 를 만든다.

    shuffle 은 train 에서만 켠다.
    val/test 에서 섞으면 Confusion Matrix 를 만들 때 예측과 정답의
    순서를 맞추기 번거로워지고, 섞어서 얻는 이득도 없다.
    """
    manifest_path = manifest_path or config.MANIFEST_PATH
    samples = read_manifest(manifest_path, split=split)
    dataset = TrafficSignDataset(samples, transform=build_transform(train=(split == "train")))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
