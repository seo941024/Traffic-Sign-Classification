"""
Keras / tf.data 데이터 로딩.

keras.utils.image_dataset_from_directory 를 쓰지 않고 manifest 를 직접 읽는다.
이유는 PyTorch 쪽과 같다 — 폴더 스캔에 클래스 순서를 맡기면
두 프레임워크의 정수 라벨이 어긋날 수 있다.

tf.data 의 강점(prefetch, 병렬 map)은 그대로 쓴다.
PyTorch 의 num_workers 에 해당하는 역할을 tf.data 가 그래프 수준에서 처리한다.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

import config
from src.data.manifest import read_manifest

AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_preprocess(path: tf.Tensor, label: tf.Tensor):
    """
    PyTorch 쪽 build_transform() 과 **정확히 같은 연산**을 해야 한다.

        decode → RGB 3채널 → resize(64,64) → [0,1] → (x - 0.5) / 0.5 → [-1,1]

    두 프레임워크의 입력 분포가 다르면 성능 차이가 구조 때문인지
    전처리 때문인지 구분할 수 없게 된다.
    """
    image = tf.io.read_file(path)
    # channels=3 을 명시하면 흑백/팔레트 이미지도 3채널로 강제 변환된다.
    # PyTorch 쪽의 .convert("RGB") 와 같은 역할이다.
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0

    mean = tf.constant(config.NORMALIZE_MEAN, dtype=tf.float32)
    std = tf.constant(config.NORMALIZE_STD, dtype=tf.float32)
    image = (image - mean) / std

    return image, label


def build_dataset(
    split: str,
    manifest_path: Path | None = None,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
) -> tf.data.Dataset:
    """
    split 이름으로 tf.data.Dataset 을 만든다.

    출력 shape: (batch, 32, 32, 3)  ← channels-last, PyTorch 와 축 순서가 다르다.
    라벨은 정수 하나 (sparse). 따라서 loss 는
    SparseCategoricalCrossentropy(from_logits=True) 를 써야 한다.
    모델 마지막 층에 softmax 를 넣지 않았기 때문에 from_logits=True 가 필수다.
    """
    manifest_path = manifest_path or config.MANIFEST_PATH
    samples = read_manifest(manifest_path, split=split)
    if not samples:
        raise ValueError(f"manifest 에 split={split!r} 샘플이 없다: {manifest_path}")

    paths = [str(sample.filepath) for sample in samples]
    labels = [sample.class_id for sample in samples]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if split == "train":
        # reshuffle_each_iteration=True 로 epoch 마다 다시 섞는다.
        # buffer 를 전체 크기로 잡아야 진짜 무작위가 된다.
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)  # 다음 배치를 미리 준비 → HDD I/O 병목 완화

    return dataset
