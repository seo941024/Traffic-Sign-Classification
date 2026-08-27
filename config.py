"""
ANN / DNN / CNN 구조 정의에 필요한 설정.

PyTorch(src/torch_impl/models.py)와 Keras(src/keras_impl/models.py)가
이 값을 공유해서 같은 조건의 모델을 만든다.
"""

# 분류할 클래스. 필요에 맞게 바꿔서 쓴다.
CLASS_NAMES: list[str] = [
    "circle_blue",
    "circle_red",
    "rectangle_blue",
    "triangle_blue",
    "triangle_red",
    "triangle_yellow",
]
NUM_CLASSES = len(CLASS_NAMES)

# 입력 이미지 규격
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3

# ANN / DNN 이 Flatten 해서 받는 입력 차원.
# 32 * 32 * 3 = 3,072
FLATTENED_INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS

# 모델 구조
ANN_HIDDEN_UNITS = (256,)
DNN_HIDDEN_UNITS = (1024, 512, 256, 128)
CNN_CONV_CHANNELS = (32, 64, 128)
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2
CNN_HEAD_UNITS = (256,)
DROPOUT = 0.5


def input_shape(framework: str) -> tuple[int, ...]:
    """
    같은 이미지라도 프레임워크마다 텐서 축 순서가 다르다.

        PyTorch : (C, H, W)  — channels-first
        Keras   : (H, W, C)  — channels-last
    """
    if framework == "torch":
        return (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    if framework == "keras":
        return (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
    raise ValueError(f"알 수 없는 framework: {framework!r} (가능: 'torch', 'keras')")
