"""
AI-Hub "신호등-도로표지판 인지 영상" annotation 파서.

실제 파일을 열어서 확인한 형식 (docs/DATA_FINDINGS.md):

    {
      "annotation": [
        {"box":[708,177,719,188], "class":"traffic_sign", "shape":"circle", "color":"blue"},
        {"shape":"circle","color":"red","kind":"normal","box":[579,151,595,167],
         "text":"0","type":"restriction","class":"traffic_sign"},
        {"light_count":"3","box":[538,160,560,167],"attribute":[{...}],
         "type":"car","class":"traffic_light","direction":"horizontal"}
      ],
      "image": {"filename":"s01027933.jpg", "imsize":[1280,720]}
    }

파싱하면서 조심해야 할 것들 — 전부 실제 데이터에서 관찰된 것이다:

  1. box 는 [x1, y1, x2, y2] 픽셀 좌표다. (x, y, w, h) 가 아니다.
     확인 방법: [708,177,719,188] 을 (x,y,w,h) 로 읽으면 x+w=1427 로
     이미지 폭 1280 을 넘는다. 따라서 (x1,y1,x2,y2) 다.

  2. **좌표 순서가 뒤집힌 경우가 있다.** 예: [315,308,287,335] 는 x2 < x1 이다.
     min/max 로 정규화하지 않으면 crop 이 빈 이미지가 되거나 예외가 난다.

  3. key 순서가 객체마다 다르다. box 가 먼저 오기도 하고 shape 가 먼저 오기도 한다.
     순서에 의존하는 파싱을 하면 안 된다.

  4. "text" 값의 타입이 일정하지 않다. 어떤 파일은 "0"(문자열), 어떤 파일은 0(정수).

  5. annotation 이 빈 배열인 파일이 많다. (표지판이 없는 프레임)

  6. type 키가 아예 없는 객체가 많다. circle_red 중 약 20% 가 그렇다.
     그래서 type 을 분류 기준으로 쓰지 않고 shape+color 를 쓴다.

  7. class 가 traffic_light 인 객체의 type 은 "car"/"bus"/"bicycle" 로,
     표지판의 type 과 의미가 완전히 다르다. class 로 먼저 걸러야 한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 이 프로젝트가 다루는 객체 종류. traffic_light / traffic_information 은 제외한다.
TARGET_OBJECT_CLASS = "traffic_sign"

# 실제 데이터에서 관찰된 값들
VALID_SHAPES = {"circle", "triangle", "rectangle"}
VALID_COLORS = {"red", "blue", "yellow"}


@dataclass(frozen=True)
class SignObject:
    """annotation 안의 표지판 객체 하나."""

    x1: int
    y1: int
    x2: int
    y2: int
    shape: str
    color: str
    sign_type: str          # 없을 수 있다 (빈 문자열)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def short_side(self) -> int:
        """짧은 변. crop 품질을 판단하는 기준으로 쓴다."""
        return min(self.width, self.height)

    @property
    def class_name(self) -> str:
        """분류 라벨. shape_color 조합. (예: circle_red)"""
        return f"{self.shape}_{self.color}"


@dataclass(frozen=True)
class ImageAnnotation:
    """json 파일 하나 = 원본 이미지 한 장."""

    filename: str           # 예: s01027933.jpg
    width: int
    height: int
    objects: list[SignObject]

    @property
    def stem(self) -> str:
        """확장자를 뗀 이름. 원천 jpg 와 매칭하는 키다."""
        return Path(self.filename).stem


def _normalize_box(raw: list) -> tuple[int, int, int, int] | None:
    """
    box 배열을 (x1, y1, x2, y2) 로 정규화한다.

    좌표가 뒤집혀 있는 경우(x2 < x1)가 실제로 존재하므로 min/max 로 강제한다.
    이걸 안 하면 PIL 의 crop 이 빈 이미지를 돌려주는데 예외는 안 난다.
    조용히 잘못된 데이터가 학습에 들어가는 최악의 경우다.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        a, b, c, d = (int(float(v)) for v in raw)
    except (TypeError, ValueError):
        return None

    x1, x2 = min(a, c), max(a, c)
    y1, y2 = min(b, d), max(b, d)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def parse_annotation(path: Path) -> ImageAnnotation | None:
    """
    json 한 개를 파싱한다. 표지판이 하나도 없으면 objects 가 빈 리스트가 된다.
    파일이 깨졌으면 None 을 돌려준다 (예외를 던지지 않는다 — 3만 개를 도는 중이므로).
    """
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None

    image_info = data.get("image") or {}
    filename = image_info.get("filename")
    if not filename:
        return None

    imsize = image_info.get("imsize") or []
    if len(imsize) == 2:
        img_width, img_height = int(imsize[0]), int(imsize[1])
    else:
        img_width = img_height = 0

    objects: list[SignObject] = []
    for entry in data.get("annotation") or []:
        if not isinstance(entry, dict):
            continue
        # traffic_light / traffic_information 은 여기서 걸러진다.
        if entry.get("class") != TARGET_OBJECT_CLASS:
            continue

        shape = str(entry.get("shape") or "").strip().lower()
        color = str(entry.get("color") or "").strip().lower()
        if shape not in VALID_SHAPES or color not in VALID_COLORS:
            continue

        box = _normalize_box(entry.get("box"))
        if box is None:
            continue

        # type 은 있으면 기록만 하고 분류에는 쓰지 않는다 (결측이 많다).
        sign_type = str(entry.get("type") or "").strip().lower()

        objects.append(
            SignObject(
                x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                shape=shape, color=color, sign_type=sign_type,
            )
        )

    return ImageAnnotation(
        filename=filename, width=img_width, height=img_height, objects=objects
    )


def crop_box(
    obj: SignObject,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    """
    실제로 자를 영역을 계산한다.

    bbox 그대로 자르면 표지판 테두리가 아슬아슬하게 잘린다.
    테두리(빨간 원, 노란 삼각형)는 이 데이터에서 클래스를 결정하는 정보 그 자체이므로
    약간의 여유를 두고 자른다.

    이미지 경계를 넘지 않도록 자른다. 넘어간 채로 PIL 에 넘기면
    검은 여백이 채워지는데, 그건 실제 배경이 아니라 가짜 정보다.
    """
    pad_x = int(obj.width * padding_ratio)
    pad_y = int(obj.height * padding_ratio)

    x1 = max(0, obj.x1 - pad_x)
    y1 = max(0, obj.y1 - pad_y)
    x2 = min(image_width, obj.x2 + pad_x) if image_width else obj.x2 + pad_x
    y2 = min(image_height, obj.y2 + pad_y) if image_height else obj.y2 + pad_y

    return x1, y1, x2, y2


def group_id_for(stem: str, bucket: int) -> str:
    """
    데이터 누수 방지용 그룹 키.

    파일명은 s01027933 처럼 접두사 + 순번이다.
    ID 가 가까우면 같은 주행 구간에서 연속 촬영됐을 가능성이 높다.
    그래서 순번을 bucket 으로 나눈 몫을 그룹으로 쓴다.

        bucket=1000 이면 s01027933 → "s01_27"   (27933 // 1000 = 27)

    bucket <= 0 이면 원본 이미지 단위로만 묶는다 (같은 사진에서 나온 crop 끼리).

    주의: 이건 어디까지나 휴리스틱이다. AI-Hub 가 ID 를 어떻게 부여했는지
    공식 문서로 확인한 게 아니다. bucket 을 키우면 더 보수적(누수에 안전)이지만
    그룹이 커져서 클래스별 분할이 치우칠 수 있다.
    scripts/04_build_manifest.py 의 출력에서 클래스별 train/val/test 분포를 보고 조정해라.
    """
    if bucket <= 0:
        return stem

    prefix = "".join(ch for ch in stem if not ch.isdigit())
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        return stem
    return f"{prefix}_{int(digits) // bucket}"
