"""
모델을 '학습시키지 않고' 구조만 분석하는 도구.

이 프로젝트에서 가장 중요한 산출물은 정확도가 아니라
"각 Layer 를 지날 때 Tensor 모양이 어떻게 변하고, 파라미터가 어디에 몰려 있는가"다.
그건 학습 전에 이미 전부 알 수 있다. 데이터가 없어도 지금 확인할 수 있다.

두 프레임워크의 서로 다른 강점을 각각 쓴다.
    PyTorch : forward hook 으로 실행 중 실제 shape 를 가로챈다 (동적 추적)
    Keras   : model.summary() 와 layer.output.shape 가 기본 제공 (정적 조회)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayerStat:
    """Layer 하나가 무엇을 했는지 한 줄로 요약한 것."""

    name: str
    kind: str                    # Conv2d / Linear / MaxPool2d / Flatten ...
    output_shape: tuple[int, ...]
    params: int                  # 학습되는 파라미터 수 (bias 포함)

    @property
    def output_elements(self) -> int:
        """배치 차원을 뺀 출력 원소 수. 해상도가 어떻게 줄어드는지 볼 때 쓴다."""
        dims = [d for d in self.output_shape[1:] if d is not None]
        total = 1
        for d in dims:
            total *= d
        return total


@dataclass
class ModelStat:
    """모델 하나 전체의 구조 요약."""

    name: str
    framework: str
    total_params: int
    trainable_params: int
    layers: list[LayerStat] = field(default_factory=list)

    @property
    def model_size_mb(self) -> float:
        """float32 기준 파라미터가 차지하는 메모리(MB). 저장된 파일 크기와 거의 같다."""
        return self.total_params * 4 / (1024 ** 2)

    def params_by_kind(self) -> dict[str, int]:
        """
        층 종류별 파라미터 합계.

        CNN 에서 이걸 찍어보면 Conv2d 전체 합보다 Linear 하나가 훨씬 크다는 게 보인다.
        "CNN 은 파라미터가 적다"가 아니라 "CNN 은 Dense 에 들어가기 전에
        Pooling 으로 입력을 줄여놨다"가 정확한 설명이라는 근거가 된다.
        """
        totals: dict[str, int] = {}
        for layer in self.layers:
            if layer.params:
                totals[layer.kind] = totals.get(layer.kind, 0) + layer.params
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    def heaviest_layer(self) -> LayerStat | None:
        """파라미터가 가장 많은 층. 세 모델 모두 'Flatten 직후 첫 Dense' 가 나온다."""
        weighted = [layer for layer in self.layers if layer.params]
        return max(weighted, key=lambda layer: layer.params) if weighted else None


# ---------------------------------------------------------------------------
# PyTorch — forward hook 으로 실제 실행 중의 shape 를 잡는다
# ---------------------------------------------------------------------------
def collect_torch_stats(model, input_shape: tuple[int, ...], name: str) -> ModelStat:
    """
    더미 배치를 한 번 흘려보내면서 각 Layer 의 출력 shape 를 기록한다.

    왜 hook 을 쓰나:
        PyTorch 는 forward() 가 파이썬 코드라서, 실행해보기 전에는
        중간 텐서 모양을 알 수 없다(정적 그래프가 아니다).
        대신 각 모듈에 hook 을 걸면 실행 중 실제 값을 그대로 가져올 수 있다.
        이게 "shape 가 정말 이렇게 변한다"는 가장 확실한 증거다.

    학습이 아니다: no_grad + eval 로 순전파만 1회 수행한다.
    """
    import torch

    stats: list[LayerStat] = []
    handles = []

    def make_hook(layer_name: str, module):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            params = sum(p.numel() for p in module.parameters(recurse=False))
            stats.append(
                LayerStat(
                    name=layer_name,
                    kind=type(module).__name__,
                    output_shape=tuple(output.shape),
                    params=params,
                )
            )

        return hook

    # 잎(leaf) 모듈에만 건다. Sequential 같은 컨테이너까지 걸면 중복 기록된다.
    for layer_name, module in model.named_modules():
        if layer_name and not list(module.children()):
            handles.append(module.register_forward_hook(make_hook(layer_name, module)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, *input_shape))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return ModelStat(
        name=name,
        framework="torch",
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        layers=stats,
    )


# ---------------------------------------------------------------------------
# Keras — 이미 알고 있는 정보를 꺼내 쓴다
# ---------------------------------------------------------------------------
def collect_keras_stats(model, name: str) -> ModelStat:
    """
    Keras 는 층을 쌓는 시점에 출력 shape 가 이미 확정된다.
    따라서 실행할 필요 없이 layer.output.shape 로 바로 조회한다.
    (이것이 model.summary() 가 학습 전에도 표를 그려줄 수 있는 이유다.)
    """
    stats: list[LayerStat] = []
    for layer in model.layers:
        try:
            output_shape = tuple(layer.output.shape)
        except AttributeError:  # 아직 build 되지 않은 층
            output_shape = ()
        stats.append(
            LayerStat(
                name=layer.name,
                kind=type(layer).__name__,
                output_shape=output_shape,
                params=int(layer.count_params()),
            )
        )

    trainable = int(sum(w.numpy().size for w in model.trainable_weights))
    non_trainable = int(sum(w.numpy().size for w in model.non_trainable_weights))

    return ModelStat(
        name=name,
        framework="keras",
        total_params=trainable + non_trainable,
        trainable_params=trainable,
        layers=stats,
    )


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def format_shape(shape: tuple[int, ...]) -> str:
    """(1, 128, 8, 8) → '(B, 128, 8, 8)'  — 배치 차원은 B 로 표기."""
    if not shape:
        return "-"
    return "(B, " + ", ".join("?" if d is None else str(d) for d in shape[1:]) + ")"


def render_stat_table(stat: ModelStat) -> str:
    """층별 shape 와 파라미터를 표로 만든다."""
    header = f"{'Layer':<22} {'Type':<14} {'Output shape':<22} {'Params':>12}"
    lines = [
        f"[{stat.name.upper()}]  framework={stat.framework}",
        header,
        "-" * len(header),
    ]
    for layer in stat.layers:
        lines.append(
            f"{layer.name:<22} {layer.kind:<14} "
            f"{format_shape(layer.output_shape):<22} {layer.params:>12,}"
        )
    lines.append("-" * len(header))
    lines.append(f"{'합계':<22} {'':<14} {'':<22} {stat.total_params:>12,}")
    lines.append(f"모델 크기 (float32): {stat.model_size_mb:.2f} MB")

    heaviest = stat.heaviest_layer()
    if heaviest:
        share = heaviest.params / stat.total_params * 100
        lines.append(
            f"최대 파라미터 층: {heaviest.name} ({heaviest.kind}) "
            f"{heaviest.params:,} — 전체의 {share:.1f}%"
        )

    by_kind = stat.params_by_kind()
    if by_kind:
        summary = ", ".join(f"{kind} {count:,}" for kind, count in by_kind.items())
        lines.append(f"층 종류별 파라미터: {summary}")

    return "\n".join(lines)


def render_comparison(stats: list[ModelStat]) -> str:
    """여러 모델의 파라미터 수를 나란히 놓고 비교한다."""
    if not stats:
        return "(비교할 모델이 없다)"

    baseline = min(stat.total_params for stat in stats)
    header = (
        f"{'Model':<8} {'FW':<7} {'Params':>14} {'Size(MB)':>10} "
        f"{'최소 대비':>10}  {'최대 파라미터 층'}"
    )
    lines = ["모델 구조 비교", header, "-" * len(header)]
    for stat in stats:
        heaviest = stat.heaviest_layer()
        where = f"{heaviest.name} ({heaviest.params:,})" if heaviest else "-"
        lines.append(
            f"{stat.name:<8} {stat.framework:<7} {stat.total_params:>14,} "
            f"{stat.model_size_mb:>10.2f} {stat.total_params / baseline:>9.2f}x  {where}"
        )
    return "\n".join(lines)
