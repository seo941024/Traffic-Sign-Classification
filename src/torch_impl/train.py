"""
PyTorch 학습 루프.

핸드오프 19절의 원칙: 세 모델에 **완전히 동일한 학습 조건**을 적용한다.
그래서 이 함수는 모델만 바꿔 끼우고 나머지는 config.py 값을 그대로 쓴다.
모델별로 lr 이나 epoch 을 손대는 순간 "구조 때문에 생긴 차이"라고 말할 수 없게 된다.

기록하는 것 (핸드오프 21절):
    epoch 별 train/val loss, train/val accuracy
    학습 시간, 추론 시간, 파라미터 수, 모델 크기
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config


@dataclass
class TrainHistory:
    """epoch 별 기록. 학습 곡선을 그리고 과적합을 판단하는 근거다."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "epoch_seconds": self.epoch_seconds,
        }


def get_device() -> torch.device:
    """GPU 가 있으면 쓴다. 없으면 CPU 로 돌아가되 느리다는 걸 알려준다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("  경고: CUDA 를 못 쓴다. CPU 로 학습하면 매우 느리다.")
    return torch.device("cpu")


def set_seed(seed: int = config.SEED) -> None:
    """
    재현성 고정.

    완전한 재현은 cudnn.deterministic 까지 켜야 하지만 그러면 느려진다.
    이 프로젝트는 모델 간 '상대 비교'가 목적이므로 seed 고정만으로 충분하다.
    다만 절대 수치를 논문처럼 인용하지는 마라.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    """한 epoch 을 돌고 (평균 loss, accuracy) 를 돌려준다."""
    model.train(train)

    total_loss = 0.0
    correct = 0
    total = 0

    # torch.set_grad_enabled 로 train/eval 을 하나의 코드로 처리한다.
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # loss.item() * batch_size 로 누적해야 마지막 배치가 작을 때도 정확하다.
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    device: torch.device | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[nn.Module, TrainHistory, float]:
    """
    모델 하나를 학습시킨다.

    반환: (최적 가중치가 로드된 모델, 학습 기록, 총 학습 시간(초))

    Early stopping 을 쓰는 이유:
        DNN 은 파라미터가 많아 금방 과적합한다. CNN 은 더 오래 개선된다.
        고정 epoch 으로 돌리면 DNN 은 과적합된 상태로, CNN 은 덜 학습된 상태로
        비교하게 된다. val loss 기준으로 각자 최적 지점에서 멈추게 하는 것이 공정하다.
        patience 값 자체는 세 모델에 동일하게 적용한다.
    """
    device = device or get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()  # 모델이 logits 을 내므로 softmax 를 넣지 않았다
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = TrainHistory()
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        elapsed = time.time() - epoch_start
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)
        history.epoch_seconds.append(elapsed)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # CPU 로 복사해두지 않으면 이후 GPU 상태 변화에 영향을 받는다.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            marker = "  *best"
        else:
            epochs_without_improvement += 1

        print(
            f"  [{model_name}] epoch {epoch:>3}/{epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.4f}  |  "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}  ({elapsed:.1f}s){marker}"
        )

        if epochs_without_improvement >= patience:
            print(f"  [{model_name}] val loss 가 {patience} epoch 동안 개선되지 않아 중단")
            break

    total_seconds = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    return model, history, total_seconds


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device | None = None):
    """
    테스트 셋 예측. (y_true, y_pred, 이미지당 추론시간ms) 를 돌려준다.

    추론 시간은 핸드오프 21절이 요구하는 기록 항목이다.
    GPU 는 비동기로 동작하므로 synchronize 를 해야 실제 시간이 측정된다.
    이걸 빼먹으면 GPU 추론이 비현실적으로 빠르게 측정된다.
    """
    device = device or get_device()
    model = model.to(device).eval()

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    total_images = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        all_pred.append(outputs.argmax(dim=1).cpu().numpy())
        all_true.append(labels.numpy())
        total_images += labels.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        elapsed_ms / max(total_images, 1),
    )


def count_parameters(model: nn.Module) -> tuple[int, float]:
    """(파라미터 수, float32 기준 MB)"""
    total = sum(p.numel() for p in model.parameters())
    return total, total * 4 / (1024 ** 2)
