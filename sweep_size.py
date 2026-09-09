"""
입력 이미지 크기를 바꿔가며 ANN / DNN / CNN 의 정확도가 어떻게 갈리는지 본다.

발표 피드백:
    "CNN 이 이기는 건 당연하다. 그러라고 만든 거니까.
     하지만 이미지가 작으면 ANN/DNN 으로도 어느 정도 나온다.
     입력 크기를 바꿔봤더니 이만큼 차이가 나더라 — 이게 있어야 한다."

그래서 16 / 24 / 32 / 48 / 64 px 로 같은 데이터를 리사이즈해서
세 구조를 각각 학습시키고, (크기 x 구조) 별 best val accuracy 표를 만든다.

실행:
    python sweep_size.py                # 전체 (GPU 면 수 분)
    python sweep_size.py --sizes 16 32  # 일부만
    python sweep_size.py --epochs 15
"""

import argparse
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config
from src.torch_impl.models import ANN, DNN, CNN

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
SEED = 42
SIZES = [16, 24, 32, 48, 64]


def load_data(data_dir, size):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.ImageFolder(data_dir, transform=tf)
    n_val = int(len(ds) * 0.2)
    tr, va = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    return (
        DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(va, batch_size=BATCH_SIZE, shuffle=False),
        len(ds.classes),
    )


def build(name, size, num_classes):
    """config 기본값(32px)이 아니라 이번 size 에 맞춰 모델을 만든다."""
    dim = size * size * config.IMAGE_CHANNELS
    if name == "ann":
        return ANN(num_classes=num_classes, input_dim=dim)
    if name == "dnn":
        return DNN(num_classes=num_classes, input_dim=dim)
    if name == "cnn":
        return CNN(num_classes=num_classes, image_size=size)
    raise ValueError(name)


def run_epoch(model, loader, crit, opt, device, train):
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def train_one(name, size, tr, va, num_classes, epochs, device):
    torch.manual_seed(SEED)
    model = build(name, size, num_classes).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    params = sum(p.numel() for p in model.parameters())

    best_loss, best_acc, patience = float("inf"), 0.0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        run_epoch(model, tr, crit, opt, device, True)
        vl, va_acc = run_epoch(model, va, crit, opt, device, False)
        if vl < best_loss:
            best_loss, best_acc, patience = vl, va_acc, 0
        else:
            patience += 1
            if patience >= 5:
                break
    return {"model": name, "size": size, "params": params,
            "val_acc": round(best_acc, 4), "epochs": ep,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="crop")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    rows = []
    for size in args.sizes:
        tr, va, nc = load_data(args.data_dir, size)
        print(f"\n{'#' * 55}\n# 입력 {size}x{size}  (flatten dim = {size*size*3:,})\n{'#' * 55}")
        for name in ["ann", "dnn", "cnn"]:
            r = train_one(name, size, tr, va, nc, args.epochs, device)
            rows.append(r)
            print(f"  {name.upper():<4} params {r['params']:>10,}  "
                  f"val_acc {r['val_acc']:.4f}  ({r['epochs']}ep, {r['seconds']}s)")

    # 표 출력
    print(f"\n\n{'=' * 60}")
    print("입력크기 x 구조  ->  val accuracy")
    print("-" * 60)
    print(f"{'size':>6} | {'ANN':>8} {'DNN':>8} {'CNN':>8} | {'CNN-ANN':>8}")
    print("-" * 60)
    by = {(r["size"], r["model"]): r["val_acc"] for r in rows}
    for size in args.sizes:
        a, d, c = by.get((size, "ann")), by.get((size, "dnn")), by.get((size, "cnn"))
        gap = f"{c - a:+.3f}" if (a is not None and c is not None) else "-"
        print(f"{size:>6} | {a:>8.4f} {d:>8.4f} {c:>8.4f} | {gap:>8}")
    print("=" * 60)

    with open("figs/sweep_size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nfigs/sweep_size.csv 저장")

    # 그림
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, mark in [("ann", "o-"), ("dnn", "s-"), ("cnn", "^-")]:
            ys = [by.get((s, name)) for s in args.sizes]
            ax.plot(args.sizes, ys, mark, label=name.upper(), linewidth=2, markersize=8)
        ax.set_xlabel("input image size (px)")
        ax.set_ylabel("validation accuracy")
        ax.set_title("Input size vs accuracy by architecture")
        ax.set_xticks(args.sizes)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("figs/sweep_size.png", dpi=130)
        print("figs/sweep_size.png 저장")
    except Exception as e:
        print(f"그림 생략: {e}")


if __name__ == "__main__":
    main()
