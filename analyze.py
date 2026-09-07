"""
발표용 그림 생성.

    python analyze.py

만드는 것 (figs/ 폴더):
    curves.png        ANN/DNN/CNN 학습곡선 (train/val acc)
    confusion.png     ANN vs CNN 혼동행렬
    per_class.png     클래스별 정확도 막대
    misclassified.png ANN 틀리고 CNN 맞춘 표지판 샘플

torch 로 세 모델을 다시 학습(약 6분, GPU)한 뒤 val 예측을 뽑아 그린다.
학습 조건은 train_torch.py 와 동일 (seed 42, Adam 1e-3, batch 32).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config
from src.torch_impl.models import build_model

SEED = 42
EPOCHS = 25
BATCH_SIZE = 32
LR = 1e-3
DATA_DIR = "crop"
OUT = "figs"

CODE_KO = {
    "213": "좌회전금지", "214": "우회전금지", "218": "주정차금지",
    "219": "주차금지", "221": "차높이제한", "308": "좌우회전",
    "309": "직진및좌회전", "311": "유턴", "322": "횡단보도", "329": "비보호좌회전",
}


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"   # Windows 한글
    except Exception:
        pass
    plt.rcParams["axes.unicode_minus"] = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tf_ = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    ds = datasets.ImageFolder(DATA_DIR, transform=tf_)
    labels = ds.classes
    label_ko = [f"{c}\n{CODE_KO.get(c, c)}" for c in labels]

    n_val = int(len(ds) * 0.2)
    tr, va = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    tl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(va, batch_size=256, shuffle=False)

    history = {}
    preds = {}
    y_true = None

    for name in ("ann", "dnn", "cnn"):
        torch.manual_seed(SEED)
        model = build_model(name, num_classes=len(labels)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = nn.CrossEntropyLoss()
        params = sum(p.numel() for p in model.parameters())
        tr_hist, va_hist = [], []

        for ep in range(1, EPOCHS + 1):
            model.train()
            correct = total = 0
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                lossf(out, y).backward()
                opt.step()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
            tr_acc = correct / total

            model.eval()
            c = t = 0
            P, Y = [], []
            with torch.no_grad():
                for x, y in vl:
                    out = model(x.to(device))
                    p = out.argmax(1).cpu().numpy()
                    P.append(p); Y.append(y.numpy())
                    c += (p == y.numpy()).sum(); t += len(y)
            va_acc = c / t
            tr_hist.append(tr_acc); va_hist.append(va_acc)
            print(f"  {name} {ep:2}/{EPOCHS}  train {tr_acc:.3f}  val {va_acc:.3f}")

        history[name] = (tr_hist, va_hist, params)
        preds[name] = np.concatenate(P)
        y_true = np.concatenate(Y)

    # 1) 학습곡선
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for a, name in zip(ax, ("ann", "dnn", "cnn")):
        tr_h, va_h, params = history[name]
        e = range(1, EPOCHS + 1)
        a.plot(e, tr_h, "--", label="train")
        a.plot(e, va_h, "-", label="val")
        a.set_title(f"{name.upper()}  ({params:,} params)\nbest val = {max(va_h):.3f}")
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.set_ylim(0.4, 1.0)
    ax[0].set_ylabel("accuracy"); ax[0].legend()
    fig.suptitle("학습곡선 — train vs val accuracy")
    fig.tight_layout(); fig.savefig(f"{OUT}/curves.png", dpi=130)
    print(f"saved {OUT}/curves.png")

    # 2) 혼동행렬 ANN vs CNN  (x축 라벨 수평 → 폭 넓히고 세로도 그만큼 키움)
    fig, ax = plt.subplots(1, 2, figsize=(24, 11))
    for a, name in zip(ax, ("ann", "cnn")):
        cm = confusion_matrix(y_true, preds[name], normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=label_ko).plot(
            ax=a, colorbar=False, values_format=".2f", cmap="Blues", xticks_rotation=0)
        a.tick_params(axis="x", labelsize=9)
        a.tick_params(axis="y", labelsize=9)
        acc = (preds[name] == y_true).mean()
        a.set_title(f"{name.upper()}  (val acc {acc:.3f})")
    fig.suptitle("혼동행렬 (행=정답, 열=예측, 행 정규화)")
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion.png", dpi=130)
    print(f"saved {OUT}/confusion.png")

    # 3) 클래스별 정확도
    fig, a = plt.subplots(figsize=(13, 5))
    x = np.arange(len(labels))
    w = 0.27
    for i, name in enumerate(("ann", "dnn", "cnn")):
        acc = [(preds[name][y_true == k] == k).mean() for k in range(len(labels))]
        a.bar(x + (i - 1) * w, acc, w, label=name.upper())
    a.set_xticks(x); a.set_xticklabels(label_ko, rotation=45, ha="right")
    a.set_ylabel("정확도"); a.set_ylim(0.6, 1.0); a.grid(axis="y", alpha=.3); a.legend()
    a.set_title("클래스별 정확도 — 안쪽 패턴만 다른 쌍(218 주정차금지 / 219 주차금지)에서 ANN 이 무너진다")
    fig.tight_layout(); fig.savefig(f"{OUT}/per_class.png", dpi=130)
    print(f"saved {OUT}/per_class.png")

    # 4) ANN 틀리고 CNN 맞춘 샘플
    va_idx = va.indices
    bad = [i for i in range(len(y_true))
           if preds["ann"][i] != y_true[i] and preds["cnn"][i] == y_true[i]]
    bad = bad[:20]
    if bad:
        imgs = torch.stack([ds[va_idx[i]][0] for i in bad]) * 0.5 + 0.5
        grid = torchvision.utils.make_grid(imgs, nrow=5).permute(1, 2, 0).numpy()
        fig, a = plt.subplots(figsize=(12, 10))
        a.imshow(grid); a.axis("off")
        cap = "  |  ".join(
            f"{CODE_KO[labels[y_true[i]]]}→ANN:{CODE_KO[labels[preds['ann'][i]]]}"
            for i in bad[:6])
        a.set_title(f"ANN 오답 / CNN 정답 샘플\n{cap}", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{OUT}/misclassified.png", dpi=130)
        print(f"saved {OUT}/misclassified.png  ({len(bad)}개)")

    # 요약 텍스트
    print("\n=== 요약 ===")
    for name in ("ann", "dnn", "cnn"):
        _, va_h, params = history[name]
        print(f"  {name.upper():4} {params:>10,} params   best val {max(va_h):.3f}")


if __name__ == "__main__":
    main()
