"""혼동행렬을 ANN / CNN 따로 PNG 2장으로 생성. ANN, CNN 만 학습 → 약 4분.
   figs/confusion_ann.png, figs/confusion_cnn.png (+ 합본 confusion.png)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config
from src.torch_impl.models import build_model

SEED, EPOCHS = 42, 25
CODE_KO = {"213": "좌회전금지", "214": "우회전금지", "218": "주정차금지", "219": "주차금지",
           "221": "차높이제한", "308": "좌우회전", "309": "직진및좌회전", "311": "유턴",
           "322": "횡단보도", "329": "비보호좌회전"}

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
device = "cuda" if torch.cuda.is_available() else "cpu"

tf_ = transforms.Compose([transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                          transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
ds = datasets.ImageFolder("crop", transform=tf_)
labels = ds.classes
label_ko = [f"{c}\n{CODE_KO.get(c, c)}" for c in labels]
n_val = int(len(ds) * 0.2)
tr, va = random_split(ds, [len(ds)-n_val, n_val], generator=torch.Generator().manual_seed(SEED))
tl = DataLoader(tr, batch_size=32, shuffle=True)
vl = DataLoader(va, batch_size=256)

preds, y_true = {}, None
for name in ("ann", "cnn"):
    torch.manual_seed(SEED)
    m = build_model(name, num_classes=len(labels)).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        m.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
    m.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in vl:
            P.append(m(x.to(device)).argmax(1).cpu().numpy()); Y.append(y.numpy())
    preds[name] = np.concatenate(P); y_true = np.concatenate(Y)
    print(name, "acc", (preds[name] == y_true).mean())


def draw(name, path):
    acc = (preds[name] == y_true).mean()
    cm = confusion_matrix(y_true, preds[name], normalize="true")
    fig, a = plt.subplots(figsize=(13, 11))
    ConfusionMatrixDisplay(cm, display_labels=label_ko).plot(
        ax=a, colorbar=True, values_format=".2f", cmap="Blues", xticks_rotation=0)
    a.set_title(f"{name.upper()} 혼동행렬  (검증 정확도 {acc:.3f})\n"
                f"행 = 실제 표지판, 열 = 모델 예측, 숫자 = 비율(행 합 = 1)", fontsize=12)
    a.set_xlabel("모델이 예측한 표지판"); a.set_ylabel("실제 표지판")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print("saved", path)


draw("ann", "figs/confusion_ann.png")
draw("cnn", "figs/confusion_cnn.png")

# 합본도 갱신
fig, ax = plt.subplots(1, 2, figsize=(24, 11))
for a, name in zip(ax, ("ann", "cnn")):
    cm = confusion_matrix(y_true, preds[name], normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=label_ko).plot(
        ax=a, colorbar=False, values_format=".2f", cmap="Blues", xticks_rotation=0)
    a.set_title(f"{name.upper()}  (val acc {(preds[name]==y_true).mean():.3f})")
fig.suptitle("혼동행렬 (행=정답, 열=예측, 행 정규화)")
fig.tight_layout(); fig.savefig("figs/confusion.png", dpi=130)
print("saved figs/confusion.png")
