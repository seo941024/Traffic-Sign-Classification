"""analyze.py 의 Keras 버전. figs_keras/ 에 그림 저장.

    curves_keras.png / confusion_keras_ann.png / confusion_keras_cnn.png
    / per_class_keras.png

train_keras.py 와 동일 조건으로 ANN/DNN/CNN 재학습 후 val 예측을 뽑아 그린다.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import config
from src.keras_impl.models import build_model

SEED, EPOCHS, OUT = 42, 25, "figs_keras"
CODE_KO = {"213": "좌회전금지", "214": "우회전금지", "218": "주정차금지", "219": "주차금지",
           "221": "차높이제한", "308": "좌우회전", "309": "직진및좌회전", "311": "유턴",
           "322": "횡단보도", "329": "비보호좌회전"}

os.makedirs(OUT, exist_ok=True)
try:
    plt.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False

size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
train = keras.utils.image_dataset_from_directory(
    "crop", validation_split=0.2, subset="training", seed=SEED, image_size=size, batch_size=32)
val = keras.utils.image_dataset_from_directory(
    "crop", validation_split=0.2, subset="validation", seed=SEED, image_size=size, batch_size=32)
labels = train.class_names
label_ko = [f"{c}\n{CODE_KO.get(c, c)}" for c in labels]

norm = keras.layers.Rescaling(1.0 / 127.5, offset=-1)
train = train.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
# val 은 매 순회마다 셔플되므로(기본값) 그림용으로는 순서를 고정한다.
# 안 그러면 예측과 정답이 어긋나 혼동행렬이 무작위처럼 나온다.
val = val.map(lambda x, y: (norm(x), y))
val_fixed = list(val.as_numpy_iterator())          # [(x_batch, y_batch), ...] 순서 고정
val = val.prefetch(tf.data.AUTOTUNE)               # fit() 용은 그대로

y_true = np.concatenate([y for _, y in val_fixed])

history, preds = {}, {}
for name in ("ann", "dnn", "cnn"):
    keras.utils.set_random_seed(SEED)
    model = build_model(name, num_classes=len(labels))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    h = model.fit(train, validation_data=val, epochs=EPOCHS,
                  callbacks=[keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=5, restore_best_weights=True)],
                  verbose=2)
    history[name] = h.history
    p = [np.argmax(model.predict(xb, verbose=0), axis=1) for xb, _ in val_fixed]
    preds[name] = np.concatenate(p)
    print(name, "best val acc", max(h.history["val_accuracy"]),
          "| 그림용 acc", (preds[name] == y_true).mean())

# 1) 학습곡선
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for a, name in zip(ax, ("ann", "dnn", "cnn")):
    h = history[name]
    e = range(1, len(h["accuracy"]) + 1)
    a.plot(e, h["accuracy"], "--", label="train")
    a.plot(e, h["val_accuracy"], "-", label="val")
    a.set_title(f"{name.upper()}  best val = {max(h['val_accuracy']):.3f}")
    a.set_xlabel("epoch"); a.grid(alpha=.3); a.set_ylim(0.4, 1.0)
ax[0].set_ylabel("accuracy"); ax[0].legend()
fig.suptitle("Keras — 학습곡선 (train vs val accuracy)")
fig.tight_layout(); fig.savefig(f"{OUT}/curves_keras.png", dpi=130); plt.close(fig)

# 2) 혼동행렬 ANN / CNN 따로
for name in ("ann", "cnn"):
    acc = (preds[name] == y_true).mean()
    cm = confusion_matrix(y_true, preds[name], normalize="true")
    fig, a = plt.subplots(figsize=(13, 11))
    ConfusionMatrixDisplay(cm, display_labels=label_ko).plot(
        ax=a, colorbar=True, values_format=".2f", cmap="Blues", xticks_rotation=0)
    a.set_title(f"Keras {name.upper()} 혼동행렬  (검증 정확도 {acc:.3f})\n"
                f"행 = 실제 표지판, 열 = 모델 예측, 숫자 = 비율(행 합 = 1)", fontsize=12)
    a.set_xlabel("모델이 예측한 표지판"); a.set_ylabel("실제 표지판")
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_keras_{name}.png", dpi=130); plt.close(fig)

# 3) 클래스별 정확도
fig, a = plt.subplots(figsize=(13, 5))
x = np.arange(len(labels)); w = 0.27
for i, name in enumerate(("ann", "dnn", "cnn")):
    acc = [(preds[name][y_true == k] == k).mean() for k in range(len(labels))]
    a.bar(x + (i - 1) * w, acc, w, label=name.upper())
a.set_xticks(x); a.set_xticklabels(label_ko, rotation=45, ha="right")
a.set_ylabel("정확도"); a.set_ylim(0.6, 1.0); a.grid(axis="y", alpha=.3); a.legend()
a.set_title("Keras — 클래스별 정확도")
fig.tight_layout(); fig.savefig(f"{OUT}/per_class_keras.png", dpi=130); plt.close(fig)

print("\n=== Keras 요약 ===")
for name in ("ann", "dnn", "cnn"):
    print(f"  {name.upper():4}  best val {max(history[name]['val_accuracy']):.4f}")
