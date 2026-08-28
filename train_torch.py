"""
PyTorch 로 ANN / DNN / CNN 을 학습시키고 비교한다.

train_keras.py 와 같은 일을 한다. 차이는 딱 하나:
    Keras 는 model.fit() 한 줄이면 끝나는 것을
    PyTorch 는 for 문으로 직접 써야 한다.

그래서 오히려 좋다. fit() 안에서 실제로 무슨 일이 벌어지는지가
아래 train_one_epoch() 함수에 전부 드러나 있다.

실행:
    python train_torch.py                 # 세 모델 전부
    python train_torch.py --model cnn     # 하나만
    python train_torch.py --epochs 3      # 빠른 확인
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config
from src.torch_impl.models import build_model

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-3
SEED = 42


def load_data(data_dir):
    """crop 폴더를 train/val 로 나눠서 읽는다."""

    # Keras 의 Rescaling(1/127.5, offset=-1) 과 같은 결과를 만드는 전처리.
    #   ToTensor()      : [0,255] uint8  →  [0,1] float, 그리고 (H,W,C) → (C,H,W)
    #   Normalize(.5,.5): (x - 0.5) / 0.5  →  [-1,1]
    # 두 프레임워크가 같은 값을 보게 해야 비교가 성립한다.
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # ImageFolder 는 하위 폴더 이름을 클래스로 삼는다.
    # 폴더명을 '알파벳 순으로 정렬'해서 0,1,2... 번호를 매긴다.
    # Keras 의 image_dataset_from_directory 도 같은 방식이라 라벨 번호가 일치한다.
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # 8:2 로 나눈다. generator 에 seed 를 줘야 매번 같은 분할이 나온다.
    # (안 주면 실행할 때마다 분할이 바뀌어서 모델 간 비교가 무의미해진다)
    n_val = int(len(dataset) * 0.2)
    train_set, val_set = random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    # shuffle 은 train 에만 켠다.
    #   train: 매 epoch 순서를 섞어야 특정 순서를 외우는 걸 막는다.
    #   val  : 성능만 재면 되므로 섞을 이유가 없다.
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, dataset.classes


def run_epoch(model, loader, criterion, optimizer, device, train):
    """
    한 epoch = 데이터 전체를 한 번 훑는 것. (평균 loss, 정확도) 를 돌려준다.

    train=True 면 학습, False 면 평가. 둘의 차이는 두 가지뿐이다.
      1) model.train() / model.eval()
         Dropout 은 학습 때만 뉴런을 꺼야 한다. 평가 때 끄면 성능이 들쭉날쭉해진다.
      2) 역전파를 하느냐 마느냐
    """
    model.train(train)

    total_loss, correct, total = 0.0, 0, 0

    # 평가 때는 기울기를 계산할 필요가 없다.
    # set_grad_enabled(False) 로 꺼두면 메모리도 아끼고 속도도 빨라진다.
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            # 데이터를 GPU 로 보낸다. 모델이 GPU 에 있으면 입력도 GPU 에 있어야 한다.
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)              # 순전파: 예측값(logits) 계산
            loss = criterion(outputs, labels)    # 정답과 얼마나 틀렸는지

            if train:
                # ↓ 이 3줄이 Keras 의 fit() 이 대신 해주던 일 전부다.
                optimizer.zero_grad()   # 이전 batch 의 기울기를 지운다.
                                        # 안 지우면 계속 누적돼서 학습이 망가진다.
                loss.backward()         # 역전파: 각 가중치를 얼마나 바꿔야 할지 계산
                optimizer.step()        # 계산된 방향으로 가중치를 실제로 수정

            # loss 는 batch 평균값이라 batch 크기를 곱해서 누적해야
            # 마지막 batch 가 작을 때도 전체 평균이 정확하다.
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()  # 가장 큰 logit = 예측
            total += labels.size(0)

    return total_loss / total, correct / total


def run(name, train_loader, val_loader, num_classes, epochs, device):
    # 모델을 만들기 직전마다 seed 를 다시 고정한다.
    # 루프 밖에서 한 번만 하면 앞 모델의 학습이 난수를 소모해서
    # 뒤 모델이 다른 초기값을 받게 되고, 조건이 어긋난다.
    torch.manual_seed(SEED)

    model = build_model(name, num_classes=num_classes).to(device)

    # 모델 마지막 층이 softmax 없이 logits 을 그대로 내보내므로
    # CrossEntropyLoss 를 쓴다 (내부에서 log_softmax 를 알아서 해준다).
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 55}")
    print(f"{name.upper()}  -  파라미터 {params:,}")
    print("=" * 55)

    best_loss, best_acc, patience = float("inf"), 0.0, 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)

        # val loss 가 개선되면 기록하고, 5 epoch 연속 안 좋아지면 멈춘다 (early stopping).
        # 고정 epoch 으로 돌리면 DNN 은 과적합된 상태로, CNN 은 덜 학습된 상태로
        # 비교하게 된다. 그건 구조 차이가 아니라 멈춘 시점의 차이다.
        mark = ""
        if val_loss < best_loss:
            best_loss, best_acc, patience, mark = val_loss, val_acc, 0, "  *best"
        else:
            patience += 1

        print(f"  epoch {epoch:>3}/{epochs}  "
              f"train {train_loss:.4f}/{train_acc:.4f}  "
              f"val {val_loss:.4f}/{val_acc:.4f}{mark}")

        if patience >= 5:
            print(f"  val loss 가 5 epoch 동안 개선되지 않아 중단")
            break

    return {"name": name, "params": params, "acc": best_acc, "seconds": time.time() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="crop")
    parser.add_argument("--model", choices=["ann", "dnn", "cnn"], default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cpu":
        print("경고: GPU 를 못 쓴다. CPU 로는 매우 느리다.")

    train_loader, val_loader, class_names = load_data(args.data_dir)
    print(f"클래스 {len(class_names)}개: {class_names}")
    print(f"train {len(train_loader.dataset):,}장 / val {len(val_loader.dataset):,}장")

    targets = [args.model] if args.model else ["ann", "dnn", "cnn"]
    results = [run(n, train_loader, val_loader, len(class_names), args.epochs, device)
               for n in targets]

    print(f"\n{'=' * 55}")
    print(f"{'Model':<6} {'Params':>12} {'Val Acc':>9} {'학습(s)':>9}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<6} {r['params']:>12,} {r['acc']:>9.4f} {r['seconds']:>9.1f}")
    print("=" * 55)
    print(f"\n무작위 추측 = {1 / len(class_names):.3f}  (이걸 못 넘으면 학습이 안 된 것)")


if __name__ == "__main__":
    main()
