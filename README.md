# 교통 표지판 이미지 분류: ANN · DNN · CNN 구조 비교

한국 도로 표지판 이미지 10종을 같은 조건의 ANN·DNN·CNN으로 학습하고,
깊이와 파라미터 수가 아니라 파라미터를 배치한 구조가 성능을 가르는지 비교한 실험.
같은 구조를 PyTorch와 Keras로 각각 구현해 결과를 교차 확인.

## 검증 항목
1. 완전연결층을 더 깊게 쌓으면, 즉 ANN에서 DNN으로 가면 정확도가 오르는가
2. 성능을 가르는 것은 파라미터의 양인가, 파라미터를 배치한 구조인가

## 데이터
- AI-Hub 신호등·도로표지판 인지 영상(수도권) 표지판 crop 데이터, 104개 분류 코드
- 장수 상위 10개 코드 선택, 코드당 1,000장 추출, 짧은 변 24px 미만 이미지 제외: `build_dataset.py`
- 입력: 32×32 RGB

| 코드 | 표지판 | 코드 | 표지판 |
|---|---|---|---|
| 213 | 좌회전금지 | 308 | 좌우회전 |
| 214 | 우회전금지 | 309 | 직진및좌회전 |
| 218 | 주정차금지 | 311 | 유턴 |
| 219 | 주차금지 | 322 | 횡단보도 |
| 221 | 차높이제한 | 329 | 비보호좌회전 |

## 모델 구조

`config.py` 설정을 PyTorch·Keras 구현이 공유.

| 모델 | 구조 | 파라미터 |
|---|---|---|
| ANN | Flatten → Dense 256 | 789K |
| DNN | Flatten → Dense 1024 · 512 · 256 · 128 | 3,837K |
| CNN | Conv 32 · 64 · 128, 커널 3, 풀링 2 → Dense 256 | 620K |

공통: Dropout 0.5. PyTorch는 channels-first, Keras는 channels-last 입력 축 사용.

## 학습 조건
- Adam, 학습률 1e-3, 배치 32, 최대 25 에포크
- 검증 손실 5회 연속 미개선 시 조기 종료
- 시드 42 고정, 모델 생성 직전마다 재설정

## 결과

검증 정확도, 32×32 입력. 10종 무작위 추측 기준 0.10.

| 모델 | PyTorch | Keras |
|---|---|---|
| ANN | 0.90 | 0.90 |
| DNN | 0.90 | 0.90 |
| CNN | 0.95 | 0.96 |

- DNN: ANN 대비 파라미터 약 5배, 층 4배에도 정확도 차이 없음
- CNN: 가장 적은 파라미터로 가장 높은 정확도

입력 크기별 비교: `sweep_size.py`, PyTorch, 결과 파일 `figs/sweep_size.csv`

| 입력 | ANN | DNN | CNN |
|---|---|---|---|
| 16px | 0.913 | 0.906 | 0.947 |
| 24px | 0.899 | 0.901 | 0.948 |
| 32px | 0.905 | 0.900 | 0.946 |
| 48px | 0.891 | 0.873 | 0.947 |
| 64px | 0.893 | 0.899 | 0.948 |

입력이 작아도 ANN·DNN 0.87~0.91 유지, CNN은 모든 크기에서 ANN 대비 3.4~5.6%p 우위.

## 한계
- 10종, 종당 1,000장 규모의 소규모 실험
- 모델당 1회 실행, 재실행 시 CNN 정확도 3~5%p 변동
- 깊이·연산 종류·파라미터 수가 동시에 달라 요소별 효과 분리 불가

## 실행

Python 3.11 사용. TensorFlow 휠이 없어 3.13 이상은 설치 불가.
PyTorch GPU 버전은 인덱스를 지정해 설치.

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tensorflow pillow matplotlib scikit-learn pandas

python build_dataset.py --src "<AI-Hub 데이터 Training 폴더>"   # crop/<코드>/ 생성
python train_torch.py                # ANN·DNN·CNN 학습, --model cnn 으로 하나만
python train_keras.py
python analyze.py                    # 학습곡선·혼동행렬·클래스별 정확도 → figs/
python analyze_keras.py              # Keras 버전 → figs_keras/
python sweep_size.py                 # 입력 크기별 비교
```

`run_colab.ipynb`: Colab 실행용 노트북

## 파일 구성
```
config.py            클래스·입력 크기·모델 구조 공통 설정
build_dataset.py     상위 10개 코드 crop 데이터 구성
prepare_data.py      초기 방식: AI-Hub 라벨 json에서 표지판 잘라내기
src/torch_impl/      PyTorch ANN·DNN·CNN
src/keras_impl/      Keras ANN·DNN·CNN
train_torch.py       PyTorch 학습 루프 직접 구현
train_keras.py       Keras fit 기반 학습
analyze.py           발표용 그림 생성
analyze_keras.py     Keras 결과 그림
regen_confusion.py   ANN·CNN 혼동행렬 따로 생성
sweep_size.py        입력 크기별 정확도 비교
```
