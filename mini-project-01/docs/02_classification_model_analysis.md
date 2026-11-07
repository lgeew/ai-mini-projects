# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 수치와 관찰을 기록합니다. `___`는 실제 값 또는 1~2문장으로 바꾸세요. Precision·Recall·F1은 구매 클래스(`Revenue=1`) 기준입니다.

## 1. 기준 모델 및 평가 지표 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)
- **주 평가 지표:** F1 (구매 클래스)
- **보조 지표:** Precision, Recall, Accuracy
- **모델 비교에 사용한 데이터:** 
DummyClassifier
- **DummyClassifier — Validation Accuracy / F1:** 0.845 / 0.00
- **LogisticRegression — Train F1 / Validation F1:** 0.505 / 0.49
- **LogisticRegression — Validation Precision / Recall:** 0.734 / 0.367
- **LogisticRegression의 Validation 혼동행렬:** 
TN 1526 / FP 38 / FN 181 / TP 105

**분석:** 불균형 데이터에서 기준 모델의 Accuracy와 F1이 다르게 나타나는 이유를 작성합니다.  
데이터의 84.5%가 미구매(0)인 불균형 상태라 다수 클래스(0)만 잘 맞춰도 Accuracy는 0.882로 높게 나옴. 반면 소수 클래스인 구매(1)의 실제 예측 성공률(Recall 0.367)이 낮아, 구매 클래스의 정밀도와 재현율을 종합 평가하는 F1(0.49)은 대폭 낮게 측정.

## 2. PyTorch MLP-A 구현 및 학습 (CP-05)

- **신경망 구조:** 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1)
- **손실 함수 / 옵티마이저:** `BCEWithLogitsLoss` / Adam
- **학습 조건:** 최대 25 epoch, batch size 256, learning rate 0.001
- **Validation F1이 가장 높았던 에포크:** 25
- **선택된 MLP-A의 Train F1 / Validation F1:** 0.709 / 0.633
- **MLP-A Validation Precision / Recall:** 0.717 / 0.566

**분석:** Train loss와 Train·Validation F1 그래프에서 확인한 학습 경향을 작성합니다.  
에폭이 진행됨에 따라 Train loss는 꾸준히 하락하고 Train F1은 상승했으나, 학습 초중반 이후 Validation F1이 정체하거나 소폭 변동하는 양상을 보였습니다. 기본 손실 함수만 사용한 경우 불균형 데이터의 한계로 인해 검증 성능이 일정 수준 이상으로 향상되는 데 한계가 있음을 확인함.

## 3. 클래스 가중 손실과 MLP 비교 (CP-06)

- **MLP-B 변경 사항:** 양성 클래스 가중치(`pos_weight`)를 적용한 `BCEWithLogitsLoss`
- **Train 데이터의 `pos_weight`(미구매 수 ÷ 구매 수):** 5.46
- **MLP-B의 최적 에포크:** 17
- **MLP-B Train F1 / Validation F1:** 0.692 / 0.645
- **MLP-B Validation Precision / Recall:** 0.575 / 0.734
- **MLP-A와 MLP-B의 Validation F1 차이:** 0.012
- **LogisticRegression과 두 MLP의 성능 비교에서 발견한 점:** 
선형 모델에 비해 신경망(MLP) 모델들이 비선형 패턴을 학습하여 F1 점수가 대폭 향상되었으며, 특히 가중 손실을 적용한 MLP-B가 소수 클래스 탐지 능력을 크게 개선함.
- **Train·Validation F1 차이에서 관찰한 점:** 
MLP-B의 Train-Valid F1 Gap(0.047)이 MLP-A(0.077)보다 작아, 가중 손실 적용이 과적합(Overfitting)을 완화하고 일반화 성능을 높이는 데 기여함.

**분석:** 양성 클래스 가중치를 적용한 뒤 Precision·Recall·F1이 어떻게 달라졌는지 작성합니다.  
가중치를 부여한 결과 재현율(Recall)이 크게 상승(0.566 -> 0.734)하여 실제 구매 고객을 훨씬 잘 찾아내게 되었으며, 그로 인해 종합 성능 지표인 F1-score가 향상됨.

> 전체 실험 결과는 Notebook의 `results_df`와 `results/02_model_comparison.csv`를 참고합니다.

## 4. 예측 오류 및 임계값 분석 (CP-07)

- **비교한 두 신경망:** MLP-A / MLP-B
- **기본 임계값 0.5에서 MLP-A의 FP / FN:** FP= 64 / FN= 124
- **기본 임계값 0.5에서 MLP-B의 FP / FN:** FP= 155 / FN= 76
- **임계값 0.3에서 Precision·Recall 및 FP·FN 변화:** 
A - FP= 125, FN= 92, B - FP= 283, FN= 46
- **임계값 0.7에서 Precision·Recall 및 FP·FN 변화:** 
A - FP= 20, FN= 185, B - FP= 99, FN= 111
- **Validation F1이 가장 높은 MLP·임계값 조합:** MLP-B (weighted BCE) 0.5
- **선택한 MLP의 최적 에포크 수:** 17
- **(선택) PR 곡선 또는 추가 임계값 실험에서 확인한 점:** 
임계값을 낮출수록(0.5 -> 0.3) Recall은 크게 증가하나 오탐(FP)이 급증하고, 임계값을 높일수록(0.5 -> 0.7) Precision은 상승하나 구매자를 놓치는 위음성(FN)이 급증함. 

구매자를 놓치는 FN 비용과 미구매자에게 마케팅 리소스를 낭비하는 FP 비용의 균형을 종합적으로 고려하여, F1-score가 0.645로 가장 높고 Precision(0.575)과 Recall(0.734) 간 조화가 가장 우수한 MLP-B, 임계값 0.5를 최적 조합으로 선택함.

> 임계값별 상세 결과는 `threshold_df`와 `results/02_threshold_comparison.csv`를 참고합니다.

## 5. 최종 신경망과 Test 평가 (CP-08)

- **Test 확인 전에 확정한 MLP 설정 / 임계값:** 
MLP-B (weighted BCE) / threshold: 0.5
- **최종 재학습 데이터(Train + Validation):** 10480행
- **최종 재학습 에포크 수 / 가중 손실 적용 여부:** 17 / Y, weighted BCE
- **Test Accuracy / Precision / Recall / F1:** 
0.858 / 0.526 / 0.843 / 0.648
- **Test 혼동행렬:** TN 1347 / FP 217 / FN 45 / TP 241
- **모델 번들(`02_final_classifier.joblib`)·설정 JSON 저장 및 재로딩 확인:** True
- **`predict_purchase()` 실행 결과:** 
{'purchase_probability': 0.0058, 'prediction': 0, 'threshold': 0.5}

**해석:** 최종 Test 결과와 동일한 모델·임계값의 Validation 결과를 비교해 알게 된 점을 작성합니다.  
Test F1(0.648)이 Valid F1(0.645)과 거의 동일하게 유지되어 과적합 없이 일반화가 잘 이루어졌음을 확인함. 특히 Test 세트에서 Recall이 0.843까지 상승하여 실제 구매자의 약 84%를 탐지해내는 우수한 성과를 보였음.

> Test 결과는 모델 선택에 사용하지 않습니다. 최종 Test 수치는 Notebook과 이 문서에만 기록합니다.

## 6. 실험 요약

- **가장 중요한 실험 결과:** 
단순 Accuracy 중심 평가에서 벗어나, 클래스 가중 손실(Weighted BCE)과 임계값(0.5) 최적화를 적용한 결과 선형 모델(F1 0.490) 대비 최종 Test F1 0.648, Recall 0.843으로 소수 클래스 탐지 성능을 크게 향상시켰음
- **남아 있는 모델의 한계:** 
구매 탐지율(Recall)을 극대화함에 따라 미구매자를 구매자로 잘못 판단하는 오탐(FP=217)이 발생하여 마케팅 비용 리스크가 존재하며, 향후 비즈니스 투자 대비 수익률 기반의 임계값 미세조정이 추가로 필요함.

> **최종 MLP 설정·임계값의 선택과 근거**는 `adr.md`의 의사결정 ④에 작성합니다.
