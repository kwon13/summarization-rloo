# Summarization with Reinforcement Learning

**Back to Basics: 강화학습을 적용한 요약 모델 개발**

## 개요

이 프로젝트는 RLOO 학습 방법을 사용하여, 강화학습을 통한 요약 모델의 성능 향상을 탐구합니다. 실험 결과, 강화학습 기반의 접근 방식이 요약 모델의 성능을 효과적으로 향상시킬 수 있음을 보여주었습니다.
![rloo](./src/rloo.png)

## 주요 기능

- G-Eval Metric을 사용한 모델 성능 평가
- 국립국어원 ‘일상 대화 요약 말뭉치 2023’ 데이터셋을 사용한 훈련 및 평가

## 설치

1. **레포지토리 클론**:

   ```bash
   git clone https://github.com/username/summarization-rl.git
   cd summarization-rl
   ```

2. **필요한 패키지 설치**:

   ```bash
   pip install -r requirements.txt
   ```

## 사용법

1. **데이터 전처리**: 대화 데이터셋을 전처리하고, 선호 문장과 비선호 문장을 생성합니다.

2. **모델 학습**: SFT, DPO, RLOO 방법 중 선택하여 모델을 학습시킵니다.

3. **성능 평가**: 학습된 모델을 G-Eval Metric을 사용하여 평가하고, 결과를 비교합니다.

4. **모델 최적화**: 필요에 따라 모델을 Pruning하고, 다양한 파라미터 값으로 추가 실험을 수행합니다.

## 결과

- RLOO 방식은 G-Eval Metric 기준으로 Base 모델 대비 0.12점 향상 (7.62/10)
- RLOO 방식이 가장 높은 성능을 기록하여, 강화학습의 효과를 입증

## 향후 연구

현재 Llama3.1 모델을 Pruning한 경량화된 모델을 구축하고 있으며, 이를 통해 다양한 파라미터 값에 대한 추가 실험을 계획하고 있습니다. 이러한 접근을 통해 더욱 강력하고 효율적인 요약 모델을 개발할 수 있을 것으로 기대합니다.
