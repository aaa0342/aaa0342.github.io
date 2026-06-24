---
title: "[논문리뷰] Balancing Quality and Alignment in Knowledge Distillation : Leveraging Various Teacher Models and Sequence Types"
---

## 요약

이 논문은 2025 한국컴퓨터종합학술대회에서 발표된 논문으로이다. 이 논문에서는 지식 증류 성능이 교사 모델과 입력 시퀀스 선택에 따라 크게 달라진다는 문제를 지적하며, 이를 해결하기 위한 방법 3가지 (추가 기반, 순차적 훈련, 전이 기반)을 제시한다. 

실험을 통하여 단순히 교사 모델의 성능은 충분하지 않고, 교사-학생 간 파라미터 격차와 입력 시퀀스의 특성까지 균형 있게 고려해야함을 보였다.

## 서론

LLM은 높은 성능을 지님과 동시에 자원 소모와 추론 비용 문제를 갖고 있다. 이에 따라서 대형 모델의 지식을 소형 모델로 이전하여 효율성을 확보하는 압축 기법인 **지식 증류 (Knowledg Distilation)** 기법이 제안되었다.

기존 KD 연구들은 고정된 입력 데이터셋과 Kullback-Leilbler Divergence 기반 손실함수를 활용하여 고성능의 교사 모델의 출력을 일반적으로 더 적은 파라미터를 가진 학생 모델이 모방하도록 하는 방식이 주를 이루어졌다는 것을 지적한다.

이 논문에서 문제로 지적하는 것은, 교사 모델과 입력 시퀀스의 선택이 학생 모델 성능에 영향을 준다는 것을 기존 방식은 간과하고 있다는 것이다.

<img src="{{ site.baseurl }}/assets/images/paperReview/Balancing Quality and Alignment in Knowledge Distillation/Balancing Quality and Alignment in Knowledge Distillation_그림1.png" alt="Rank-Weighted Reward Optimization algorithm" style="display: block; margin: 0 auto;">

이 논문에서 지적하는 것을 조금 더 구체적으로 보면 다음과 같다.

1. 고품질 입력인 ground truth 시퀀스와 학생 모델 생성 출력간 입력 학습-추론 불일치가 생긴다. 이는 다시 말해서, 학생 모델이 학습할 때 보는 입력 시퀀스와 실제 추론할 때 보는 시퀀스가 다르다는 것이다. 

    이는 GT-prefix를 보고 다음 토큰 예측과 실제 추론 시 학생 모델이 만든 prefix를 보고 다음 토큰 예측하는 것의 차이를 말한다. 

2. 교사 모델의 성능이 좋을 수록, 학생 모델 간 과도한 파라미터 크기 격차가 발생하여 학생 모델의 alignmnent를 저해할 수 있다. 

    이는 교사 모델의 성능이 아무리 좋아도 학생 모델의 파라미터 수 제한 때문에 발생하는 것으로, 학생 모델 자체의 성능 제한 때문에 교사 모델의 출력 분포 등을 효과적으로 학습할 수 없기 때문에 발생하는 문제다.

3. 교사 모델과 학생 모델의 alignment는 학습에 도움이 되지만, 이를 위해 파인튜닝을 시도할 경우 오히려 교사 모델의 성능이 낮아질 수 있다.

    이는 교사 모델을 학생 모델을 "잘 가르치기" 위해 파인 튜닝하게 된다면 교사 모델의 본래 "강한 교사"의 특징을 잃게 된다는 것을 지적하는 것이다.

그림 1은 이러한 문제들 때문에 발생하는 성능 편차를 보여준다.

## 대형 언어 모델의 지식 증류

KD의 대표적인 방법인 KLD 손실함수는 학생 모델이 교사 모델의 출력 분포를 모방하도록 학습시킨다. KLD를 토큰 단위 손실로 분해하여, 다음 토큰 예측의 분포를 근사하도록 학습이 이루어진다.

예시를 들어보면 다음과 같다.

입력: 나는 오늘 학교에

예측할 다음 토큰: 갔다

기존 연구들은 주로 파라미터 크기 차이만 존재하는 단일 교사 모델만 사용하고, 입력 시퀀스는 학슴 데이터 셋과 동일한 데이터를 고정하고 학습을 진행했으며, 이 논문에서는 모델 선택과 입력 시퀀스 선택의 중요성을 간과했다는 것을 문제삼는다.

입력 시퀀스 선택이라는 말이 추상적으로 들리지만, 사실 간단하다. 교사 모델의 출력 분포를 학생 모델일 학습하기 위해서는 교사 모델의 입력이 필요하다.

교사 모델의 입력으로는 ground truth가 들어갈 수도 있고, 학생 모델의 출력 결과가 들어갈 수 있다. 이렇게 교사 모델에게 적절한 입력 시퀀스를 선택하는 것을 이 논문에서 지적을 한다고 보면 된다.

## 다양한 교사 모델과 다양한 시퀀스 유형 활용 방법론

<img src="{{ site.baseurl }}/assets/images/paperReview/Balancing Quality and Alignment in Knowledge Distillation/Balancing Quality and Alignment in Knowledge Distillation_그림2.png" alt="Rank-Weighted Reward Optimization algorithm" style="display: block; margin: 0 auto;">

### 1. 추가 기반 방법

작은 교사 모델에서 고품질 GT 시퀀스를 입력으로 사용하고, 교사 모델은 학생이 생성한 출력(=SSO)을 사용하는 방식으로 각 교사 모델의 특성에 맞춰 최적의 입력을 선택함으로써 학습 효과를 극대화한다.

### 2. 순차적 훈련 방법

모델 간의 파라미터 크기 격차를 극복하기 위해 제시된 방법이다. 작든 교사로 먼저 학습을 수렴시킨 후, 더 큰 교사로 추가로 학습하는 방법이다. 이러한 점진적인 학습을 제공하여 안정적으로 성능을 향상시킨다.

### 3. 전이 기반 방법

GT와 SSO의 장점을 모두 살리기 위한 방법이다. 학생 출력으로 파인튜닝된 교사 모델과 원래 교사 모델을 함께 사용하는 방법으로, 두 교사의 출력을 가중 합산하여 점진적인 모델로 전이하는 구조이다.

## 실험 결과

학생 모델은 OpenLLaMA-3B, 교사 모델은 OpenLLaMA-7B와 OpenLLaMA-13B 모델을 선택하였다고 한다.

앞서 말했듯이, 7B 교사 모델은 3B와 파라미터 차이가 13B에 비해 상대적으로 적기 때문에 alignment가 높게 된다.

데이터 셋은 LIMA, 손실함수는 Jensen-Shannon Divergence로 통일하였다고 하며, 성능 평가는 GPT-4 Eval, Rouge-L 메트릭을 사용하였다고 한다.

<img src="{{ site.baseurl }}/assets/images/paperReview/Balancing Quality and Alignment in Knowledge Distillation/Balancing Quality and Alignment in Knowledge Distillation_표1.png" alt="Rank-Weighted Reward Optimization algorithm" style="display: block; margin: 0 auto;">

표 1을 통해서, 제안된 방법들이 기존 단일 교사, 단일 입력 방식 대비 일관된 성능 향상이 보였다는 것을 알 수 있다.

추가 기반 방법은 단일 전략보다 성능이 개선이 되지만 성능 향상 폭이 제한적이었다. 이는 교사 모델과 학생 모델의 입력이 다르기 때문에 학생 모델이 출력 분포를 학습할 때 서로 방해가 될 수 있기 때문이라고 한다.

순차적 훈련 방법 역시 안정적 수렴을 통해 상당한 성능 향상을 보였고, 최종적으로 전이 기반 방법이 가장 높은 성능을 기록했다.