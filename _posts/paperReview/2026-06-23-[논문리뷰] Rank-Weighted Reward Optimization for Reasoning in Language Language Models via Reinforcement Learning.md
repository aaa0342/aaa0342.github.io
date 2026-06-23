---
title: "[논문리뷰] Rank-Weighted Reward Optimization for Reasoning in Language Language Models via Reinforcement Learning"
---

## 요약

이 논문은 한국정보과학회 2025 한국컴퓨터종합학술대회에서 발표된 논문이다. 이 논문에서는 수학 문제 풀이와 같은 추론 중심 과제에서의 언어 모델 최적화를 위한 critic-free 정책 학습 알고리즘을 제안한다.

해당 방식은 기존 방식의 그룹 내 top-1 응답에만 학습 신호가 집중되는 문제를 해결하기 위해 보상 기분 응답 순위를 부드럽게 확률화하여 모든 샘플이 기여할 수 있도록 하는 구조를 제안한다.

## 서론

기존 강화학습 기반 LLM 최적화는 대부분 가치 함수를 사용하는 actor-critic 구조를 사용한다. 

actor-critic 구조는 다음과 같다.

여기서 actor는 실제 답변을 생성하는 LLM이고, critic은 보상을 예측하는 모델이다. 

1. actor가 예측
2. 실제 보상과 ciritc의 예상 값과 다르면 답변 확률을 조정

    - 실제 보상 < critic 예상 값 : 답변 확률 낮춤

    - 실제 보상 > critic 예상 값 : 답변 확률 높임

이 구조에서 문제점은 critic 모델도 하나의 모델이기 때문에 학습 비용도 크고 advantage 계산의 불안정성을 수반한다는 문제점이 있다.

이를 해결하기 위한 방법이 GRPO로, 가치 함수 없이 학습이 가능하도록 한다.

GRPO는 같은 입력에 대해서 생성된 후보 응답을 그룹으로 묶고 각 응답에 대해 부여된 보상을 z-score 정규화를 통해 relative advantage로 변환 후 정책을 업데이트 한다.

이때 advantage가 양수가 되면 해당 답변의 확률을 올리고 음수가 되면 해당 답변의 확률을 내리게 된다. 

문제점은 대부분의 advantage가 음수로 gradient가 억제가 되고, 따라서 실직적 높은 보상을 받은 샘플에만 gradient가 집중되어 학습 분산이 커지며 그룹 내 순위 정보를 온전히 활용하지 못한다고 한다.

따라서 이 논문에서는 이를 해결하기 위한 해결책으로 **대규모 언어 모델 사후 학습 방식**을 제안한다.

제안 방법은 다음과 같다

1) 보상 기반 순위를 부드럽게 확률화하여 모든 샘플이 기여할 수 있도록 함

2) 추론 중심 태스크에 초점을 맞추어 성능 향상

## 연구 배경

이 논문은 추론 중심 태스크 중에서 특히 수학 문제 풀이에 초점을 두며 GRPO의 문제점을 제시하며 학습 분산을 줄이고 수렴 속도를 빠르게 하는 알고리즘을 제시하기 위해 제안되었다고 한다.

## 본론

기존 GRPO는 각 샘플의 보상에 대해 그룹 평균 및 표준편차를 이용해 advantage를 계산하고, 곧바로 정책 업데이트를 한다. 이때 GRPO의 advantage는 다음과 같다

$$
\widehat{A}_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r)}
$$

이 논문에서는 advantage가 평균이 0으로 정규화되기 때문에 대부분의 값이 음수가 되고, 가장 보상이 큰 하나의 응답만이 양수 gradient를 받게 될 확률이 높다는 것을 지적한다.

따라서 논문에서는 $\widehat{A}_i$ 가 아닌 가중보상

$$
\tilde{A}_i = w_i \dot r_i
$$

을 활용한다.

이때 가중치

$$
w_i = \frac{exp(\widehat{A}_i) / \tau)}{\sum_j exp(\widehat{A}_j / \tau)}
$$

에서 $\tau$ 는 하이퍼파라미터로 낮은 값을 가질 수록 상위 샘플에 집중하고, 높은 값을 가질 수록 균등 분포에 가까워지게 하는 역할을 한다고 한다.

이를 통해 모든 샘플이 그 중요도에 비례해서 gradient에 기여할 수 있게 되고 샘플 효율성과 학습 안정성을 동시에 확보할 수 있게 된다고 한다.

<img src="{{ site.baseurl }}/assets/images/paperReview/Rank-Weighted Reward Optimization for Reasoning in Language Language Models via Reinforcement Learning/Rank-Weighted Reward Optimization for Reasoning_algorithm.png" alt="Rank-Weighted Reward Optimization algorithm" style="display: block; margin: 0 auto;">

최종적으로 손실함수는 다음과 같이 정의된다.

$$
\mathcal{L}(\theta)
=
-
\frac{1}{\sum_{i=1}^{G} |o_i|}
\sum_{i=1}^{G}
\sum_{t=1}^{|o_i|}
\left[
\min
\left(
\rho_{i,t} \cdot \tilde{A}_i,
\operatorname{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) \cdot \tilde{A}_i
\right)
-
\beta D_{KL}(\pi_{\theta} \parallel \pi_{\mathrm{ref}})
\right]
$$

수식에서 

$$
\rho_{i,t}
=
\frac{
\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q, o_{i,<t})
}
$$

는 중요도 샘플링 비율을 나타내고 두 번째 항은 KL 정규화를 통해서 기존 정책에서 크게 벗어나는 것을 방지한다고 한다.

## 실험

**세팅**

이 논문에서는 수학 문제 풀이 과정 과제을 통해 알고리즘을 검증하였다고 한다.

대표적인 수학 추론 벤치마크인 GSM8K를 통해 실험을 하였고 모델을 Qwen 2.5 3B Instruct를 선택하였다고 한다.

파인 튜닝은 QLoRA 기반 4비트 양자화 방식을 수행하였고, Unsloth, vLLM 등 여러 라이브러리를 활용하여 효율성을 증진시켰다고 한다.

또한 핵심인 $\tau$ 는 0.5로 고정하였다고 한다.

GRPO와 논문에서 제안하는 것은 동일한 모델 가중치, 데이터셋 분할, batch size, step 수를 설정하였고 동일한 보상 함수와 샘플을 설정했다고 한다.

**결과**

<img src="{{ site.baseurl }}/assets/images/paperReview/Rank-Weighted Reward Optimization for Reasoning in Language Language Models via Reinforcement Learning/Rank-Weighted Reward Optimization for Reasoning_figure1.png" alt="Rank-Weighted Reward Optimization algorithm" style="display: block; margin: 0 auto;">

그림 1을 통해서 GRPO와 SoftRank-GRPO 모두 초기에는 유사한 증가 양상을 보였으나, 약 150 step 이후에 SoftRank-GRPO가 빠르게 보상이 상승하며 기존 방식에 비해 약 100 step 빠르게 수렴하는 양상을 보였다고 한다.

이후에도 평균 보상이 더 높은 수준에서 유지되며 안정적인 추세를 보이는 것을 통해서 

1. 제안한 알고리즘이 모든 샘플 정보를 부드럽게 활용하여 gradient 분산을 줄이고 빠른 정책 개선을 유도함

2. 기존 GRPO 방식의 top-1 응답에만 보상이 집중되는 단점 때문에 느린 수렴이 발생하게 됨

이라는 사실을 알 수 있다고 한다.

<img src="{{ site.baseurl }}/assets/images/paperReview/Rank-Weighted Reward Optimization for Reasoning in Language Language Models via Reinforcement Learning/Rank-Weighted Reward Optimization for Reasoning_figure2.png" alt="Rank-Weighted Reward Optimization algorithm" style="display: block; margin: 0 auto;">

표 1은 최종 학습 이후의 성능을 보여준다. 이를 통해서 제안한 알고리즘이 수학 추론 태스크와 구조적 추론 문제에서 안정적 학습이 이루어졌음을 알 수 있다고 한다.
