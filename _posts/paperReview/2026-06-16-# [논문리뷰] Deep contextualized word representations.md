# Abtstract

이 논문에서 제안하는 것은 

1) 단어 사용의 복잡한 특징

2) 단어 사용이 언어적 맥락에서 어떻게 달라지는가

이 두 가지를 모델링 하는 deep contextualized word representation이다. 이러한 모델링된 벡터 표현을 deep biLM의 내부 상태에 대한 함수라고 표현하며, 이를 통해서 이미 존재하는 모델에 쉽게 더해지고 NLP 과제들에서 SOTA를 향상시킬 수 있다고 한다.

# Introduction

사전 학습된 단어 표현은 매우 중요하지만 고퀄리티의 표현을 얻는 것은 여전히 어렵다고 지적한다. 

이 논문에서는 

(1) 단어 사용의 복잡한 특징

(2) 단어 사용의 언어적 맥락에 따른 다양성

이 두 가지를 모델링하는 deep contextualized word representaion의 새로운 타입임과 동시에 존재하는 모델에 쉽게 통합될 수 있는 모델을 제안한다.

biLM의 앞 LM과 뒤 LM의 목적함수가 공통으로 두어 ELMo를 생성한다고 한다. 이때 각 task에 대해서 각 입력 단어에 대한 LSTM의 각 layer 출력값들의 선형 조합을 학습한다는 점에서 biLM의 모든 내부 layer에 대한 함수라고 한다.

내부 상태를 조합하여 풍부한 단어 표현을 만들어낼 수 있다고 한다.

- Higher Level : context-dependent 측면을 잘 포착

- Lower Level : 문법 측면을 잘 포착

이러한 신호를 노출하여 학습된 모델이 각 task에 맞게 준지도 학습의 유형을 선택할 수 있게 된다고 한다.

또한 ELMo는 존재하는 모델에 쉽게 추가될 수 있으며 SOTA의 향상을 이끌어냈다고 하며 최종적으로 LSTM의 top layer만 사용하는 것을 능가한다고 말했다.

# Related Work

**사전학습된 단어 벡터**

사전학습된 단어 벡터들은 중요한 요소이지만 문맥 상관 없이 하나의 고정된 벡터만 생성한다는 단점이 있다고 한다. 

**subword 활용 벡터**

ELMo 역시 character convolution을 통해 subword 정보를 활용한다고 한다. 추가적으로 biLM 문맥 표현울 통해 다의어 정보를 만들며 downstream task로 전달이 가능하다고 했다.

**context-dependent / contextual-embedding**

이러한 접근법은 대규모 데이터 셋을 통해서 장점을 갖고 있으며, 이 논문 역시 단일언어 데이터를 충분히 학습(30M 문장 말뭉치)하여 최대 이점을 얻는다고 한다. 

**deep biRNN**

이전 work에서 deep biRNN의 각 층마다 각기 다른 정보를 갖고 있음을 보였다고 한다. ELMo에서도 마찬가지로 비슷한 신호를 보였으며, downstream task에 대한 모델들의 학습에 도움이 된다고 말한다.

**Dai and Le / Ramachandran et al**

task specific 지도학습 파인튜닝을 활용하는 반면, 사전학습된 biLM의 가중치는 고정하고 추가적인 task sepcific model을 학습하도록 하여 downstream 데이터셋의 사이즈가 작아도 biLM 표현을 활용할 수 있다고 한다.

# ELMo : Embeddings from Language Model

## ELMo

ELMo는 biLM에서의 레이어 표현들의 조합이라고 표현한다.

ELMo는 다음과 같은 원리로 생성된다.

1. 각 토큰 $t_k$에 대해서 $L-Layer$ biLM은 총 $2L+1$ 표현을 생성한다.

    $$
    R_k = \left\{ x^{LM}_k, \overrightarrow{h}^{LM}_{k,j}, \overleftarrow{h}^{LM}_{k,j} \mid j = 1, \dots, L \right\}
    = \left\{ h^{LM}_{k,j} \mid j = 0, \dots, L \right\}
    $$

    - $h^{LM}_{k, 0}$ 은 token layer을 의미하고
    - $h^{LM}_{k,j} = [\overrightarrow{h}^{LM}_{k,j}; \overleftarrow{h}^{LM}_{k,j}]$ 은 biLSTM에서 Forward LM과 backward LM layer 의미한다.

2. downstream model에게 전달되기 위해서 $R$의 모든 레이어들이 하나의 벡터로 압축이 된다.

    $$
    ELMo_k = E(R_k; \theta_e)
    $$

3. 더 일반적으로, 모든 biLM layer에 대해서 task specific weighting을 적용한다.

    $$
    ELMo^{task}_k = E(R_k ; \theta^{task}) = \gamma^{task} \sum^L_{j=0} s^{task}_j h^{LM}_{k,j}
    $$

    - $s^{task}$ : 소프트맥스 정규화 가중치
    - $\gamma^{task}$ : 전체 ELMo 벡터를 스케일링하는 스칼라 파라미터 

    biLM에서 각 layer의 activation이 다른 분포를 갖는 경우가 있기 때문에 가중치 적용하기 전 layer normalization을 적용하는 것이 도움이 되는 경우가 있다고도 한다.

## ELMo의 활용

이후 NLP task의 향상을 위해서는 biLM을 활용한다고 한다.

1. 대부분의 supervise NLP model은 가장 낮은 layer에서 공통의 구조를 갖기 때문에 supervised model의 가장 낮은 layer를 활용한다.

    - 토큰 시퀀스 $(t_1, ..., t_N)$이 주어졌을 때 각 토큰 위치에 대해서 사전 학습된 단어 임베딩 또는 경우에 따라 문자 기반 표현을 활용하여 context-independent 토큰 표현 x_k를 생성한다.

2. 모델이 biRNN, CNN 등을 활용하여 context-sensitive 표현 h_k를 생성한다. 

    - biLM 모델의 가중치를 고정한다
    - ELMO 벡터인 $ELMo^{task}_k$를 x_k를 결합하여 만든 $[x_k; ELM0^{task}_k]$를 task RNN에 전달한다.

최종적으로 $h_k$를 $[h_k; ELMo^{task}_k]$ 대체해서 성능 향상을 이끌었다고 한다. 또한 supervised model은 변하지 않기 때문에, 더 복잡한 신경망 모델에도 적용이 가능하다고 한다.

또한 ELMo에 drop out을 적용하거나 ELMo의 가중치를 $\lambda\lVert w \rVert_2^2$ 정규화를 하는 것을 통하여 bias를 부여해 ELMo 가중치가 biLM 레이어의 평균치와 가깝게 유지되어 이점을 얻는 경우도 있다고 한다.

몇몇 상황에서는 biLM을 도메인 특화 데이터로 특화하는 것이 복잡도를 낮추고 성능도 증가시키는 경우가 많다고 한다. 이는 biLM으로 도메인 전환이 이루어진 것으로 보인다. 따라서 최종적으로는 파인 튜닝된 biLM을 대부분의 경우에 사용했다고 한다.

## ELMo의 구조

사전학습된 biLM은 Jo ́zefowicz et al. (2016) and Kim et al. (2015)와 유사하지만 양방향이 같은 목적함수를 쓰고 LSTM layer 사이 (1번째와 2번째 layer 사이) residual connection을 추가했다는 점에서 차이가 있다고 한다.

나아가 모델의 사이즈가 커지면 ELMo 차원이 커져 downstream 연산의 복잡해지기 때문에 단일 최고 모델 임베딩과 hidden dimension을 절반으로 줄여서 활용했다고 한다. 최종적으로 L = 2 biLSTM 모델을 사용했다고 한다. 

context insensitive type 표현은 2048 character n-gram convolution filter와 feedforward 계열 layer를 활용하였고, 그 결과 biLM은 각 입력 토큰별로 총 3개의 layer 표현을 얻게 된다.

- character CNN
- LSTM의 layer 2개

특히, character CNN를 통해서 biLM 학습 말뭉치에 존재하지 않는 단어도 표현을 할 수 있게 된다고 한다. 

## Evaluation

<img src="{{ site.baseurl }}/assets/images/paperReview/ELMo/ELMo_Table_1.png" alt="ELMo Architecture" style="display: block; margin: 0 auto;">

ELMo를 단순히 더하기만 하여도 6개의 벤치마크 NLP task에서 새로운 SOTA를 만들었으며, 상대 오류를 6~20% 줄이는 효과를 만들었다고 한다.

참고로 Relative Error reduction은 다음과 같이 계산한다.

$$
\frac{\text{baseline error} - \text{ELMo error}}{\text{baseline error}}
$$

## Analysis

이 논문에서는 ELMo 표현의 내적 측면을 설명하기 위해 다음의 추가적 분석을 진행하였다.

- layer 가중치 구조의 변화에 대한 분석
- ELMo를 어디에 넣어야 하는지에 대한 분석
- biLM으로 어떠한 정보를 포착할 수 있는지에 대한 분석
- ELMo의 효용성에 대한 분석
- 학습된 가중치의 시각화 분석

### 1. layer 가중치 구조의 변화

ELMo의 활용 부분을 보면 $\lambda$를 적용하여 정규화를 한다는 내용이 있었다. 이 분석에서는 이 $\lambda$의 값의 변화에 따라 layer 가중치가 어떻게 변화하고 그에따른 성능 비교를 한다.

<img src="{{ site.baseurl }}/assets/images/paperReview/ELMo/ELMo_Table_2.png" alt="ELMo Architecture" style="display: block; margin: 0 auto;">

$\lambda$가 $\lambda = 1$처럼 큰 값을 갖는 경우에는 layer의 가중치가 비슷하게 평균되는 경향을 보인다고 하였다.

반면에 $\lambda$가 $\lambda = 0.001$처럼 작은 값을 갖는 경우에는 layer의 가중치가 다양해지는 경향을 보인다고 한다.

$Table2$는 다음을 시사한다.
- baseline(=biLM을 사용하지 않는 모델)보다 biLM을 사용하는 것이 성능이 우세하다
- 모든 biLM의 layer를 활용하는 것이 마지막 layer만 사용하는 것보다  성능 향상이 있다.
- 모든 biLM의 layer를 활용할 때, $\lambda$를 작은 값으로 설정하여 다양한 layer 가중치를 얻게 하는 것이 대부분의 경우 도움이 된다.

### 2. ELMo를 어디에 사용해야 하는가

이 논문에 나오는 task 구조들은 단어 임베딩을 가장 낮은 layer에 입력으로 넣는 방식이지만, ELMo를 task 특화 구조의 biRNN output에 ELMo를 결합하는 것에서 성능 향상이 보였다고 한다.

<img src="{{ site.baseurl }}/assets/images/paperReview/ELMo/ELMo_Table_3.png" alt="ELMo Architecture" style="display: block; margin: 0 auto;">

$Table3$를 통해서 다음을 알 수 있다.

- SQuAD와 SNLI에서는 Input과 Ountput에 모두 ELMo를 결합하는 게 도움이 된다.
- 반면, SRL의 경우에는 Input에만 결합하는 게 도움이 된다.

저자는 이러한 현상이 SQuAD나 SNLI는 attention layer가 biRNN 뒤에 있기 때문에 ELMo를 추가하여 모델이 biLM의 내부 표현을 직접 이해할 수 있게 하는 반면, SRL의 경우는 SRL의 task RNN의 $h_k$가 ELMo의 정보보다 더 중요하기 때문이라고 말한다. 


### 그 외

<img src="{{ site.baseurl }}/assets/images/paperReview/ELMo/ELMo_Table_4-6.png" alt="ELMo Architecture" style="display: block; margin: 0 auto;">


**1) biLM 표현으로 어떠한 정보가 잡히는가?**

이 논문에서 제안한 ELMo를 사용하여 기존 단어 벡터만 사용했을 때보다 성능 향상이 있었기 때문에 biLM의 문맥적인 표현이 반드시 NLP task에 일반적으로 도움이 되는 정보를 갖고 있을 것으로 설명한다.

$Table4$를 통해서 biLM의 표현이 문맥 정도를 사용하여 단어 의미가 모호하지 않게 만든다는 것을 보여준다.

또한 이를 정량적으로 평가하는 방법을 제시한다.

$Table5$는 Word Sense Disambiguation : 단어 의미 구별의 성능을 평가하여 기존 SOTA와 견줄 정도의 성능을 second layer를 통해서 얻을 수 있었다는 것을 보인다.

$Table6$는 POS tagging task : 품사 예측의 성능을 평가하여 WSD와 다르게 first layer를 통해서 task 특화된 biLSTM와 견줄만한 성능을 얻을 수 있었다는 것을 보인다.

결론은 다음과 같다

biLM의 다른 layer들은 다른 종류의 정보를 표현하며, 이것이 왜 모든 biLM layer가 downstream task에서 가장 좋은 성능을 내는지에 대한 근거다.

**2) 샘플 효율성**

ELMo를 모델에 더하는 것은 또한 샘플 효율성을 향상시킨다고 한다. 

샘플 효율성은 다음과 같다.

- SOTA를 달성하기 위한 업데이트 해야하는 파라미터 수
- SOTA를 달성하기 위한 전반적인 학습 데이터 사이즈

ELMo를 사용하여 baseline(= SRL model)에 비해 98% 감소한 epoch 10으로 초대치를 달성했다고 한다.

<img src="{{ site.baseurl }}/assets/images/paperReview/ELMo/ELMo_Figure_1.png" alt="ELMo Architecture" style="display: block; margin: 0 auto;">

$Figure1$는 ELMo를 활용하면 더 적은 학습 데이터셋을 효율적으로 사용할 수 있음을 보여준다. ELMo를 통한 개선은 더 작은 훈련 세트에서 가장 크며 주어진 성능 수준에 도달하는 데 필요한 훈련 데이터의 양을 크게 줄였다고 한다.

**3) 가중치의 시각화**

<img src="{{ site.baseurl }}/assets/images/paperReview/ELMo/ELMo_Figure_2.png" alt="ELMo Architecture" style="display: block; margin: 0 auto;">

$Figure2$는 softmax-normalized learned layer weight를 시각화한다. 

이를 통해 downstream의 input layer에 ELMo를 더하였을 때는 $h_1$, 즉 첫 번째 layer가 선호가 되는 것을 알 수 있고, output layer에 더했을 때는 균형을 이루지만 $h_0$와 $h_1$, 즉 낮은 layer가 더 선호되는 경향을 보임을 알 수 있다.