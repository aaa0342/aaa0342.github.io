---
title: "[논문리뷰] Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation"
---

## Abstract

이 논문에서는 RNN Encoder-Decoder라는 새로운 신경망 모델을 제시한다. 해당 모델은 두 개의 RNN이루어져 있다. 하나는 심볼의 시퀀스를 인코딩하여 고정된 길이의 벡터 표현을 생성하고, 또 다른 모델은 벡터 표현을 또 다른 심볼의 시퀀스로 디코딩을 하는 역할을 한다. 두 개의 RNN을 공통으로 학습하여 타겟 시퀀스의 조건부 확률을 최대화한다고 한다.

제안된 모델을 기존에 존재하는 log-linear 모델에 추가적인 피처로 활용했을 때 의미론적, 구문론적 표현을 잘 학습하여 성능 향상을 이끌어냈다고 한다.

## Introduction

심층 신경망이 발달하면서 다양한 분야에서 좋은 성공을 이끌어내고 있고, 그 중에서 이 논문은 Statistical Machin Translation (= SMT)에 초점을 둔다. SMT에서도 심층 신경망이 활용이 되며 대표적으로 feedforward neural networ를 예시로 든다.

이러한 흐름 속에서 이 논문은 새로운 신경망 구조를 제안한다. 해당 기존 구문 기반 SMT에 일부로 사용이 가능하며, 두 개의 RNN으로 이루어져있다. 한 개의 RNN은 가변 길이의 소스 시퀀스를 고정 길이의 벡터로 인코딩하는 인코더, 다른 하나는 고정 길이의 벡터를 가변 길이의 타겟 시퀀스로 디코딩한다. 

두 개의 RNN이 모두 함께 학습이 되며 주어진 시퀀스에 대해서 타겟 시퀀스의 확률을 최대화하도록 학습이 된다고 한다. 거기에 추가적으로 정교한 hidden unit을 제안하여 학습을 보다 쉽게하고 메모리 용량을 향상시켰다.

제안된 모델 평가는 영어를 프랑스어로 번역하는 과제를 통해서 이루어졌다.주어진 영어 구문을 알맞은 프랑스어로 번역될 확률을 학습하게 되고, 이후 모델이 SMT의 일부로 활용되어 각 구분 테이블의 구분쌍을 점수화하도록 하였고 성능 향상으로 이끌었다고 한다.

또한 존재하는 모델돠 구문 점수를 비교하여 정량적으로 평가하였다고 하고, 그 결과 제안한 모델이 구문 테이블의 언어적 규칙성을 잘 포착하는 것을 보인다고 한다.

요약하자면 다음과 같다

**두 개의 RNN으로 이루어진 RNN Encoder-Decoder를 제안하였고, 제안된 모델은 구문의 의미론적, 문법적 구조가 잘 보존되는 연속적인 공간 표현을 학습하여 성능 향상으로 이끌었다.**

## RNN Encoder-Decoder

### RNN

이 논문에서 자주 등장하는 RNN에 대해서 살펴보자.

RNN은 hidden state $h$와 선택적 output $y$로 이루어진 신경망 모델로, 가변 길이 시퀀스 $x = (x_1, ..., x_T)$를 입력받는다.

각 step $t$마다 hidden state $h_{\<t\>}$가 업데이트 아래 식으로 업데이트가 된다.

$$
h_{\langle t \rangle} = f(h_{\langle t-1 \rangle}, x_t)
$$

RNN은 시퀀스의 다음 기호를 예측하도록 학습함으로써 시퀀스에 대한 확률 분포를 학습하게 된다. 따라서, 각 시점 $t$ 에서의 출력은 $p(x_t \| x_{t-1}, ..., x_1)$이 된다.

예를 들어 여러 후보에 대한 확률은 softmax를 활용하여 다음과 같이 결과가 출력될 수 있다.

$$
p(x_{t,j} = 1 | x_{t-1}, ..., x_1) = \frac{exp(w_j h_{\langle t \rangle})}{\sum^K_{j'=1} exp(w_{j'}h_{\langle t \rangle})}
$$

그리고 이러한 확률을 모두 활용해서 시퀀스 $x$에 대한 확률을 구할 수 있다.

$$
p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_{t-1}, \dots, x_1)
$$

### RNN Encoder-Decoder

n번째 말하고 있지만, 이 논문에서 제안하는 모델 구조는 두 개의 RNN을 활용하고 역할은 다음과 같다.

- Encoder RNN : 가변 길이의 시퀀스를 고정 길이의 벡터 표현으로 변경

- Decoder RNN : 고정 길이의 벡터 표현을 다시 가변 길이의 시퀀스로 변경 

따라서 이 모델은 다른 가변 길이 시퀀스를 조건으로 한 가변 길이 시퀀스에 대한 조건부 분포를 학습하게 된다.

$$
p(y_1, ..., y_{T'} | x_1, ..., x_T)
$$

모델의 흐름은 다음과 같다.

1. Encoder가 입력 시퀀스 $x$를 순차적으로 읽고 hidden state를 업데이트한다.

2. 시퀀스를 끝까지 다 읽으면 RNN의 hidden state가 전체 입력 시퀀스에 대한 요약본 $c$가 된다.

3. decoder가 hidden state $h_{\langle t \rangle}$에 대해서 다음 심볼인 $y_t$를 예측하여 output sequence를 생성한다.

여기서 중요한 점은 RNN이 $y_t$와 $h_{\langle t \rangle}$뿐만 아니라 입력 시퀀스의 요약본인 $c$이라는 조건을 받게 된다. 따라서 decoder의 $t$시점 hidden state는 다음과 같이 된다.

$$
h_{\langle t \rangle} = f(h_{\langle t-1 \rangle}, y_{t-1}, c)
$$

다음 심볼에 대한 조건부 확률 역시

$$
P(y_t \mid y_{t-1}, y_{t-2}, \dots, y_1, \mathbf{c})
=
g(\mathbf{h}_{\langle t \rangle}, y_{t-1}, \mathbf{c})
$$

가 된다.

<img src="{{ site.baseurl }}/assets/images/paperReview/RNN_Encdoer_Decoder/RNN_Encoder_Decoder_Figure1.png" alt="RNN Encoder-Decoder Figure 1" style="display: block; margin: 0 auto;">

$figure1$을 보면 두 개의 RNN인 Encoder와 Decoder가 보이는데, 이 둘은 모두 다음 식을 최대화하게 학습이 된다.

$$
\max_{\theta} \frac{1}{N} \sum_{n=1}^{N}
\log p_{\theta}(\mathbf{y}_n \mid \mathbf{x}_n)
$$

모델이 학습이 되고 나면 두 가지 방법으로 사용될 수 있다.

- 입력 시퀀스에 대해서 target sequence를 생성하도록 함

- 입력과 출력 시퀀스 쌍에 대해서 $score$를 계산하도록 사용

### Hidden Unit that Adaptively Remembers and Forgets

Introduction에서 이 모델은 추가적인 hidden unit을 사용하였다고 한다. 그것은 바로 LSTM에 기반해서 연산이 간단하고 구현이 쉬운 hidden unit이다.

<img src="{{ site.baseurl }}/assets/images/paperReview/RNN_Encdoer_Decoder/RNN_Encoder_Decoder_Figure2.png" alt="RNN Encoder-Decoder Figure 2" style="display: block; margin: 0 auto;">

각 hidden unit은 reset gate와 update gate로 나뉜다.

**reset gate**

$$
z_j = \sigma \left( [\mathbf{W}_r \mathbf{x}]_j + [\mathbf{U}_r \mathbf{h}_{\langle t-1 \rangle}]_j \right)
$$

입력 $\mathbf{x}$와 이전 hidden state 값인 $\mathbf{h}_{\langle t-1 \rangle}$에 가중치가 곱해져 시그모이드 함수에 들어가는 형태이다.

**update gate**

$$
z_j = \sigma \left( [\mathbf{W}_z \mathbf{x}]_j + [\mathbf{U}_z \mathbf{h}_{\langle t-1 \rangle}]_j \right)
$$

이때 $h_j$는 다음과 같이 연산이 된다.

$$
h_j^{\langle t \rangle}
=
z_j h_j^{\langle t-1 \rangle}
+
(1 - z_j)\tilde{h}_j^{\langle t \rangle}
$$

$$
\tilde{h}_j^{\langle t \rangle}
=
\phi \left(
[\mathbf{W}\mathbf{x}]_j
+
[\mathbf{U}(\mathbf{r} \odot \mathbf{h}_{\langle t-1 \rangle})]_j
\right)
$$

이 식에서 reset gate(=$\mathbf{r}$)가 0에 가까워지면 hidden state는 이전 hidden state의 값을 잊게 하고, 현재 입력만으로 초기화한다.

즉, update gate는 이전의 hidden state의 값을 얼마나 갖고 갈지를 정하게 된다.

결론적으로 short-term을 포착하도록 학습된 unit은 reset gate가 더 자주 활성화가 될 것이고, long-term을 포착하는 것은 update gate가 자주 활성화가 될 것이다.

## Statistical Machine Translation

SMT에서는 system의 목표는 주어진 문장 $e$에 대해서 다음을 최대화하는 번역 $f$를 찾는 것이다.

$$
p(f|e) \propto p(e|f)p(f)
$$

이때 $p(e\|f)$는 번역 모델, $p(f)$는 언어 모델이라고 한다.
언어 모델과 번역 모델이 갑자기 왜 등장하는가 싶을 수 있는데, 이들은 기존 SMT 시스템의 구성 요소로 생각하면 된다.

언어 모델은 target이 자연스러운지 판단하는 모델, 번역 모델은 잘 번역이 되었는지를 판단하는 모델 정도로 생각하면 된다.

관행적으로. SMT system은 단순히 저 두 항만 사용하지 않고, 여러 feature들을 합친 log-linear 모델로 $log\ p(f\|e)$ 를 모델링한다. 이때 log-linear는 feature와 그 feature에 대응되는 가중치들로 이루어진다

$$
log \ p(f|e) = \sum^N_{n=1} w_n f_n(f,e) + log \ Z(e)
$$

여기에서 $f_n$은 n번째 feature가 되고, $w_n$은 그에 대응되는 가중치를 의미하게 된다. $Z(e)$는 정규화항이다.

이때 feature가 무엇인가 하면 "target sentence가 자연스러운지가?", "단어 수가 너무 적진 않은가?" 와 같은 번역 후보에 대한 여러 평가 점수라고 생각하면 된다.

여기서 가중치들은 학습 데이터셋에 과적합되는 것을 방지하기 위해 개발 데이터셋에 대한 $BLUE$ 점수를 최대화하도록 최적화가 된다

**Phrase-based SMT**

Phrase-based SMT는 번역 모델 $log\ p(e\|f)$를 phrase pair 단위로 쪼갠 뒤 각 확률을 구하여 계산한다. 이렇게 계산된 확률은 log-linear model에 feature로 활용된다.

예시를 들어보자.

$$
source\ sentence = I\ will\ go\ home\\
target\ sentence = Je\ vais\ rentrer\ chez\ moi
$$

이를 phrase 단위로 짝지어보면 다음과 같이 된다.

$$
(I, Je)\\
(Will go, vais rentrer)\\
(home, chez moi)
$$

따라서 번역 모델인 $p(e\|f)$는 아래와 같이 계산할 수 있다. 계산된 것은 마찬가지로 하나의 feature로 사용된다.

$$
p(e∣f)≈p(I∣Je)⋅p(will go∣vais rentrer)⋅p(home∣chez moi) \\

log\ p(e∣f)≈log\ p(I∣Je)+log\ p(will go∣vais rentrer)+log\ p(home∣chez moi)
$$

SMT 분야에서 신경망 구조가 널리 활용되기 시작했고, 최근에는 신경망 모델을 학습시켜서 단순히 target 문장이 자연스러운지만 판단하는 것이 아니라 source 문장도 함께 사용하여 적적한 번역인지 점수화하는 방식이 제안되기 시작했다고 한다.

관련 방법들이 3.2절에 제시가 되니 궁금한 사람만 읽어보면 되겠다. 

### Scoring Phrase Pairs with RNN Encoder-Decoder

다시 이 논문에서 제안하는 바로 돌아오자. 이 논문에서는 RNN Encoder-Decoder를 학습하여 phrase pair table의 값들을 점수화하여 추가적인 피처를 활용하는 것이다.

이때 RNN Encoder-Decoder를 학습할 때, 원래 말뭉치에서 phrase pair의 빈도를 고려하지 않았다고 한다.

여기서 원래 말뭉치 (= original corpora)는 무엇인가?

original corpora와 phrase pair table은 데이터 처리 단계가 다르다. original corpura는 원래 말뭉치, 즉 "영어 문장과 프랑스어 문장 쌍"을 의미한다. phrase pair table은 그 original corpora에서 전처리기를 거쳐 구문쌍을 추출해 만든 사전이다.

다시 돌아와서, 원래 말뭉치의 빈도를 고려하지 않은 이유는 다음과 같다.

1. 빈도를 고려하여 가중 샘플링 (정규화 빈도 활용)하게 되면 발생하는 연산 비용을 줄일 수 있다.

2. RNN Encoder-Decoder가 phrase pair의 등장횟수에 따른 순위를 단순히 학습하기를 원하지 않았다.

다시 말해 phrase table에서의 이미 존재하는 확률은 원래 말뭉치의 phrase pair의 빈도를 반영하고 있기 때문에 말뭉치의 빈도를 고려하지 않은 것이다.

학습 된 이후에는 RNN Encoder-Decoder가 각 phrase pair에 대해서 점수를 구하고 추가적인 feature로 활용한다.

## Experiments

제안된 모델을 평가하기 위해 WMT'14 (workshop of machine translation 2014)의 영어/프랑스어 번역 과제를 진행하였다고 한다.

### Data and Baseline

데이터와 베이스라인에 대해서 간단하게만 설명하려고 한다. 추가적으로 궁금한 사람은 4.1절을 읽어보면 되겠다.

우선 영어/프랑스어 SMT 시스템을 만들기에 충분한 양의 데이터가 있다고 한다. 하지만 통계학적 모델들에게 모든 데이터를 합쳐서 학습하는 것이 반드시 최적의 성능으로 이끄는 것이 아니며 다루기 함들다는 것을 지적한다.

따라서 주어진 task를 위해 관련있는 subset에만 집중하는 것을 제안한다. 최종적으로

- 언어 모델 : 2G 단어 중에 418M 단어

- RNN Encdoer-Decoder : 850M 단어 중에 348M 단어

를 선택했다고 한다. 

참고로 여기서 등장하는 언어 모델은 앞서 나왔던 그 언어 모델을 말한다. 

개발용 데이터셋과 검증용 데이터 셋은 각각 다음을 활용했다.

- 개발용 : newstest2012 and 2013

- 검증용 : newstest2014

여기서 말하는 개발용이란, data selection과 SMT feature weight를 튜닝하는 데이터들을 말한다.

RNN Encoder-Decoder를 학습할 때, 모델의 vocabulary 크기를 영어, 프랑스어 각각 15,000으로 제한했다고 한다. 앞에서는 348M으로 학습했다면서 이게 무슨 말인가 싶을 수 있다. 앞에서 말한 348M은 서로 다른 단어 3억 4800만 개가 아니고, 등장한 단어 토큰 수가 348M이라는 말이다.

예를 들어보자

(the, the, the, end, and, dog, dog, cat)

이것의 토큰 수는 8개이지만, 단어 수는 4개이다.

이것처럼, RNN Encoder-Decoder의 vocabulary를 빈도 기준으로 상위 15,000개로 제한했을 때, 전체의 93%를 커버할 수 있다는 것이다.

위의 예시를 생각해봤을 때, vocabulary를 2으로 제한한다고 하자.

Top-2는 (the, dog)가 될 것이고, 전체의 75%이다.

나머지 제외된 단어들은 모두 special token인 [UNK]로 매핑했다고 한다.

### RNN Encoder-Decoder

RNN Encdoer-Decoder의 구조는 encoder와 decoder마다 1000개의 hidden unit + gate으로 구성이 된다.

앞서 vocabulary를 15,000로 제한했다고 말했다. 이때 1000개의 hidden unit을 사용하게 되면 입력 단어 $x\< t \>$와 hidden state의 가중치 행렬이 매우 커지게 된다.

따라서 논문에서는 기존의 큰 입력 행렬을 직접 학습하지 않고, 입력 단어를 100차원 임베딩으로 변환하는 행렬과 이 임베딩을 다시 1000차원 hidden state 입력으로 변환하는 중간차원이 100인 행렬의 곱으로 근사하였다..


hidden state에서 output으로 연결은 심층 신경망을 사용하였고 파라미터 초기화할 때 Recurrent Weight와 나머지 초기화 방식이 약각 다른데, 관한 내용은 직접 논문을 읽어보는 것을 추천한다.

간단하게 말하면, 바로 분포에서 샘플링하냐 혹은 얻은 행렬을 SVD를 한 번 거쳐서 활용하는가에 대한 차이다. 

모델을 업데이트할 때는 348M으로 얻은 phrase table에서 phrase pair 랜덤 64래를 선택해서 업데이트한다고 한다.

### Neural Language Model

이 부분은 제안된 모델을 통한 scoring의 효율성을 확인하기 위해 Language Model (CSLM)과 비교한 내용이다.

실험은 CSLM을 사용하는 SMT system과 RNN Encoder-Decoder를 활용하는 SMT system으로 나뉜다.

CSLM의 구조를 간략하게만 정리해보자.

- CSLM은 7-gram이다

- 각 입력 단어는 512차원으로 임베딩이 되고, 총 3072 차원 벡터로 합쳐진다.

    3072가 어떻게 나왔냐 하면, 7-gram이므로 target 1개를 예측하기 위해 6개의 단어를 보기 때문에 512 * 6이 되기 때문이다.

- 두 개의 ReLU 계열 layer를 통과한다.

- output layer는 간단한 소프트맥스 layer이다.

CSLM은 RNN Encdoer-Decoder와 다르게, 번역 후보가 생성 중에 번역 후보에 대해서 점수를 측정하여 탐색 방향을 조정하는 방식이다.

최종적으로 후보를 모두 생성하고 난 뒤에 rescore하는 n-best list rescroing보다 BLEU 점수가 높았다고 한다.

<img src="{{ site.baseurl }}/assets/images/paperReview/RNN_Encdoer_Decoder/RNN_Encoder_Decoder_Table1.png" alt="RNN Encoder-Decoder Figure 1" style="display: block; margin: 0 auto;">

$Table1$을 통해서 신경망 모델을 통해서 feature을 추가하는 것이 baseline의 성능 향상에 개선된다는 것을 알 수 있다.

나아가 CSLM과 RNN Encoder-Decoder를 함께 사용하였을 때가 성능이 가장 좋았던 것을 통해서, 서로 연관이 되어 있지 않고 개별 단독을 사용했을 때보다 성능 향상에 서로 도움이 된다는 것을 알 수 있다.

또한 앞서서 OOV들은 [UNK] 토큰으로 대체한다고 하였는데 이때문에 [UNK]가 과대평가될 문제를 지적한다.

OOV 단어들이 모두 [UNK] 토큰이 되기 때문에 [UNK] 토큰의 확률은 다음과 같다.

$$
\begin{aligned}
p(x_t = \text{[UNK]} \mid x_{<t})
&= p(x_t \notin \mathrm{SL} \mid x_{<t}) \\
&= \sum_{x_t^j \notin \mathrm{SL}} p(x_t^j \mid x_{<t})
\ge p(x_t^i \mid x_{<t})
\end{aligned}
$$

이 말인 즉, [UNK] 토큰은 OOV들이 등장할 확률의 합이기 때문에 항상 OOV 개별의 확률보다 크거나 같다는 것을 문제로 지적한다.

이를 해결하기 위해서 unknown word의 단우 수를 추가적인 feature로 활용하는 방법을 제안하였으나, 성능 향상에서 크게 이득을 보지 못했다고 한다.

### Qualitative Analysis

이 부분은 성능 향상이 어디에서 생기는지를 이해하기 위해서 기존 존재하는 번역 모델과 비교하는 내용이다.

기존 번역 모델은 통계 기반이기 때문에, 빈번하게 나오는지에 따라 안정성이 떨어진다고 주장한다. 반면 RNN Encoder-Decoder는 앞서 말했듯, 빈도에 대한 정보를 없이 언어적 규칙성을 학습하도록 하였기 때문에 장점이 있다고 한다.

이 실험에서는 source phrase가 $long$ and $frequent$한 것과 $long$ and $rare$한 것을 대상으로 진행하였다.

<img src="{{ site.baseurl }}/assets/images/paperReview/RNN_Encdoer_Decoder/RNN_Encoder_Decoder_Table2.png" alt="RNN Encoder-Decoder Figure 1" style="display: block; margin: 0 auto;">

$Table2$는 source phrase에 대한 top-3 target phrase를 보인다.

RNN Encoder-Decoder의 대부분의 선택은 실제 또는 문자 그대로 번역에 가깝다는 것을 알 수 있고, 짧은 phrase를 선호한다는 것도 알 수 있다.

여기서 중요하게 봐야하는 것은, 많은 phrase pair가 번역 모델과 RNN Encoder-Decoder가 유사한 점수를 주었지만 점수가 다른 것이 꽤 많다는 것에 집중해야 한다.

<img src="{{ site.baseurl }}/assets/images/paperReview/RNN_Encdoer_Decoder/RNN_Encoder_Decoder_Figure3.png" alt="RNN Encoder-Decoder Figure 1" style="display: block; margin: 0 auto;">

이를 통해서 통계 기반으로 접근할 경우, 단어가 드문 경우 1번 등장했는데 우연히 일치할 수 있기 때문에 불안정하지만 RNN Encoder-Decoder는 빈도를 고려하지 않기 때문에 다른 결정을 내린다(= 다른 정보를 본다)는 것을 알 수 있다.

<img src="{{ site.baseurl }}/assets/images/paperReview/RNN_Encdoer_Decoder/RNN_Encoder_Decoder_Table3.png" alt="RNN Encoder-Decoder Figure 1" style="display: block; margin: 0 auto;">

$Table3$는 source phrase에 대해서 RNN Encoder-Decoder가 생성한 phrase이다.

결과적으로 생성된 phrase는 well-fromed된 phrase이며, 완전히 phrase table과 일치하지 않는다는 점에서 phrase table을 완전히 또는 일부 교체하는 것이 가능하다는 점을 보여준다.

### Word and Phrase Representations

저자는 RNN Encoder-Decoder가 기계 번역 task에만 국한되지 않는다는 것을 주장하며 모델의 특징을 설명한다.

신경망을 활용하는 연속 공간 언어 모델이 의미적으로 유의미한 임베딩을 갖는다는 것이 알려져있다는 점을 언급한다.

RNN Encoder-Deocder 또한 단어의 시퀀스를 연속적인 벡터 공간으로 투영하고 매핑하기 때문에 유의미한 임베딩을 갖을 것이라고 주장한다.

<img src="{{ site.baseurl }}/assets/images/paperReview/RNN_Encdoer_Decoder/RNN_Encoder_Decoder_Figure4.png" alt="RNN Encoder-Decoder Figure 1" style="display: block; margin: 0 auto;">

$Figure4$는 RNN Encoder-Decoder에 의해 학습된 단어 임베딩 행렬을 이용한 단어 2차원 임베딩을 보여준다. 

이를 통해서 의미적으로 유사한 단어들이 가깝게 군집을 이루는 것을 볼 수 있다 (오른쪽 Figure)

<img src="{{ site.baseurl }}/assets/images/paperReview/RNN_Encdoer_Decoder/RNN_Encoder_Decoder_Figure5.png" alt="RNN Encoder-Decoder Figure 1" style="display: block; margin: 0 auto;">

또한 RNN Encoder-Decoder는 phrase의 연속적인 공간 표현도 생성한다. 따라서 마찬가지로 phrase의 의미론적 표현도 잘 포착할 수 있을 것이라고 기대할 수 있다.

실제로 $Figure5$를 통해서 phrase의 의미론적, 구문론적 구조를 모두 잘 포착했음을 알 수 있다.

예를 들어서 왼쪽 아래에는 시간과 관련되고 구문적으로 비슷한 것들이 군집을 이루었으며, 오른쪽 아래에는 의미적으로 유사한 (국가나 지역)이 군집을 이룬 것을 볼 수 있다. 마지막으로 오른쪽 위는 구문적으로 비슷한 phrase가 군집을 이루고 있었다.

최종적으로 RNN Encoder-Decoder는 구조상 written language에만 제한되지 않기 때문에 다른 분야에 적용하는 것이 중요하다고 강조한다.