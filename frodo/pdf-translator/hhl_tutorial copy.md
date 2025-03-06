# HHL 및 Qiskit 구현을 사용한 선형 시스템 방정식 풀이

본 튜토리얼에서는 HHL 알고리즘을 소개하고, 회로를 도출하고, Qiskit을 사용하여 구현합니다. HHL을 시뮬레이터와 5 큐비트 장치에서 실행하는 방법을 보여줍니다.

## 목차
1. [소개](#introduction)
2. [HHL 알고리즘](#hhlalg)
    1. [수학적 배경](#mathbackground)
    2. [HHL 설명](#hhldescription)
    3. [HHL 내의 양자 위상 추정 (QPE)](#qpe)
    4. [비정확한 QPE](#qpe2)
4. [Qiskit 구현](#implementation)
    1. [시뮬레이터에서 HHL 실행: 일반적인 방법](#implementationsim)
    2. [실제 양자 장치에서 HHL 실행: 최적화된 예제](#implementationdev)
5. [문제](#problems)
6. [참고 문헌](#references)
## 1. 소개 <a id='introduction'></a>

선형 시스템 방정식은 광범위한 분야에서 많은 실제 응용 분야에서 볼 수 있습니다. 예로는 편미분 방정식의 해, 금융 모델의 보정, 유체 시뮬레이션 또는 수치장 계산이 있습니다. 이 문제는 행렬 $A\in\mathbb{C}^{N\times N}$와 벡터 $\vec{b}\in\mathbb{C}^{N}$가 주어졌을 때, $A\vec{x}=\vec{b}$를 만족하는 $\vec{x}\in\mathbb{C}^{N}$를 찾는 것으로 정의할 수 있습니다.

예를 들어, $N=2$인 경우,

$$A = \begin{pmatrix}1 & -1/3\\-1/3 & 1 \end{pmatrix},\quad \vec{x}=\begin{pmatrix} x_{1}\\ x_{2}\end{pmatrix}\quad \text{and} \quad \vec{b}=\begin{pmatrix}1 \\ 0\end{pmatrix}$$

그러면 이 문제는 다음과 같이 `{latex} x_{1}, x_{2}\in\mathbb{C}`를 찾는 것으로 쓸 수도 있습니다.
$$\begin{cases}x_{1} - \frac{x_{2}}{3} = 1 \\ -\frac{x_{1}}{3} + x_{2} = 0\end{cases} $$

선형 방정식 시스템은 행 또는 열당 최대 $s$개의 0이 아닌 항목을 갖는 경우 $s$-희소(sparse)하다고 합니다. 크기가 $N$인 $s$-희소 시스템을 고전적인 컴퓨터로 푸는 데에는 켤레 기울기법을 사용하여 $\mathcal{ O }(Ns\kappa\log(1/\epsilon))$의 실행 시간이 필요합니다 <sup>[1](#conjgrad)</sup>. 여기서 $\kappa$는 시스템의 조건수를 나타내고 $\epsilon$은 근사의 정확도를 나타냅니다.

HHL 알고리즘은 $\mathcal{ O }(\log(N)s^{2}\kappa^{2}/\epsilon)$<sup>[2](#hhl)</sup>의 실행 시간 복잡도로 해의 함수를 추정합니다. 행렬 $A$는 에르미트 행렬이어야 하며, 데이터 로딩, 해밀토니안 시뮬레이션 및 해의 함수 계산을 위한 효율적인 오라클이 있다고 가정합니다. 이는 시스템 크기에 대한 지수적인 속도 향상이지만, HHL은 해 벡터의 함수만 근사할 수 있고 고전적인 알고리즘은 전체 해를 반환한다는 단점이 있습니다.

## 2. HHL 알고리즘<a id='hhlalg'></a>

### A. 수학적 배경<a id='mathbackground'></a>
양자 컴퓨터로 선형 방정식 시스템을 풀기 위한 첫 번째 단계는 문제를 양자 언어로 인코딩하는 것입니다. 시스템을 재조정하여 $\vec{b}$와 $\vec{x}$가 정규화되었다고 가정하고 이를 각각 양자 상태 $|b\rangle$와 $|x\rangle$에 매핑할 수 있습니다. 일반적으로 사용되는 매핑은 $\vec{b}$ (또는 $\vec{x}$)의 $i$번째 구성 요소가 양자 상태 $|b\rangle$ (또는 $|x\rangle$)의 $i$번째 기저 상태의 진폭에 해당하는 방식입니다. 이제부터는 재조정된 문제에 집중하겠습니다.

$$ A|x\rangle=|b\rangle$$

$A$는 에르미트 행렬이므로 다음과 같은 스펙트럼 분해를 가집니다.
$$
A=\sum_{j=0}^{N-1}\lambda_{j}|u_{j}\rangle\langle u_{j}|,\quad \lambda_{j}\in\mathbb{ R }
$$
여기서 $|u_{j}\rangle$는 $A$의 $j$번째 고유벡터이고 $\lambda_{j}$는 해당 고유값입니다. 그러면,
$$
A^{-1}=\sum_{j=0}^{N-1}\lambda_{j}^{-1}|u_{j}\rangle\langle u_{j}|
$$
이고 시스템의 우변은 $A$의 고유기저로 다음과 같이 쓸 수 있습니다.
$$
|b\rangle=\sum_{j=0}^{N-1}b_{j}|u_{j}\rangle,\quad b_{j}\in\mathbb{ C }
$$
HHL의 목표는 판독 레지스터를 다음 상태로 만들어 알고리즘을 종료하는 것임을 명심하는 것이 유용합니다.
$$
|x\rangle=A^{-1}|b\rangle=\sum_{j=0}^{N-1}\lambda_{j}^{-1}b_{j}|u_{j}\rangle
$$
여기서 양자 상태에 대해 이야기하고 있으므로 이미 암묵적인 정규화 상수가 있습니다.

### B. HHL 알고리즘 설명 <a id='hhldescription'></a>

이 알고리즘은 세 개의 양자 레지스터를 사용하며, 알고리즘 시작 시 모두 $|0\rangle$로 설정됩니다. 하나의 레지스터는 아래 첨자 $n_{l}$로 표시되며, $A$의 고유값의 이진 표현을 저장하는 데 사용됩니다. 두 번째 레지스터는 $n_{b}$로 표시되며, 벡터 해를 포함하고, 앞으로 $N=2^{n_{b}}$입니다. 계산의 중간 단계에 사용되는 보조 큐비트에 대한 추가 레지스터가 있습니다. 다음 설명에서는 각 계산 시작 시 $|0\rangle$이고 각 개별 연산이 끝날 때 다시 $|0\rangle$로 복원되므로 보조 레지스터는 무시할 수 있습니다.

다음은 해당 회로의 개략적인 그림과 함께 HHL 알고리즘의 개요입니다. 단순화를 위해 모든 계산은 이후 설명에서 정확하다고 가정하며, 정확하지 않은 경우에 대한 자세한 설명은 섹션 [2.D.](#qpe2)에 제공됩니다.


<!-- vale QiskitTextbook.Spelling = NO -->

1. 데이터 $|b\rangle\in\mathbb{ C }^{N}$를 로드합니다. 즉, 변환을 수행합니다.
    $$ |0\rangle _{n_{b}} \mapsto |b\rangle _{n_{b}} $$
2. 다음을 사용하여 양자 위상 추정 (QPE)을 적용합니다.
    $$ U = e ^ { i A t } := \sum _{j=0}^{N-1}e ^ { i \lambda _ { j } t } |u_{j}\rangle\langle u_{j}| $$
	$A$의 고유기저로 표현된 레지스터의 양자 상태는 다음과 같습니다.
    $$ \sum_{j=0}^{N-1} b _ { j } |\lambda _ {j }\rangle_{n_{l}} |u_{j}\rangle_{n_{b}} $$
    여기서 `{latex} |\lambda _ {j }\rangle_{n_{l}}`는 $\lambda _ {j }$의 $n_{l}$비트 이진 표현입니다.
    
3. 보조 큐비트를 추가하고 $|\lambda_{ j }\rangle$에 따라 조건부 회전을 적용합니다.
    $$ \sum_{j=0}^{N-1} b _ { j } |\lambda _ { j }\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} \left( \sqrt { 1 - \frac { C^{2}  } { \lambda _ { j } ^ { 2 } } } |0\rangle + \frac { C } { \lambda _ { j } } |1\rangle \right) $$
	여기서 $C$는 정규화 상수이며, 위에서 현재 형태로 표현된 것처럼 가장 작은 고유값 $\lambda_{min}$보다 작아야 합니다. 즉, $|C| < \lambda_{min}$입니다.
    
4. QPE$^{\dagger}$를 적용합니다. QPE의 가능한 오류를 무시하면 다음이 됩니다.
    $$ \sum_{j=0}^{N-1} b _ { j } |0\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} \left( \sqrt { 1 - \frac {C^{2}  } { \lambda _ { j } ^ { 2 } } } |0\rangle + \frac { C } { \lambda _ { j } } |1\rangle \right) $$
    
5. 계산 기저에서 보조 큐비트를 측정합니다. 결과가 $1$이면 레지스터는 사후 측정 상태에 있습니다.
    $$ \left( \sqrt { \frac { 1 } { \sum_{j=0}^{N-1} \left| b _ { j } \right| ^ { 2 } / \left| \lambda _ { j } \right| ^ { 2 } } } \right) \sum _{j=0}^{N-1} \frac{b _ { j }}{\lambda _ { j }} |0\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} $$
	이는 정규화 인수를 제외하고 해에 해당합니다.

6. 관측 가능한 $M$을 적용하여 $F(x):=\langle x|M|x\rangle$를 계산합니다.

### C. HHL 내의 양자 위상 추정 (QPE) <a id='qpe'></a>

양자 위상 추정은 3장에서 더 자세히 설명되어 있습니다. 그러나 이 양자 절차가 HHL 알고리즘의 핵심이므로 여기에서 정의를 상기합니다. 대략적으로 말하면, 이는 고유벡터 $|\psi\rangle_{m}$ 및 고유값 $e^{2\pi i\theta}$를 갖는 유니타리 $U$가 주어졌을 때 $\theta$를 찾는 양자 알고리즘입니다. 이를 다음과 같이 공식적으로 정의할 수 있습니다.

**정의:** $U\in\mathbb{ C }^{2^{m}\times 2^{m}}$를 유니타리라고 하고 $|\psi\rangle_{m}\in\mathbb{ C }^{2^{m}}$를 각각 고유값 $e^{2\pi i\theta}$를 갖는 고유벡터 중 하나라고 합시다. **양자 위상 추정** 알고리즘 (약칭 **QPE**)은 $U$에 대한 유니타리 게이트와 상태 `{latex} |0\rangle_{n}|\psi\rangle_{m}`을 입력으로 사용하고 상태 `{latex} |\tilde{\theta}\rangle_{n}|\psi\rangle_{m}`을 반환합니다. 여기서 $\tilde{\theta}$는 $2^{n}\theta$에 대한 이진 근사값을 나타내고 $n$ 아래 첨자는 $n$자리로 잘렸음을 나타냅니다.
$$
\operatorname { QPE } ( U , |0\rangle_{n}|\psi\rangle_{m} ) = |\tilde{\theta}\rangle_{n}|\psi\rangle_{m}
$$

HHL에서는 $U = e ^ { i A t }$인 QPE를 사용합니다. 여기서 $A$는 풀고자 하는 시스템과 관련된 행렬입니다. 이 경우,
$$
e ^ { i A t } = \sum_{j=0}^{N-1}e^{i\lambda_{j}t}|u_{j}\rangle\langle u_{j}|
$$
그러면 고유값 $e ^ { i \lambda _ { j } t }$를 갖는 고유벡터 `{latex} |u_{j}\rangle_{n_{b}}`에 대해 QPE는 `{latex} |\tilde{\lambda }_ { j }\rangle_{n_{l}}|u_{j}\rangle_{n_{b}}`를 출력합니다. 여기서 $\tilde{\lambda }_ { j }$는 $2^{n_l}\frac{\lambda_ { j }t}{2\pi}$에 대한 $n_{l}$비트 이진 근사값을 나타냅니다. 따라서 각 $\lambda_{j}$를 $n_{l}$비트로 정확하게 나타낼 수 있다면,
$$
\operatorname { QPE } ( e ^ { i A t } , \sum_{j=0}^{N-1}b_{j}|0\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} ) = \sum_{j=0}^{N-1}b_{j}|\lambda_{j}\rangle_{n_{l}}|u_{j}\rangle_{n_{b}}
$$

### D. 비정확한 QPE <a id='qpe2'></a>

실제로 초기 상태에 QPE를 적용한 후 레지스터의 양자 상태는 다음과 같습니다.

$$ \sum _ { j=0 }^{N-1} b _ { j } \left( \sum _ { l = 0 } ^ { 2 ^ { n_{l} } - 1 } \alpha _ { l | j } |l\rangle_{n_{l}} \right)|u_{j}\rangle_{n_{b}} $$
여기서

$$ \alpha _ { l | j } = \frac { 1 } { 2 ^ { n_{l} } } \sum _ { k = 0 } ^ { 2^{n_{l}}- 1 } \left( e ^ { 2 \pi i \left( \frac { \lambda _ { j } t } { 2 \pi } - \frac { l } { 2 ^ { n_{l} } } \right) } \right) ^ { k } $$

$\tilde{\lambda_{j}}$를 $\lambda_{j}$에 대한 최상의 $n_{l}$비트 근사값이라고 합시다 (단, $1\leq j\leq N$). 그러면 `{latex} |l + \tilde { \lambda } _ { j } \rangle_{n_{l}}`의 진폭을 나타내도록 $\alpha _ { l | j }$가 되도록 $n_{l}$-레지스터의 레이블을 다시 지정할 수 있습니다. 이제,

$$\alpha _ { l | j } : = \frac { 1 } { 2 ^ { n_{l}} } \sum _ { k = 0 } ^ { 2 ^ { n_{l} } - 1 } \left( e ^ { 2 \pi i \left( \frac { \lambda _ { j } t } { 2 \pi } - \frac { l + \tilde { \lambda } _ { j } } { 2 ^ { n_{l} } } \right) } \right) ^ { k }$$

각 $\frac { \lambda _ { j } t } { 2 \pi }$를 $n_{l}$ 이진 비트로 정확하게 나타낼 수 있다면 `{latex} \frac { \lambda _ { j } t } { 2 \pi }=\frac { \tilde { \lambda } _ { j } } { 2 ^ { n_{l} } }` $\forall j$입니다. 따라서 이 경우 $\forall j$, $1\leq j \leq N$에 대해 $\alpha _ { 0 | j } = 1$이고 $\alpha _ { l | j } = 0 \quad \forall l \neq 0$입니다. 이 경우에만 QPE 후 레지스터의 상태를 다음과 같이 쓸 수 있습니다.

$$ \sum_{j=0}^{N-1} b _ { j } |\lambda _ {j }\rangle_{n_{l}} |u_{j}\rangle_{n_{b}}$$

그렇지 않으면 `{latex} \frac { \lambda _ { j } t } { 2 \pi } \approx \frac { l + \tilde { \lambda } _ { j } } { 2 ^ { n_{l} } }`인 경우에만 $|\alpha _ { l | j }|$가 큽니다. 레지스터의 상태는 다음과 같습니다.

$$ \sum _ { j=0 }^{N-1}  \sum _ { l = 0 } ^ { 2 ^ { n_{l} } - 1 } \alpha _ { l | j } b _ { j }|l\rangle_{n_{l}} |u_{j}\rangle_{n_{b}} $$

## 3. 예: 4-큐비트 HHL<a id='example1'></a>

소개에서 사용된 작은 예를 통해 알고리즘을 설명해 보겠습니다. 즉,
$$A = \begin{pmatrix}1 & -1/3\\-1/3 & 1 \end{pmatrix}\quad \text{and} \quad |b\rangle=\begin{pmatrix}1 \\ 0\end{pmatrix}$$

$|b\rangle$ (나중에 해 $|x\rangle$)를 나타내는 데 $n_{b}=1$ 큐비트, 고유값의 이진 표현을 저장하는 데 $n_{l}=2$ 큐비트, 조건부 회전 (따라서 알고리즘)이 성공했는지 여부를 저장하는 데 $1$개의 보조 큐비트를 사용합니다.

알고리즘을 설명하기 위해 약간 속임수를 써서 $A$의 고유값을 계산하여 $t$를 선택하여 $n_{l}$-레지스터에서 재조정된 고유값의 정확한 이진 표현을 얻을 수 있습니다. 그러나 HHL 알고리즘 구현에는 고유값에 대한 사전 지식이 필요하지 않다는 점을 명심하십시오. 즉, 짧은 계산을 통해 다음을 얻을 수 있습니다.
$$\lambda_{1} = 2/3\quad\text{and}\quad\lambda_{2}=4/3$$

이전 섹션에서 QPE는 $\frac{\lambda_ { j }t}{2\pi}$에 대한 $n_{l}$비트 (이 경우 $2$비트) 이진 근사값을 출력한다는 것을 상기하십시오. 따라서 다음을 설정하면
$$t=2\pi\cdot \frac{3}{8}$$
QPE는 다음의 $2$비트 이진 근사값을 제공합니다.
$$\frac{\lambda_ { 1 }t}{2\pi} = 1/4\quad\text{and}\quad\frac{\lambda_ { 2 }t}{2\pi}=1/2$$
이는 각각
$$|01\rangle_{n_{l}}\quad\text{and}\quad|10\rangle_{n_{l}}$$

고유벡터는 각각 다음과 같습니다.
$$|u_{1}\rangle=\frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ -1\end{pmatrix}\quad\text{and}\quad|u_{2}\rangle=\frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ 1\end{pmatrix}$$

다시 말하지만, HHL 구현을 위해 고유벡터를 계산할 필요는 없습니다. 실제로 $N$차원의 일반적인 에르미트 행렬 $A$는 최대 $N$개의 서로 다른 고유값을 가질 수 있으므로 이를 계산하는 데 $\mathcal{O}(N)$ 시간이 걸리고 양자 이점이 손실됩니다.

그러면 $A$의 고유기저에서 $|b\rangle$를 다음과 같이 쓸 수 있습니다.
$$|b\rangle _{n_{b}}=\sum_{j=1}^{2}\frac{1}{\sqrt{2}}|u_{j}\rangle _{n_{b}}$$

이제 HHL 알고리즘의 여러 단계를 거칠 준비가 되었습니다.

1. 이 예에서 상태 준비는 $|b\rangle=|0\rangle$이므로 간단합니다.
2. QPE를 적용하면 다음이 생성됩니다.
$$
\frac{1}{\sqrt{2}}|01\rangle|u_{1}\rangle + \frac{1}{\sqrt{2}}|10\rangle|u_{2}\rangle
$$
3. $C=1/8$인 조건부 회전은 $\frac {1} {4}$의 가장 작은 (재조정된) 고유값보다 작습니다. 여기서 상수 $C$는 $\frac {1} {4}$의 가장 작은 (재조정된) 고유값보다 작지만 보조 큐비트를 측정할 때 상태 $|1>$에 있을 확률이 커지도록 가능한 한 크게 선택해야 합니다.
$$\frac{1}{\sqrt{2}}|01\rangle|u_{1}\rangle\left( \sqrt { 1 - \frac { (1/8)^{2}  } {(1/4)^{2} } } |0\rangle + \frac { 1/8 } { 1/4 } |1\rangle \right) + \frac{1}{\sqrt{2}}|10\rangle|u_{2}\rangle\left( \sqrt { 1 - \frac { (1/8)^{2}  } {(1/2)^{2} } } |0\rangle + \frac { 1/8 } { 1/2 } |1\rangle \right)
$$
$$
=\frac{1}{\sqrt{2}}|01\rangle|u_{1}\rangle\left( \sqrt { 1 - \frac { 1  } {4 } } |0\rangle + \frac { 1 } { 2 } |1\rangle \right) + \frac{1}{\sqrt{2}}|10\rangle|u_{2}\rangle\left( \sqrt { 1 - \frac { 1  } {16 } } |0\rangle + \frac { 1 } { 4 } |1\rangle \right)
$$
4. QPE$^{\dagger}$를 적용한 후 양자 컴퓨터는 다음 상태에 있습니다.
$$
\frac{1}{\sqrt{2}}|00\rangle|u_{1}\rangle\left( \sqrt { 1 - \frac { 1  } {4 } } |0\rangle + \frac { 1 } { 2 } |1\rangle \right) + \frac{1}{\sqrt{2}}|00\rangle|u_{2}\rangle\left( \sqrt { 1 - \frac { 1  } {16 } } |0\rangle + \frac { 1 } { 4 } |1\rangle \right)
$$
5. 보조 큐비트를 측정할 때 결과가 $1$이면 상태는 다음과 같습니다.
$$
\frac{\frac{1}{\sqrt{2}}|00\rangle|u_{1}\rangle\frac { 1 } { 2 } |1\rangle + \frac{1}{\sqrt{2}}|00\rangle|u_{2}\rangle\frac { 1 } { 4 } |1\rangle}{\sqrt{5/32}}
$$
빠른 계산을 통해 다음을 알 수 있습니다.
$$
\frac{\frac{1}{2\sqrt{2}}|u_{1}\rangle+ \frac{1}{4\sqrt{2}}|u_{2}\rangle}{\sqrt{5/32}} = \frac{|x\rangle}{||x||}
$$
6. 추가 게이트를 사용하지 않고도 $|x\rangle$의 노름을 계산할 수 있습니다. 이는 이전 단계에서 보조 큐비트에서 $1$을 측정할 확률입니다.
$$
P(|1\rangle) = \left(\frac{1}{2\sqrt{2}}\right)^{2} + \left(\frac{1}{4\sqrt{2}}\right)^{2} = \frac{5}{32} = ||x||^{2}
$$
## 4. Qiskit 구현<a id='implementation'></a>

이제 예제의 문제를 분석적으로 해결했으므로 양자 시뮬레이터와 실제 하드웨어에서 HHL을 실행하는 방법을 설명하는 데 사용하겠습니다. 다음은 이 [리포지토리](https://github.com/anedumla/quantum_linear_solvers)에서 찾을 수 있고 해당 `Readme` 파일에 설명된 대로 설치할 수 있는 Qiskit 기반 패키지인 `quantum_linear_solvers`를 사용합니다. 양자 시뮬레이터의 경우 `quantum_linear_solvers`는 가장 간단한 예에서 행렬 $A$와 $|b\rangle$만 입력으로 요구하는 HHL 알고리즘의 구현을 이미 제공합니다. NumPy 배열로 일반적인 에르미트 행렬과 임의의 초기 상태를 알고리즘에 제공할 수 있지만 이러한 경우 양자 알고리즘은 지수적 속도 향상을 달성하지 못합니다. 이는 기본 구현이 정확하고 큐비트 수에서 지수적이기 때문입니다. 정확한 임의의 양자 상태를 준비하거나 일반적인 에르미트 행렬 $A$에 대해 정확한 연산 $e^{iAt}$를 수행할 수 있는 큐비트 수에서 다항식 리소스 알고리즘은 없습니다. 특정 문제에 대한 효율적인 구현을 알고 있다면 행렬 및/또는 벡터를 `QuantumCircuit` 객체로 제공할 수 있습니다. 또는 삼중 대각 토플리츠 행렬에 대한 효율적인 구현이 이미 있으며 앞으로 더 많을 수 있습니다.

그러나 현재 양자 컴퓨터는 노이즈가 심하고 작은 회로만 실행할 수 있습니다. 따라서 [4.B.](#implementationdev) 섹션에서는 예제가 속한 문제 클래스에 사용할 수 있는 최적화된 회로를 살펴보고 양자 컴퓨터의 노이즈를 처리하기 위한 기존 절차를 언급합니다.

## A. 시뮬레이터에서 HHL 실행: 일반적인 방법<a id='implementationsim'></a>

이 페이지의 코드를 실행하려면 [선형 솔버 패키지](https://github.com/anedumla/quantum_linear_solvers)를 설치해야 합니다. 다음 명령을 통해 이 작업을 수행할 수 있습니다.

```
pip install git+https://github.com/anedumla/quantum_linear_solvers
```

선형 시스템 문제를 해결하기 위한 모든 알고리즘의 인터페이스는 `LinearSolver`입니다. 해결할 문제는 `solve()` 메서드가 호출될 때만 지정됩니다.
```python
LinearSolver(...).solve(matrix, vector)
```

가장 간단한 구현은 행렬과 벡터를 NumPy 배열로 사용합니다. 아래에서는 솔루션을 검증하기 위해 `NumPyLinearSolver`(고전 알고리즘)도 만듭니다.

```python
import numpy as np
# pylint: disable=line-too-long
from linear_solvers import NumPyLinearSolver, HHL
matrix = np.array([[1, -1/3], [-1/3, 1]])
vector = np.array([1, 0])
naive_hhl_solution = HHL().solve(matrix, vector)
```

고전 솔버의 경우 HHL 내에서 `vector`가 양자 상태로 인코딩된 후 발생하는 재정규화를 고려하기 위해 우변(즉, `vector / np.linalg.norm(vector)`)의 스케일을 조정해야 합니다.

```python
classical_solution = NumPyLinearSolver().solve(matrix,
                                               vector/np.linalg.norm(vector))
```

`linear_solvers` 패키지에는 특정 유형의 행렬에 대한 효율적인 구현을 위한 자리 표시자가 되도록 설계된 `matrices`라는 폴더가 포함되어 있습니다. 현재 작성 시점에서 포함된 유일하게 진정으로 효율적인 구현(즉, 큐비트 수에서 다항식으로 확장되는 복잡성)은 `TridiagonalToeplitz` 클래스입니다. 삼중 대각 토플리츠 대칭 실수 행렬은 다음 형식을 갖습니다.
$$A = \begin{pmatrix}a & b & 0 & 0\\b & a & b & 0 \\ 0 & b & a & b \\ 0 & 0 & b & a \end{pmatrix}, a,b\in\mathbb{R}$$
(이 설정에서는 HHL 알고리즘이 입력 행렬이 에르미트 행렬이라고 가정하므로 비대칭 행렬은 고려하지 않습니다).

예제의 행렬 $A$가 이 형식이므로 `TridiagonalToeplitz(num_qubits, a, b)`의 인스턴스를 만들고 결과를 입력으로 배열을 사용하여 시스템을 해결하는 것과 비교할 수 있습니다.

```python
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
tridi_matrix = TridiagonalToeplitz(1, 1, -1 / 3)
tridi_solution = HHL().solve(tridi_matrix, vector)
```

HHL 알고리즘은 고전적인 알고리즘보다 시스템 크기에서 지수적으로 더 빠르게 솔루션을 찾을 수 있습니다(즉, 다항식 대신 로그 복잡도). 그러나 이 지수적 속도 향상의 대가는 전체 솔루션 벡터를 얻지 못한다는 것입니다.
대신 벡터 $x$를 나타내는 양자 상태를 얻고 이 벡터의 모든 구성 요소를 학습하는 데 차원에서 선형 시간이 걸리므로 양자 알고리즘으로 얻은 속도 향상이 줄어듭니다.

따라서 솔루션에 대한 정보를 학습하기 위해 $x$(소위 관측 가능량)에서 함수만 계산할 수 있습니다.
이는 `solve()`에서 반환된 `LinearSolverResult` 객체에 반영되며 다음 속성이 포함되어 있습니다.
- `state`: 솔루션을 준비하는 회로 또는 벡터로의 솔루션
- `euclidean_norm`: 알고리즘이 계산 방법을 알고 있는 경우 유클리드 노름
- `observable`: 계산된 관측 가능량(목록)
- `circuit_results`: 회로(목록)의 관측 가능량 결과

당분간 `observable`과 `circuit_results`는 무시하고 이전에 얻은 솔루션을 확인해 보겠습니다.

먼저 `classical_solution`은 고전 알고리즘의 결과이므로 `.state`를 호출하면 배열이 반환됩니다.

```python
print('classical state:', classical_solution.state)
```

    classical state: [1.125 0.375]

다른 두 예제는 양자 알고리즘이므로 양자 상태에만 액세스할 수 있습니다. 이는 솔루션 상태를 준비하는 양자 회로를 반환하여 달성됩니다.

```python
print('naive state:')
print(naive_hhl_solution.state)
print('tridiagonal state:')
print(tridi_solution.state)
```

    naive state:
          ┌────────────┐┌──────┐        ┌─────────┐
      q4: ┤ circuit-85 ├┤3     ├────────┤3        ├
          └────────────┘│      │┌──────┐│         │
    q5_0: ──────────────┤0     ├┤2     ├┤0        ├
                        │  QPE ││      ││  QPE_dg │
    q5_1: ──────────────┤1     ├┤1     ├┤1        ├
                        │      ││  1/x ││         │
    q5_2: ──────────────┤2     ├┤0     ├┤2        ├
                        └──────┘│      │└─────────┘
      q6: ──────────────────────┤3     ├───────────
                                └──────┘           
    tridiagonal state:
           ┌─────────────┐┌──────┐        ┌─────────┐
      q26: ┤ circuit-298 ├┤3     ├────────┤3        ├
           └─────────────┘│      │┌──────┐│         │
    q27_0: ───────────────┤0     ├┤2     ├┤0        ├
                          │  QPE ││      ││  QPE_dg │
    q27_1: ───────────────┤1     ├┤1     ├┤1        ├
                          │      ││  1/x ││         │
    q27_2: ───────────────┤2     ├┤0     ├┤2        ├
                          └──────┘│      │└─────────┘
      q28: ───────────────────────┤3     ├───────────
                                  └──────┘           

벡터 `{latex} \mathbf{x}=(x_1,\dots,x_N)`에 대한 유클리드 노름은 $||\mathbf{x}||=\sqrt{\sum_{i=1}^N x_i^2}$로 정의됩니다. 따라서 섹션 B의 5단계에서 보조 큐비트에서 $1$을 측정할 확률은 $\mathbf{x}$의 제곱 노름입니다. 이는 HHL 알고리즘이 항상 솔루션의 유클리드 노름을 계산할 수 있음을 의미하며 결과의 정확도를 비교할 수 있습니다.

```python
print('classical Euclidean norm:', classical_solution.euclidean_norm)
print('naive Euclidean norm:', naive_hhl_solution.euclidean_norm)
print('tridiagonal Euclidean norm:', tridi_solution.euclidean_norm)
```

    classical Euclidean norm: 1.1858541225631423
    naive Euclidean norm: 1.185854122563138
    tridiagonal Euclidean norm: 1.1858541225631365

솔루션 벡터를 구성 요소별로 비교하는 것은 더 까다롭고 양자 알고리즘에서 전체 솔루션 벡터를 얻을 수 없다는 아이디어를 다시 반영합니다. 그러나 교육 목적으로 얻은 다양한 솔루션 벡터가 벡터 구성 요소 수준에서도 좋은 근사값인지 확인할 수 있습니다.

이를 위해 먼저 `quantum_info` 패키지에서 `Statevector`를 사용하고 올바른 벡터 구성 요소, 즉 보조 큐비트(회로에서 맨 아래)가 $1$이고 작업 큐비트(회로에서 중간 두 개)가 $0$인 구성 요소를 추출해야 합니다. 따라서 솔루션 벡터의 첫 번째 및 두 번째 구성 요소에 해당하는 상태 `10000` 및 `10001`에 관심이 있습니다.

```python
from qiskit.quantum_info import Statevector

naive_sv = Statevector(naive_hhl_solution.state).data
tridi_sv = Statevector(tridi_solution.state).data

# Extract vector components; 10000(bin) == 16 & 10001(bin) == 17
naive_full_vector = np.array([naive_sv[16], naive_sv[17]])
tridi_full_vector = np.array([tridi_sv[16], tridi_sv[17]])

print('naive raw solution vector:', naive_full_vector)
print('tridi raw solution vector:', tridi_full_vector)
```

    naive raw solution vector: [0.75+3.52055626e-16j 0.25+1.00756137e-16j]
    tridi raw solution vector: [0.75-9.01576529e-17j 0.25+3.74736911e-16j]

언뜻보기에 구성 요소가 실수가 아닌 복소수이기 때문에 이것이 잘못된 것처럼 보일 수 있습니다. 그러나 허수 부분이 매우 작고 컴퓨터 정확도로 인해 발생할 가능성이 높으며 이 경우 무시할 수 있습니다(배열의 `.real` 속성을 사용하여 실수 부분을 가져옵니다).

다음으로 회로의 다른 부분에서 오는 상수를 억제하기 위해 벡터를 각각의 노름으로 나눕니다. 그런 다음 이러한 정규화된 벡터에 위에 계산된 각 유클리드 노름을 곱하여 전체 솔루션 벡터를 복구할 수 있습니다.

```python
def get_solution_vector(solution):
    """Extracts and normalizes simulated state vector
    from LinearSolverResult."""
    solution_vector = Statevector(solution.state).data[16:18].real
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)

print('full naive solution vector:', get_solution_vector(naive_hhl_solution))
print('full tridi solution vector:', get_solution_vector(tridi_solution))
print('classical state:', classical_solution.state)
```

    full naive solution vector: [1.125 0.375]
    full tridi solution vector: [1.125 0.375]
    classical state: [1.125 0.375]

사용된 모든 기본 방법이 정확하기 때문에 `naive_hhl_solution`이 정확하다는 것은 놀라운 일이 아닙니다. 그러나 `tridi_solution`은 $2\times 2$ 시스템 크기 경우에만 정확합니다. 더 큰 행렬의 경우 아래의 약간 더 큰 예에서와 같이 근사값이 됩니다.
```python
from scipy.sparse import diags

NUM_QUBITS = 2
MATRIX_SIZE = 2 ** NUM_QUBITS
# 삼중 대각 Toeplitz 대칭 행렬의 항목
# pylint: disable=invalid-name
a = 1
b = -1/3

matrix = diags([b, a, b],
               [-1, 0, 1],
               shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()

vector = np.array([1] + [0]*(MATRIX_SIZE - 1))
# 알고리즘 실행
classical_solution = NumPyLinearSolver(
                        ).solve(matrix, vector / np.linalg.norm(vector))
naive_hhl_solution = HHL().solve(matrix, vector)
tridi_matrix = TridiagonalToeplitz(NUM_QUBITS, a, b)
tridi_solution = HHL().solve(tridi_matrix, vector)

print('classical euclidean norm:', classical_solution.euclidean_norm)
print('naive euclidean norm:', naive_hhl_solution.euclidean_norm)
print('tridiagonal euclidean norm:', tridi_solution.euclidean_norm)
```

    classical euclidean norm: 1.237833351044751
    naive euclidean norm: 1.209980623111888
    tridiagonal euclidean norm: 1.2094577218705271

정확한 방법과 효율적인 구현 간의 리소스 차이를 비교할 수도 있습니다. $2\times 2$ 시스템 크기는 정확한 알고리즘이 더 적은 리소스를 필요로 한다는 점에서 다시 특별하지만 시스템 크기를 늘리면 정확한 방법이 큐비트 수에서 지수적으로 확장되는 반면 `TridiagonalToeplitz`는 다항식이라는 것을 알 수 있습니다.

```python
from qiskit import transpile

MAX_QUBITS = 4
a = 1
b = -1/3

i = 1
# 리소스 사용량을 비교하기 위해 다양한 큐비트 수에 대한 회로 깊이를 계산합니다 (경고 : 실행하는 데 시간이 오래 걸립니다).
naive_depths = []
tridi_depths = []
for n_qubits in range(1, MAX_QUBITS+1):
    matrix = diags([b, a, b],
                   [-1, 0, 1],
                   shape=(2**n_qubits, 2**n_qubits)).toarray()
    vector = np.array([1] + [0]*(2**n_qubits -1))

    naive_hhl_solution = HHL().solve(matrix, vector)
    tridi_matrix = TridiagonalToeplitz(n_qubits, a, b)
    tridi_solution = HHL().solve(tridi_matrix, vector)

    naive_qc = transpile(naive_hhl_solution.state,
                         basis_gates=['id', 'rz', 'sx', 'x', 'cx'])
    tridi_qc = transpile(tridi_solution.state,
                         basis_gates=['id', 'rz', 'sx', 'x', 'cx'])

    naive_depths.append(naive_qc.depth())
    tridi_depths.append(tridi_qc.depth())
    i +=1
```


```python
sizes = [f"{2**n_qubits}×{2**n_qubits}"
         for n_qubits in range(1, MAX_QUBITS+1)]
columns = ['size of the system',
           'quantum_solution depth',
           'tridi_solution depth']
data = np.array([sizes, naive_depths, tridi_depths])
ROW_FORMAT ="{:>23}" * (len(columns) + 2)
for team, row in zip(columns, data):
    print(ROW_FORMAT.format(team, *row))
```

         size of the system                    2×2                    4×4                    8×8                  16×16
     quantum_solution depth                    334                   2593                  34008                 403899
       tridi_solution depth                    565                   5107                  14756                  46552


구현이 여전히 지수적 리소스가 필요한 것처럼 보이는 이유는 현재 조건부 회전 구현(섹션 2.B의 3단계)이 정확하기 때문입니다($n_l$에서 지수적 리소스가 필요함). 대신 Tridiagonal과 비교하여 기본 구현에 필요한 리소스가 얼마나 더 많은지 계산할 수 있습니다. 이는 $e^{iAt}$를 구현하는 방법만 다르기 때문입니다.


```python
print('excess:',
      [naive_depths[i] - tridi_depths[i] for i in range(0, len(naive_depths))])
```

    excess: [-231, -2514, 19252, 357347]


가까운 장래에 조건부 회전의 다항식 구현을 얻기 위해 `qiskit.circuit.library.arithmetics.PiecewiseChebyshev`를 통합할 계획입니다.

이제 observable 주제로 돌아가서 `observable` 및 `circuit_results` 속성에 무엇이 포함되어 있는지 알아볼 수 있습니다.

솔루션 벡터 $\mathbf{x}$의 함수를 계산하는 방법은 `.solve()` 메서드에 `LinearSystemObservable`을 입력으로 제공하는 것입니다. 입력으로 제공할 수 있는 두 가지 유형의 `LinearSystemObservable`가 있습니다.


```python
from linear_solvers.observables import AbsoluteAverage, MatrixFunctional
```

벡터 `{latex} \mathbf{x}=(x_1,...,x_N)`의 경우 `AbsoluteAverage` observable은 $|\frac{1}{N}\sum_{i=1}^{N}x_i|$를 계산합니다.


```python
NUM_QUBITS = 1
MATRIX_SIZE = 2 ** NUM_QUBITS
# 삼중 대각 Toeplitz 대칭 행렬의 항목
a = 1
b = -1/3

matrix = diags([b, a, b],
               [-1, 0, 1],
               shape=(MATRIX_SIZE, MATRIX_SIZE)).toarray()
vector = np.array([1] + [0]*(MATRIX_SIZE - 1))
tridi_matrix = TridiagonalToeplitz(1, a, b)

average_solution = HHL().solve(tridi_matrix,
                               vector,
                               AbsoluteAverage())
classical_average = NumPyLinearSolver(
                        ).solve(matrix,
                                vector / np.linalg.norm(vector),
                                AbsoluteAverage())

print('quantum average:', average_solution.observable)
print('classical average:', classical_average.observable)
print('quantum circuit results:', average_solution.circuit_results)
```

    quantum average: 0.7499999999999962
    classical average: 0.75
    quantum circuit results: (0.4999999999999952+0j)


`MatrixFunctional` observable은 벡터 $\mathbf{x}$와 삼중 대각 Toeplitz 대칭 행렬 $B$에 대해 $\mathbf{x}^T B \mathbf{x}$를 계산합니다. 클래스는 생성자 메서드에 대해 행렬의 주 대각선 및 비대각선 값을 사용합니다.


```python
observable = MatrixFunctional(1, 1 / 2)

functional_solution = HHL().solve(tridi_matrix, vector, observable)
classical_functional = NumPyLinearSolver(
                          ).solve(matrix,
                                  vector / np.linalg.norm(vector),
                                  observable)

print('quantum functional:', functional_solution.observable)
print('classical functional:', classical_functional.observable)
print('quantum circuit results:', functional_solution.circuit_results)
```

    quantum functional: 1.8281249999999818
    classical functional: 1.828125
    quantum circuit results: [(0.6249999999999941+0j), (0.4999999999999952+0j), (0.1249999999999988+0j)]


따라서 `observable`에는 $\mathbf{x}$에 대한 함수의 최종 값이 포함되고 `circuit_results`에는 회로에서 얻은 원시 값이 포함되어 `observable`의 결과를 처리하는 데 사용됩니다.

이 '결과 처리 방법'은 `.solve()`가 사용하는 인수를 살펴보면 더 잘 설명됩니다. `solve()` 메서드는 최대 5개의 인수를 허용합니다.
```python
def solve(self, matrix: Union[np.ndarray, QuantumCircuit],
          vector: Union[np.ndarray, QuantumCircuit],
          observable: Optional[Union[LinearSystemObservable, BaseOperator,
                                     List[BaseOperator]]] = None,
          post_rotation: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
          post_processing: Optional[Callable[[Union[float, List[float]]],
                                             Union[float, List[float]]]] = None) \
        -> LinearSolverResult:
```
처음 두 개는 선형 시스템을 정의하는 행렬과 방정식의 우변에 있는 벡터이며 이미 다루었습니다. 나머지 매개변수는 솔루션 벡터 $x$에서 계산할 observable(목록)과 관련되며 두 가지 다른 방식으로 지정할 수 있습니다. 한 가지 옵션은 세 번째 및 마지막 매개변수로 `LinearSystemObservable`(목록)을 제공하는 것입니다. 또는 `observable`, `post_rotation` 및 `post_processing`의 자체 구현을 제공할 수 있습니다.
- `observable`은 observable의 기대값을 계산하는 연산자이며 예를 들어 `PauliSumOp`일 수 있습니다.
- `post_rotation`은 추가 게이트가 필요한 경우 정보를 추출하기 위해 솔루션에 적용할 회로입니다.
- `post_processing`은 계산된 확률에서 observable 값을 계산하는 함수입니다.

즉, `post_rotation` 회로만큼 많은 `circuit_results`가 있으며 `post_processing`은 `circuit_results`를 인쇄할 때 표시되는 값을 사용하여 `observable`을 인쇄할 때 표시되는 값을 얻는 방법을 알고리즘에 알려줍니다.

마지막으로 `HHL` 클래스는 생성자 메서드에서 다음 매개변수를 허용합니다.
- 오류 허용 오차: 솔루션 근사의 정확도, 기본값은 `1e-2`입니다.
- 기대값: 기대값을 평가하는 방법, 기본값은 `PauliExpectation`입니다.
- quantum instance: `QuantumInstance` 또는 백엔드, 기본값은 `Statevector` 시뮬레이션입니다.


```python
from qiskit import Aer

backend = Aer.get_backend('aer_simulator')
hhl = HHL(1e-3, quantum_instance=backend)

accurate_solution = hhl.solve(matrix, vector)
classical_solution = NumPyLinearSolver(
                    ).solve(matrix,
                            vector / np.linalg.norm(vector))

print(accurate_solution.euclidean_norm)
print(classical_solution.euclidean_norm)
```

    1.185854122563138
    1.1858541225631423


## B. 실제 양자 장치에서 HHL 실행: 최적화된 예<a id='implementationdev'></a>

이전 섹션에서는 Qiskit에서 제공하는 표준 알고리즘을 실행했으며 $7$ 큐비트를 사용하고 깊이가 ~$100$ 게이트이며 총 $54$개의 CNOT 게이트가 필요함을 확인했습니다. 이러한 숫자는 현재 사용 가능한 하드웨어에 적합하지 않으므로 이러한 양을 줄여야 합니다. 특히 CNOT 게이트는 단일 큐비트 게이트보다 충실도가 나쁘기 때문에 CNOT 수를 $5$배 줄이는 것이 목표입니다. 또한 큐비트 수를 문제의 원래 설명과 같이 $4$개로 줄일 수 있습니다. Qiskit 메서드는 일반적인 문제에 대해 작성되었기 때문에 $3$개의 추가 보조 큐비트가 필요합니다.

그러나 게이트와 큐비트 수를 줄이는 것만으로는 실제 하드웨어에서 솔루션에 대한 좋은 근사값을 얻을 수 없습니다. 이는 회로 실행 중 발생하는 오류와 판독 오류의 두 가지 오류 소스가 있기 때문입니다.

Qiskit은 모든 기본 상태를 개별적으로 준비하고 측정하여 판독 오류를 완화하는 모듈을 제공합니다. 이 주제에 대한 자세한 내용은 Dewes 등의 논문에서 찾을 수 있습니다.<sup>[3](#readouterr)</sup> 오류를 완화하기 위해 Richardson 외삽법을 사용하여 각 CNOT 게이트를 각각 $1$, $3$ 및 $5$ CNOT로 대체하여 회로를 세 번 실행하여 오류를 0으로 제한할 수 있습니다<sup>[4](#richardson)</sup>. 아이디어는 이론적으로 세 개의 회로가 동일한 결과를 생성해야 하지만 실제 하드웨어에서 CNOT를 추가하면 오류가 증폭된다는 것입니다. 오류가 증폭된 결과를 얻었고 각 경우에 오류가 얼마나 증폭되었는지 추정할 수 있으므로 수량을 재결합하여 분석 솔루션에 더 가까운 새 결과를 얻을 수 있습니다.

아래에서는 다음 형식의 문제에 사용할 수 있는 최적화된 회로를 제공합니다.
$$A = \begin{pmatrix}a & b\\b & a \end{pmatrix}\quad \text{and} \quad |b\rangle=\begin{pmatrix}\cos(\theta) \\ \sin(\theta)\end{pmatrix},\quad a,b,\theta\in\mathbb{R}$$

다음 최적화는 삼중 대각 대칭 행렬에 대한 HHL에 대한 작업에서 추출되었습니다<sup>[[5]](#tridi)</sup>, 이 특정 회로는 UniversalQCompiler 소프트웨어의 도움으로 파생되었습니다<sup>[[6]](#qcompiler)</sup>.

```python
from qiskit import QuantumRegister, QuantumCircuit
import numpy as np

t = 2  # 최적화되지 않았습니다. 연습으로,
       # 최상의 결과를 얻을 수 있는 값으로 설정하십시오. 해결 방법은 8절을 참조하십시오.

NUM_QUBITS = 4  # 총 큐비트 수
nb = 1  # 해를 나타내는 큐비트 수
nl = 2  # 고유값을 나타내는 큐비트 수

theta = 0  # |b>를 정의하는 각도

a = 1  # 행렬 대각선
b = -1/3  # 행렬 비대각선

# 양자 및 고전 레지스터 초기화
qr = QuantumRegister(NUM_QUBITS)

# 양자 회로 생성
qc = QuantumCircuit(qr)

qrb = qr[0:nb]
qrl = qr[nb:nb+nl]
qra = qr[nb+nl:nb+nl+1]

# 상태 준비.
qc.ry(2*theta, qrb[0])

# QPE with e^{iAt}
for qu in qrl:
    qc.h(qu)

qc.p(a*t, qrl[0])
qc.p(a*t*2, qrl[1])

qc.u(b*t, -np.pi/2, np.pi/2, qrb[0])


# \lambda_{1}에 대한 제어된 e^{iAt}:
params=b*t

qc.p(np.pi/2,qrb[0])
qc.cx(qrl[0],qrb[0])
qc.ry(params,qrb[0])
qc.cx(qrl[0],qrb[0])
qc.ry(-params,qrb[0])
qc.p(3*np.pi/2,qrb[0])

# \lambda_{2}에 대한 제어된 e^{2iAt}:
params = b*t*2

qc.p(np.pi/2,qrb[0])
qc.cx(qrl[1],qrb[0])
qc.ry(params,qrb[0])
qc.cx(qrl[1],qrb[0])
qc.ry(-params,qrb[0])
qc.p(3*np.pi/2,qrb[0])

# 역 QFT
qc.h(qrl[1])
qc.rz(-np.pi/4,qrl[1])
qc.cx(qrl[0],qrl[1])
qc.rz(np.pi/4,qrl[1])
qc.cx(qrl[0],qrl[1])
qc.rz(-np.pi/4,qrl[0])
qc.h(qrl[0])

# 고유값 회전
t1=(-np.pi +np.pi/3 - 2*np.arcsin(1/3))/4
t2=(-np.pi -np.pi/3 + 2*np.arcsin(1/3))/4
t3=(np.pi -np.pi/3 - 2*np.arcsin(1/3))/4
t4=(np.pi +np.pi/3 + 2*np.arcsin(1/3))/4

qc.cx(qrl[1],qra[0])
qc.ry(t1,qra[0])
qc.cx(qrl[0],qra[0])
qc.ry(t2,qra[0])
qc.cx(qrl[1],qra[0])
qc.ry(t3,qra[0])
qc.cx(qrl[0],qra[0])
qc.ry(t4,qra[0])
qc.measure_all()

print(f"Depth: {qc.depth()}")
print(f"CNOTS: {qc.count_ops()['cx']}")
qc.draw(fold=-1)
```

    Depth: 26
    CNOTS: 10





    
    



아래 코드는 회로, 실제 하드웨어 백엔드 및 사용할 큐비트 세트를 입력으로 받아 지정된 장치에서 실행할 수 있는 인스턴스를 반환합니다. $3$ 및 $5$ CNOT를 사용하여 회로를 만드는 것은 동일하지만 올바른 양자 회로로 transpile 메서드를 호출합니다.

실제 하드웨어 장치는 정기적으로 재보정해야 하며 특정 큐비트 또는 게이트의 충실도는 시간이 지남에 따라 변경될 수 있습니다. 또한 칩마다 큐비트 연결이 다릅니다. 지정된 장치에서 연결되지 않은 두 큐비트 간에 2 큐비트 게이트를 수행하는 회로를 실행하려고 하면 트랜스파일러가 SWAP 게이트를 추가합니다. 따라서 다음 코드를 실행하기 전에 IBM Quantum Experience 웹페이지<sup>[[7]](#qexperience)</sup>에서 확인하고 주어진 시간에 올바른 연결과 가장 낮은 오류율을 가진 큐비트 세트를 선택하는 것이 좋습니다.


```python
from qiskit import IBMQ, transpile
from qiskit.utils.mitigation import complete_meas_cal

provider = IBMQ.load_account()

backend = provider.get_backend('ibmq_quito') # 실제 하드웨어를 사용하여 보정
layout = [2,3,0,4]
chip_qubits = 5

# 실제 하드웨어용으로 트랜스파일된 회로
qc_qa_cx = transpile(qc, backend=backend, initial_layout=layout)
```

다음 단계는 판독 오류를 완화하는 데 사용되는 추가 회로를 만드는 것입니다<sup>[[3]](#readouterr)</sup>.


```python
meas_cals, state_labels = complete_meas_cal(qubit_list=layout,
                                            qr=QuantumRegister(chip_qubits))
qcs = meas_cals + [qc_qa_cx]

job = backend.run(qcs, shots=10)
```

다음 그림<sup>[[5]](#tridi)</sup>은 위의 회로를 $10$개의 다른 초기 상태에 대해 실제 하드웨어에서 실행한 결과를 보여줍니다. $x$축은 각 경우에 초기 상태를 정의하는 각도 $\theta$를 나타냅니다. 결과는 판독 오류를 완화한 다음 $1$, $3$ 및 $5$ CNOT가 있는 회로의 결과에서 회로 실행 중에 발생하는 오류를 외삽하여 얻었습니다.

<img src="images/norm_public.png">

CNOT에서 오류 완화 또는 외삽 없이 결과를 비교하십시오<sup>[5](#tridi)</sup>.

<img src="images/noerrmit_public.png">

## 8. 문제<a id='problems'></a>

##### 실제 하드웨어:

1. 최적화된 예제의 시간 매개변수를 설정합니다.

<details>
    <summary> 해결 방법(확장하려면 클릭)</summary>
    t = 2.344915690192344

가장 작은 고유값을 정확하게 나타낼 수 있도록 설정하는 것이 가장 좋은 결과입니다. 그 이유는 해당 역수가 해에 가장 큰 기여를 하기 때문입니다.
</details>

2. 주어진 회로 '`qc`'에서 $3$ 및 $5$ CNOT에 대해 트랜스파일된 회로를 만듭니다. 회로를 만들 때 `transpile()` 함수를 사용할 때 이러한 연속적인 CNOT 게이트가 취소되지 않도록 장벽을 추가해야 합니다.
3. 실제 하드웨어에서 회로를 실행하고 결과에 이차 적합을 적용하여 외삽된 값을 얻습니다.

## 9. 참고 문헌<a id='references'></a>

<!-- vale off -->

1. J. R. Shewchuk. An Introduction to the Conjugate Gradient Method Without the Agonizing Pain. Technical Report CMU-CS-94-125, School of Computer Science, Carnegie Mellon University, Pittsburgh, Pennsylvania, March 1994.<a id='conjgrad'></a> 
2. A. W. Harrow, A. Hassidim, and S. Lloyd, “Quantum algorithm for linear systems of equations,” Phys. Rev. Lett. 103.15 (2009), p. 150502.<a id='hhl'></a>
3. A. Dewes, F. R. Ong, V. Schmitt, R. Lauro, N. Boulant, P. Bertet, D. Vion, and D. Esteve, “Characterization of a two-transmon processor with individual single-shot qubit readout,” Phys. Rev. Lett. 108, 057002 (2012). <a id='readouterr'></a>
4. N. Stamatopoulos, D. J. Egger, Y. Sun, C. Zoufal, R. Iten, N. Shen, and S. Woerner, “Option Pricing using Quantum Computers,” arXiv:1905.02666 . <a id='richardson'></a>
5. A. Carrera Vazquez, A. Frisch, D. Steenken, H. S. Barowski, R. Hiptmair, and S. Woerner, “Enhancing Quantum Linear System Algorithm by Richardson Extrapolation,” ACM Trans. Quantum Comput. 3 (2022).<a id='tridi'></a>
6. R. Iten, O. Reardon-Smith, L. Mondada, E. Redmond, R. Singh Kohli, R. Colbeck, “Introduction to UniversalQCompiler,” arXiv:1904.01072 .<a id='qcompiler'></a>
7. https://quantum-computing.ibm.com/ .<a id='qexperience'></a>
8. D. Bucher, J. Mueggenburg, G. Kus, I. Haide, S. Deutschle, H. Barowski, D. Steenken, A. Frisch, "Qiskit Aqua: Solving linear systems of equations with the HHL algorithm" https://github.com/Qiskit/qiskit-tutorials/blob/master/legacy_tutorials/aqua/linear_systems_of_equations.ipynb


```python
# pylint: disable=unused-import
import qiskit.tools.jupyter
%qiskit_version_table
```


<h3>Version Information</h3><table><tr><th>Qiskit Software</th><th>Version</th></tr><tr><td><code>qiskit-terra</code></td><td>0.21.0</td></tr><tr><td><code>qiskit-aer</code></td><td>0.10.4</td></tr><tr><td><code>qiskit-ibmq-provider</code></td><td>0.19.2</td></tr><tr><td><code>qiskit</code></td><td>0.37.2</td></tr><tr><td><code>qiskit-nature</code></td><td>0.4.1</td></tr><tr><td><code>qiskit-finance</code></td><td>0.3.2</td></tr><tr><td><code>qiskit-optimization</code></td><td>0.4.0</td></tr><tr><td><code>qiskit-machine-learning</code></td><td>0.4.0</td></tr><tr><th>System information</th></tr><tr><td>Python version</td><td>3.8.13</td></tr><tr><td>Python compiler</td><td>Clang 13.1.6 (clang-1316.0.21.2.5)</td></tr><tr><td>Python build</td><td>default, Aug 29 2022 05:17:23</td></tr><tr><td>OS</td><td>Darwin</td></tr><tr><td>CPUs</td><td>8</td></tr><tr><td>Memory (Gb)</td><td>32.0</td></tr><tr><td colspan='2'>Wed Sep 21 15:26:35 2022 BST</td></tr></table>

