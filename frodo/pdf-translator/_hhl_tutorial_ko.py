# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] tags=["remove_cell"]
# # HHL 알고리즘과 Qiskit 구현을 사용한 선형 방정식 시스템 해결
# -

# 이 튜토리얼에서는 HHL 알고리즘을 소개하고, 회로를 유도하며, Qiskit을 사용하여 구현합니다. HHL을 시뮬레이터와 5 큐빗 장치에서 실행하는 방법을 보여줍니다.

# + [markdown] tags=["contents"]
# ## 목차
# 1. [소개](#introduction)
# 2. [HHL 알고리즘](#hhlalg) 
#     1. [수학적 배경](#mathbackground)
#     2. [HHL 설명](#hhldescription)
#     3. [HHL 내의 양자 위상 추정 (QPE)](#qpe)
#     4. [비정확한 QPE](#qpe2)
# 3. [예제 1: 4-큐빗 HHL](#example1)
# 4. [Qiskit 구현](#implementation)
#     1. [시뮬레이터에서 HHL 실행: 일반적인 방법](#implementationsim)
#     2. [실제 양자 장치에서 HHL 실행: 최적화된 예제](#implementationdev)
# 5. [문제](#problems)
# 6. [참고문헌](#references)
# -

# ## 1. 소개 <a id='introduction'></a>
#
# 우리는 다양한 분야의 많은 실제 응용에서 선형 방정식 시스템을 봅니다. 예로는 편미분 방정식의 해, 금융 모델의 보정, 유체 시뮬레이션 또는 수치 필드 계산이 포함됩니다. 문제는 주어진 행렬 $A\in\mathbb{C}^{N\times N}$과 벡터 $\vec{b}\in\mathbb{C}^{N}$에 대해 $A\vec{x}=\vec{b}$를 만족하는 $\vec{x}\in\mathbb{C}^{N}$를 찾는 것으로 정의할 수 있습니다.
#
# 예를 들어, $N=2$를 취하면,
#
# $$A = \begin{pmatrix}1 & -1/3\\-1/3 & 1 \end{pmatrix},\quad \vec{x}=\begin{pmatrix} x_{1}\\ x_{2}\end{pmatrix}\quad \text{and} \quad \vec{b}=\begin{pmatrix}1 \\ 0\end{pmatrix}$$
#
# 그러면 문제는 다음과 같이 쓸 수 있습니다: `{latex} x_{1}, x_{2}\in\mathbb{C}`를 찾아라
# $$\begin{cases}x_{1} - \frac{x_{2}}{3} = 1 \\ -\frac{x_{1}}{3} + x_{2} = 0\end{cases} $$
#
# 선형 방정식 시스템은 $A$가 행 또는 열당 최대 $s$개의 0이 아닌 항목을 가질 때 $s$-희소라고 합니다. 고전 컴퓨터를 사용하여 크기 $N$의 $s$-희소 시스템을 해결하는 데는 켤레 기울기 방법을 사용하여 $\mathcal{ O }(Ns\kappa\log(1/\epsilon))$의 실행 시간이 필요합니다 <sup>[1](#conjgrad)</sup>. 여기서 $\kappa$는 시스템의 조건 수를 나타내고 $\epsilon$은 근사치의 정확도를 나타냅니다.
#
# HHL 알고리즘은 $\mathcal{ O }(\log(N)s^{2}\kappa^{2}/\epsilon)$<sup>[2](#hhl)</sup>의 실행 시간 복잡도로 해의 함수를 추정합니다. 행렬 $A$는 에르미트이어야 하며, 데이터 로드, 해밀토니안 시뮬레이션 및 해의 함수 계산을 위한 효율적인 오라클이 있다고 가정합니다. 이는 시스템 크기에서의 지수적 속도 향상이며, HHL은 해 벡터의 함수만 근사할 수 있는 반면, 고전 알고리즘은 전체 해를 반환합니다.

# ## 2. HHL 알고리즘<a id='hhlalg'></a>
#
# ### A. 수학적 배경<a id='mathbackground'></a>
# 양자 컴퓨터로 선형 방정식 시스템을 해결하기 위한 첫 번째 단계는 문제를 양자 언어로 인코딩하는 것입니다. 시스템을 재조정하여 $\vec{b}$와 $\vec{x}$를 정규화된 상태로 가정하고 각각의 양자 상태 $|b\rangle$와 $|x\rangle$로 매핑할 수 있습니다. 일반적으로 사용되는 매핑은 $\vec{b}$ (또는 $\vec{x}$)의 $i^{th}$ 성분이 양자 상태 $|b\rangle$ (또는 $|x\rangle$)의 $i^{th}$ 기저 상태의 진폭에 해당하도록 합니다. 이제부터는 재조정된 문제에 집중할 것입니다
#
# $$ A|x\rangle=|b\rangle$$
#
# $A$는 에르미트이므로 스펙트럼 분해를 가집니다
# $$
# A=\sum_{j=0}^{N-1}\lambda_{j}|u_{j}\rangle\langle u_{j}|,\quad \lambda_{j}\in\mathbb{ R }
# $$
# 여기서 $|u_{j}\rangle$는 $A$의 $j^{th}$ 고유벡터이며 각각의 고유값은 $\lambda_{j}$입니다. 그러면,
# $$
# A^{-1}=\sum_{j=0}^{N-1}\lambda_{j}^{-1}|u_{j}\rangle\langle u_{j}|
# $$
# 시스템의 오른쪽은 $A$의 고유기저로 표현될 수 있습니다
# $$
# |b\rangle=\sum_{j=0}^{N-1}b_{j}|u_{j}\rangle,\quad b_{j}\in\mathbb{ C }
# $$
# HHL의 목표는 알고리즘을 종료할 때 읽기 레지스터가 상태에 있는 것입니다
# $$
# |x\rangle=A^{-1}|b\rangle=\sum_{j=0}^{N-1}\lambda_{j}^{-1}b_{j}|u_{j}\rangle
# $$
# 여기서 우리는 이미 양자 상태에 대해 암시적인 정규화 상수를 가지고 있음을 주목하십시오.

# ### B. HHL 알고리즘 설명 <a id='hhldescription'></a>
#
# 알고리즘은 모두 알고리즘 시작 시 $|0\rangle $로 설정된 세 개의 양자 레지스터를 사용합니다. 하나의 레지스터는 $n_{l}$ 하위 인덱스로 표시되며 $A$의 고유값의 이진 표현을 저장하는 데 사용됩니다. 두 번째 레지스터는 $n_{b}$로 표시되며 벡터 해를 포함하며, 이제부터 $N=2^{n_{b}}$입니다. 중간 계산 단계에서 사용되는 보조 큐빗을 위한 추가 레지스터가 있습니다. 다음 설명에서는 보조 큐빗을 무시할 수 있습니다. 각 계산의 시작 시 $|0\rangle $이며, 각 개별 연산의 끝에서 다시 $|0\rangle $로 복원됩니다.
#
# 다음은 HHL 알고리즘의 개요와 해당 회로의 고수준 그림입니다. 간단히 하기 위해 모든 계산이 정확하다고 가정하며, 비정확한 경우에 대한 자세한 설명은 섹션 [2.D.](#qpe2)에 제공됩니다.
#
# <img src="images/hhlcircuit.png" width = "75%" height = "75%">
#
# <!-- vale QiskitTextbook.Spelling = NO -->
#
# 1. 데이터 $|b\rangle\in\mathbb{ C }^{N}$을 로드합니다. 즉, 변환을 수행합니다
#     $$ |0\rangle _{n_{b}} \mapsto |b\rangle _{n_{b}} $$
# 2. 양자 위상 추정 (QPE)을 적용합니다
#     $$ U = e ^ { i A t } := \sum _{j=0}^{N-1}e ^ { i \lambda _ { j } t } |u_{j}\rangle\langle u_{j}| $$
# 	레지스터의 양자 상태는 이제 $A$의 고유기저로 표현됩니다
#     $$ \sum_{j=0}^{N-1} b _ { j } |\lambda _ {j }\rangle_{n_{l}} |u_{j}\rangle_{n_{b}} $$
#     여기서 `{latex} |\lambda _ {j }\rangle_{n_{l}}`는 $\lambda _ {j }$의 $n_{l}$-비트 이진 표현입니다.
#     
# 3. 보조 큐빗을 추가하고 $|\lambda_{ j }\rangle$에 조건부 회전을 적용합니다,
#     $$ \sum_{j=0}^{N-1} b _ { j } |\lambda _ { j }\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} \left( \sqrt { 1 - \frac { C^{2}  } { \lambda _ { j } ^ { 2 } } } |0\rangle + \frac { C } { \lambda _ { j } } |1\rangle \right) $$
# 	여기서 $C$는 정규화 상수이며, 위의 현재 형태로 표현된 것처럼, 크기에서 가장 작은 고유값 $\lambda_{min}$보다 작아야 합니다, 즉, $|C| < \lambda_{min}$.
#     
# 4. QPE$^{\dagger}$를 적용합니다. QPE에서 발생할 수 있는 오류를 무시하면, 이는 다음과 같습니다
#     $$ \sum_{j=0}^{N-1} b _ { j } |0\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} \left( \sqrt { 1 - \frac {C^{2}  } { \lambda _ { j } ^ { 2 } } } |0\rangle + \frac { C } { \lambda _ { j } } |1\rangle \right) $$
#     
# 5. 보조 큐빗을 계산 기저에서 측정합니다. 결과가 $1$이면, 레지스터는 측정 후 상태에 있습니다
#     $$ \left( \sqrt { \frac { 1 } { \sum_{j=0}^{N-1} \left| b _ { j } \right| ^ { 2 } / \left| \lambda _ { j } \right| ^ { 2 } } } \right) \sum _{j=0}^{N-1} \frac{b _ { j }}{\lambda _ { j }} |0\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} $$
# 	이는 정규화 인수에 따라 해에 해당합니다.
#
# 6. 관측 가능 $M$을 적용하여 $F(x):=\langle x|M|x\rangle$을 계산합니다.

# ### C. HHL 내의 양자 위상 추정 (QPE) <a id='qpe'></a>
#
# 양자 위상 추정은 3장에서 더 자세히 설명됩니다. 그러나 이 양자 절차는 HHL 알고리즘의 핵심이기 때문에 여기에서 정의를 다시 상기합니다. 대략적으로 말하면, 이는 단위 $U$와 고유벡터 $|\psi\rangle_{m}$ 및 고유값 $e^{2\pi i\theta}$가 주어졌을 때 $\theta$를 찾는 양자 알고리즘입니다. 이를 다음과 같이 공식적으로 정의할 수 있습니다.
#
# **정의:** $U\in\mathbb{ C }^{2^{m}\times 2^{m}}$가 단위이고 $|\psi\rangle_{m}\in\mathbb{ C }^{2^{m}}$가 고유벡터 중 하나이며 각각의 고유값은 $e^{2\pi i\theta}$입니다. **양자 위상 추정** 알고리즘, 약칭 **QPE**,은 입력으로 $U$의 단위 게이트와 상태 `{latex} |0\rangle_{n}|\psi\rangle_{m}`를 받아 상태 `{latex} |\tilde{\theta}\rangle_{n}|\psi\rangle_{m}`를 반환합니다. 여기서 $\tilde{\theta}$는 $2^{n}\theta$의 이진 근사치를 나타내며 $n$ 하위 인덱스는 $n$ 자리로 잘렸음을 나타냅니다.
# $$
# \operatorname { QPE } ( U , |0\rangle_{n}|\psi\rangle_{m} ) = |\tilde{\theta}\rangle_{n}|\psi\rangle_{m}
# $$
#
# HHL에서는 $U = e ^ { i A t }$로 QPE를 사용합니다. 여기서 $A$는 우리가 해결하고자 하는 시스템과 관련된 행렬입니다. 이 경우,
# $$
# e ^ { i A t } = \sum_{j=0}^{N-1}e^{i\lambda_{j}t}|u_{j}\rangle\langle u_{j}|
# $$
# 그런 다음, 고유벡터 `{latex} |u_{j}\rangle_{n_{b}}`, 고유값 $e ^ { i \lambda _ { j } t }$를 가진 QPE는 `{latex} |\tilde{\lambda }_ { j }\rangle_{n_{l}}|u_{j}\rangle_{n_{b}}`를 출력합니다. 여기서 $\tilde{\lambda }_ { j }$는 $2^{n_l}\frac{\lambda_ { j }t}{2\pi}$의 $n_{l}$-비트 이진 근사치를 나타냅니다. 따라서 각 $\lambda_{j}$가 $n_{l}$ 비트로 정확히 표현될 수 있다면,
# $$
# \operatorname { QPE } ( e ^ { i A t } , \sum_{j=0}^{N-1}b_{j}|0\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} ) = \sum_{j=0}^{N-1}b_{j}|\lambda_{j}\rangle_{n_{l}}|u_{j}\rangle_{n_{b}}
# $$

# ### D. 비정확한 QPE <a id='qpe2'></a>
#
# 실제로, 초기 상태에 QPE를 적용한 후 레지스터의 양자 상태는 다음과 같습니다
#
# $$ \sum _ { j=0 }^{N-1} b _ { j } \left( \sum _ { l = 0 } ^ { 2 ^ { n_{l} } - 1 } \alpha _ { l | j } |l\rangle_{n_{l}} \right)|u_{j}\rangle_{n_{b}} $$
# 여기서
#
# $$ \alpha _ { l | j } = \frac { 1 } { 2 ^ { n_{l} } } \sum _ { k = 0 } ^ { 2^{n_{l}}- 1 } \left( e ^ { 2 \pi i \left( \frac { \lambda _ { j } t } { 2 \pi } - \frac { l } { 2 ^ { n_{l} } } \right) } \right) ^ { k } $$
#
# $\tilde{\lambda_{j}}$를 $\lambda_{j}$의 최적의 $n_{l}$-비트 근사치로 나타냅니다, $1\leq j\leq N$. 그런 다음 $n_{l}$-레지스터를 다시 레이블하여 $\alpha _ { l | j }$가 `{latex} |l + \tilde { \lambda } _ { j } \rangle_{n_{l}}`의 진폭을 나타내도록 합니다. 이제,
#
# $$\alpha _ { l | j } : = \frac { 1 } { 2 ^ { n_{l}} } \sum _ { k = 0 } ^ { 2 ^ { n_{l} } - 1 } \left( e ^ { 2 \pi i \left( \frac { \lambda _ { j } t } { 2 \pi } - \frac { l + \tilde { \lambda } _ { j } } { 2 ^ { n_{l} } } \right) } \right) ^ { k }$$
#
# 각 $\frac { \lambda _ { j } t } { 2 \pi }$가 $n_{l}$ 이진 비트로 정확히 표현될 수 있다면, `{latex} \frac { \lambda _ { j } t } { 2 \pi }=\frac { \tilde { \lambda } _ { j } } { 2 ^ { n_{l} } }` $\forall j$. 따라서 이 경우 $\forall j$, $1\leq j \leq N$, $\alpha _ { 0 | j } = 1$이고 $\alpha _ { l | j } = 0 \quad \forall l \neq 0$입니다. 이 경우에만 QPE 후 레지스터의 상태를 쓸 수 있습니다
#
# $$ \sum_{j=0}^{N-1} b _ { j } |\lambda _ {j }\rangle_{n_{l}} |u_{j}\rangle_{n_{b}}$$
#
# 그렇지 않으면, $|\alpha _ { l | j }|$는 `{latex} \frac { \lambda _ { j } t } { 2 \pi } \approx \frac { l + \tilde { \lambda } _ { j } } { 2 ^ { n_{l} } }`일 때만 크며, 레지스터의 상태는
#
# $$ \sum _ { j=0 }^{N-1}  \sum _ { l = 0 } ^ { 2 ^ { n_{l} } - 1 } \alpha _ { l | j } b _ { j }|l\rangle_{n_{l}} |u_{j}\rangle_{n_{b}} $$

# ## 3. 예제: 4-큐빗 HHL<a id='example1'></a>
#
# 소개에서의 작은 예제를 사용하여 알고리즘을 설명하겠습니다. 즉,
# $$A = \begin{pmatrix}1 & -1/3\\-1/3 & 1 \end{pmatrix}\quad \text{and} \quad |b\rangle=\begin{pmatrix}1 \\ 0\end{pmatrix}$$
#
# 우리는 $n_{b}=1$ 큐빗을 사용하여 $|b\rangle$을 나타내고, 나중에 해 $|x\rangle$을 나타내며, $n_{l}=2$ 큐빗을 사용하여 고유값의 이진 표현을 저장하고, 조건부 회전이 성공했는지 여부를 저장하는 1개의 보조 큐빗을 사용합니다.
#
# 알고리즘을 설명하기 위해 약간의 속임수를 써서 $A$의 고유값을 계산하여 $n_{l}$-레지스터에서 재조정된 고유값의 정확한 이진 표현을 얻기 위해 $t$를 선택할 수 있습니다. 그러나 HHL 알고리즘 구현을 위해서는 고유값에 대한 사전 지식이 필요하지 않음을 명심하십시오. 짧은 계산을 통해
# $$\lambda_{1} = 2/3\quad\text{and}\quad\lambda_{2}=4/3$$
#
# 이전 섹션에서 QPE는 $\frac{\lambda_ { j }t}{2\pi}$의 $n_{l}$-비트 ($2$-비트인 경우) 이진 근사치를 출력한다고 언급했습니다. 따라서
# $$t=2\pi\cdot \frac{3}{8}$$
#로 설정하면 QPE는
# $$\frac{\lambda_ { 1 }t}{2\pi} = 1/4\quad\text{and}\quad\frac{\lambda_ { 2 }t}{2\pi}=1/2$$
#에 대한 $2$-비트 이진 근사치를 제공합니다. 이는 각각
# $$|01\rangle_{n_{l}}\quad\text{and}\quad|10\rangle_{n_{l}}$$
#입니다.
#
# 고유벡터는 각각
# $$|u_{1}\rangle=\frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ -1\end{pmatrix}\quad\text{and}\quad|u_{2}\rangle=\frac{1}{\sqrt{2}}\begin{pmatrix}1 \\ 1\end{pmatrix}$$
#
# 다시 말하지만, HHL 구현을 위해 고유벡터를 계산할 필요가 없음을 명심하십시오. 사실, $N$ 차원의 일반적인 에르미트 행렬 $A$는 최대 $N$개의 서로 다른 고유값을 가질 수 있으므로 이를 계산하는 데 $\mathcal{O}(N)$ 시간이 걸리며 양자 이점이 사라질 것입니다.
#
# 그런 다음 $|b\rangle$을 $A$의 고유기저로 쓸 수 있습니다
# $$|b\rangle _{n_{b}}=\sum_{j=1}^{2}\frac{1}{\sqrt{2}}|u_{j}\rangle _{n_{b}}$$
#
# 이제 HHL 알고리즘의 다양한 단계를 진행할 준비가 되었습니다.
#
# 1. 이 예제에서 상태 준비는 간단합니다. $|b\rangle=|0\rangle$이기 때문입니다.
# 2. QPE를 적용하면 다음과 같습니다
# $$
# \frac{1}{\sqrt{2}}|01\rangle|u_{1}\rangle + \frac{1}{\sqrt{2}}|10\rangle|u_{2}\rangle
# $$
# 3. $C=1/8$로 조건부 회전을 수행합니다. 이는 가장 작은 (재조정된) 고유값 $\frac {1} {4}$보다 작습니다. 여기서 $C$ 상수는 가장 작은 (재조정된) 고유값 $\frac {1} {4}$보다 작아야 하지만, 보조 큐빗이 측정될 때 $|1>$ 상태에 있을 확률이 크도록 가능한 한 크게 선택해야 합니다.
# $$\frac{1}{\sqrt{2}}|01\rangle|u_{1}\rangle\left( \sqrt { 1 - \frac { (1/8)^{2}  } {(1/4)^{2} } } |0\rangle + \frac { 1/8 } { 1/4 } |1\rangle \right) + \frac{1}{\sqrt{2}}|10\rangle|u_{2}\rangle\left( \sqrt { 1 - \frac { (1/8)^{2}  } {(1/2)^{2} } } |0\rangle + \frac { 1/8 } { 1/2 } |1\rangle \right)
# $$
# $$
# =\frac{1}{\sqrt{2}}|01\rangle|u_{1}\rangle\left( \sqrt { 1 - \frac { 1  } {4 } } |0\rangle + \frac { 1 } { 2 } |1\rangle \right) + \frac{1}{\sqrt{2}}|10\rangle|u_{2}\rangle\left( \sqrt { 1 - \frac { 1  } {16 } } |0\rangle + \frac { 1 } { 4 } |1\rangle \right)
# $$
# 4. QPE$^{\dagger}$를 적용한 후 양자 컴퓨터는 상태에 있습니다
# $$
# \frac{1}{\sqrt{2}}|00\rangle|u_{1}\rangle\left( \sqrt { 1 - \frac { 1  } {4 } } |0\rangle + \frac { 1 } { 2 } |1\rangle \right) + \frac{1}{\sqrt{2}}|00\rangle|u_{2}\rangle\left( \sqrt { 1 - \frac { 1  } {16 } } |0\rangle + \frac { 1 } { 4 } |1\rangle \right)
# $$
# 5. 보조 큐빗을 측정할 때 결과가 $1$이면 상태는 다음과 같습니다
# $$
# \frac{\frac{1}{\sqrt{2}}|00\rangle|u_{1}\rangle\frac { 1 } { 2 } |1\rangle + \frac{1}{\sqrt{2}}|00\rangle|u_{2}\rangle\frac { 1 } { 4 } |1\rangle}{\sqrt{5/32}}
# $$
# 빠른 계산을 통해 다음을 보여줍니다
# $$
# \frac{\frac{1}{2\sqrt{2}}|u_{1}\rangle+ \frac{1}{4\sqrt{2}}|u_{2}\rangle}{\sqrt{5/32}} = \frac{|x\rangle}{||x||}
# $$
# 6. 추가 게이트를 사용하지 않고 $|x\rangle$의 노름을 계산할 수 있습니다: 이는 이전 단계에서 보조 큐빗을 측정할 때 $1$을 측정할 확률입니다.
# $$
# P(|1\rangle) = \left(\frac{1}{2\sqrt{2}}\right)^{2} + \left(\frac{1}{4\sqrt{2}}\right)^{2} = \frac{5}{32} = ||x||^{2}
# $$

# ## 4. Qiskit 구현<a id='implementation'></a>
#
# 이제 예제의 문제를 분석적으로 해결했으므로 이를 사용하여 양자 시뮬레이터와 실제 하드웨어에서 HHL을 실행하는 방법을 설명하겠습니다. 다음은 `quantum_linear_solvers`라는 Qiskit 기반 패키지를 사용하며, 이 [저장소](https://github.com/anedumla/quantum_linear_solvers)에서 찾을 수 있으며 해당 `Readme` 파일에 설명된 대로 설치할 수 있습니다. 양자 시뮬레이터의 경우, `quantum_linear_solvers`는 가장 간단한 예제에서 행렬 $A$와 $|b\rangle$만 입력으로 요구하는 HHL 알고리즘의 구현을 이미 제공합니다. 일반적인 에르미트 행렬과 임의의 초기 상태를 NumPy 배열로 알고리즘에 제공할 수 있지만, 이러한 경우 양자 알고리즘은 지수적 속도 향상을 달성하지 못합니다. 기본 구현은 정확하며 따라서 큐빗 수에 대해 지수적입니다. 임의의 양자 상태를 정확하게 준비하거나 일반적인 에르미트 행렬 $A$에 대해 정확한 연산 $e^{iAt}$을 수행할 수 있는 다항식 자원을 가진 알고리즘은 없습니다. 특정 문제에 대한 효율적인 구현을 알고 있다면, 행렬 및/또는 벡터를 `QuantumCircuit` 객체로 제공할 수 있습니다. 대안으로, 이미 삼중 대각선 Toeplitz 행렬에 대한 효율적인 구현이 있으며, 미래에는 더 많은 것이 있을 수 있습니다.
#
# 그러나, 현재 작성 시점에서 기존 양자 컴퓨터는 노이즈가 많고 작은 회로만 실행할 수 있습니다. 따라서 섹션 [4.B.](#implementationdev)에서는 예제에 속하는 문제 클래스에 사용할 수 있는 최적화된 회로를 살펴보고 양자 컴퓨터의 노이즈를 처리하는 기존 절차를 언급하겠습니다.

# ## A. 시뮬레이터에서 HHL 실행: 일반적인 방법<a id='implementationsim'></a>
#
# 이 페이지의 코드를 실행하려면 [선형 솔버 패키지](https://github.com/anedumla/quantum_linear_solvers)를 설치해야 합니다. 다음 명령을 통해 설치할 수 있습니다:
#
# @@@
# pip install git+https://github.com/anedumla/quantum_linear_solvers
# @@@
#
# 선형 시스템 문제를 해결하기 위한 모든 알고리즘의 인터페이스는 `LinearSolver`입니다. 해결할 문제는 `solve()` 메서드가 호출될 때만 지정됩니다:
# @@@python
# LinearSolver(...).solve(matrix, vector)
# @@@
#
# 가장 간단한 구현은 행렬과 벡터를 NumPy 배열로 사용합니다. 아래에서는 솔루션을 검증하기 위해 `NumPyLinearSolver` (고전 알고리즘)도 생성합니다.

import numpy as np
# pylint: disable=line-too-long
from linear_solvers import NumPyLinearSolver, HHL
matrix = np.array([[1, -1/3], [-1/3, 1]])
vector = np.array([1, 0])
naive_hhl_solution = HHL().solve(matrix, vector)

# 고전 솔버의 경우, 오른쪽을 재조정해야 합니다 (즉, `vector / np.linalg.norm(vector)`) HHL 내에서 벡터가 양자 상태로 인코딩될 때 발생하는 재정규화를 고려해야 합니다.

classical_solution = NumPyLinearSolver().solve(matrix,
                                               vector/np.linalg.norm(vector))

# `linear_solvers` 패키지에는 특정 유형의 행렬에 대한 효율적인 구현을 위한 자리 표시자로 의도된 `matrices`라는 폴더가 포함되어 있습니다. 작성 시점에서 유일하게 진정으로 효율적인 구현은 `TridiagonalToeplitz` 클래스입니다. 삼중 대각선 Toeplitz 대칭 실수 행렬은 다음과 같은 형태입니다
# $$A = \begin{pmatrix}a & b & 0 & 0\\b & a & b & 0 \\ 0 & b & a & b \\ 0 & 0 & b & a \end{pmatrix}, a,b\in\mathbb{R}$$
# (이 설정에서는 HHL 알고리즘이 입력 행렬이 에르미트임을 가정하므로 비대칭 행렬은 고려하지 않습니다).
#
# 예제의 행렬 $A$가 이 형태이므로 `TridiagonalToeplitz(num_qubits, a, b)`의 인스턴스를 생성하고 배열을 입력으로 사용하여 시스템을 해결한 결과와 비교할 수 있습니다.

from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
tridi_matrix = TridiagonalToeplitz(1, 1, -1 / 3)
tridi_solution = HHL().solve(tridi_matrix, vector)

# HHL 알고리즘은 시스템의 크기에 대해 고전적 대응보다 지수적으로 빠르게 해를 찾을 수 있습니다 (즉, 다항식 복잡도 대신 로그 복잡도). 그러나 이 지수적 속도 향상의 대가는 전체 해 벡터를 얻지 못한다는 것입니다.
# 대신, 벡터 $x$에서 함수 (소위 관측 가능)를 계산하여 해에 대한 정보를 얻습니다.
# 이는 `solve()`에 의해 반환된 `LinearSolverResult` 객체에 반영되며, 다음 속성을 포함합니다
# - `state`: 솔루션을 준비하는 회로 또는 벡터로서의 솔루션
# - `euclidean_norm`: 알고리즘이 계산 방법을 알고 있는 경우 유클리드 노름
# - `observable`: 계산된 (목록의) 관측 가능
# - `circuit_results`: (목록의) 회로에서의 관측 가능 결과
#
# `observable`과 `circuit_results`를 잠시 무시하고 이전에 얻은 솔루션을 확인해 보겠습니다.
#
# 먼저, `classical_solution`은 고전 알고리즘의 결과이므로 `.state`를 호출하면 배열을 반환합니다:

print('고전 상태:', classical_solution.state)

# 다른 두 예제는 양자 알고리즘이었으므로 양자 상태에만 접근할 수 있습니다. 이는 솔루션 상태를 준비하는 양자 회로를 반환하여 달성됩니다:

print('단순 상태:')
print(naive_hhl_solution.state)
print('삼중 대각선 상태:')
print(tridi_solution.state)

# 벡터 `{latex} \mathbf{x}=(x_1,\dots,x_N)`의 유클리드 노름은 $||\mathbf{x}||=\sqrt{\sum_{i=1}^N x_i^2}$로 정의됩니다. 따라서 보조 큐빗을 5단계에서 측정할 때 $1$을 측정할 확률은 $\mathbf{x}$의 제곱 노름입니다. 이는 HHL 알고리즘이 항상 해의 유클리드 노름을 계산할 수 있음을 의미하며, 결과의 정확성을 비교할 수 있습니다:

print('고전 유클리드 노름:', classical_solution.euclidean_norm)
print('단순 유클리드 노름:', naive_hhl_solution.euclidean_norm)
print('삼중 대각선 유클리드 노름:', tridi_solution.euclidean_norm)

# 솔루션 벡터를 구성 요소별로 비교하는 것은 더 까다롭습니다. 이는 양자 알고리즘에서 전체 솔루션 벡터를 얻을 수 없다는 아이디어를 다시 반영합니다. 그러나 교육 목적으로, 실제로 얻은 다양한 솔루션 벡터가 벡터 구성 요소 수준에서도 좋은 근사치임을 확인할 수 있습니다.
#
# 이를 위해 먼저 `quantum_info` 패키지에서 `Statevector`를 사용하고 올바른 벡터 구성 요소를 추출해야 합니다. 즉, 보조 큐빗 (회로의 하단)이 $1$이고 작업 큐빗 (회로의 중간 두 개)이 $0$인 구성 요소입니다. 따라서 우리는 `10000` 및 `10001` 상태에 관심이 있으며, 이는 각각 솔루션 벡터의 첫 번째 및 두 번째 구성 요소에 해당합니다.

# +
from qiskit.quantum_info import Statevector

naive_sv = Statevector(naive_hhl_solution.state).data
tridi_sv = Statevector(tridi_solution.state).data

# 벡터 구성 요소 추출; 10000(bin) == 16 & 10001(bin) == 17
naive_full_vector = np.array([naive_sv[16], naive_sv[17]])
tridi_full_vector = np.array([tridi_sv[16], tridi_sv[17]])

print('단순 원시 솔루션 벡터:', naive_full_vector)
print('삼중 원시 솔루션 벡터:', tridi_full_vector)


# -

# 처음에는 구성 요소가 실수가 아닌 복소수이기 때문에 잘못된 것처럼 보일 수 있습니다. 그러나 주목할 점은 허수 부분이 매우 작으며, 컴퓨터 정확도 때문일 가능성이 높으며 이 경우 무시할 수 있습니다 (배열의 `.real` 속성을 사용하여 실수 부분을 얻습니다).
#
# 다음으로, 회로의 다양한 부분에서 오는 상수를 억제하기 위해 벡터를 각각의 노름으로 나눕니다. 그런 다음 전체 솔루션 벡터는 위에서 계산한 각각의 유클리드 노름으로 이 정규화된 벡터를 곱하여 복원할 수 있습니다:

# +
def get_solution_vector(solution):
    """LinearSolverResult에서 시뮬레이션된 상태 벡터를 추출하고 정규화합니다."""
    solution_vector = Statevector(solution.state).data[16:18].real
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)

print('전체 단순 솔루션 벡터:', get_solution_vector(naive_hhl_solution))
print('전체 삼중 솔루션 벡터:', get_solution_vector(tridi_solution))
print('고전 상태:', classical_solution.state)
# -

# `naive_hhl_solution`이 정확한 것은 놀라운 일이 아닙니다. 기본적으로 사용된 모든 방법이 정확하기 때문입니다. 그러나 `tridi_solution`은 $2\times 2$ 시스템 크기 경우에만 정확합니다. 더 큰 행렬의 경우 아래의 약간 더 큰 예제에서 보여주듯이 근사치가 될 것입니다.

# +
from scipy.sparse import diags

NUM_QUBITS = 2
MATRIX_SIZE = 2 ** NUM_QUBITS
# 삼중 대각선 Toeplitz 대칭 행렬의 항목
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

print('고전 유클리드 노름:', classical_solution.euclidean_norm)
print('단순 유클리드 노름:', naive_hhl_solution.euclidean_norm)
print('삼중 대각선 유클리드 노름:', tridi_solution.euclidean_norm)
# -

# 정확한 방법과 효율적인 구현의 자원 차이를 비교할 수도 있습니다. $2\times 2$ 시스템 크기는 다시 특별한 경우로, 정확한 알고리즘이 더 적은 자원을 필요로 하지만, 시스템 크기를 증가시키면 정확한 방법이 큐빗 수에 대해 지수적으로 확장되는 반면 `TridiagonalToeplitz`는 다항식입니다.

# +
from qiskit import transpile

MAX_QUBITS = 4
a = 1
b = -1/3

i = 1
# 자원 사용을 비교하기 위해 다른 큐빗 수에 대한 회로 깊이를 계산합니다 (경고: 실행하는 데 시간이 걸립니다)
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
# -

sizes = [f"{2**n_qubits}×{2**n_qubits}"
         for n_qubits in range(1, MAX_QUBITS+1)]
columns = ['시스템 크기',
           '양자_솔루션 깊이',
           '삼중_솔루션 깊이']
data = np.array([sizes, naive_depths, tridi_depths])
ROW_FORMAT ="{:>23}" * (len(columns) + 2)
for team, row in zip(columns, data):
    print(ROW_FORMAT.format(team, *row))

# 구현이 여전히 지수적 자원을 필요로 하는 이유는 현재 조건부 회전 구현 (섹션 2.B의 3단계)이 정확하기 때문입니다 (즉, $n_l$에 대해 지수적 자원이 필요함). 대신 기본 구현이 $e^{iAt}$를 어떻게 구현하는지에 따라 기본 구현이 Tridiagonal보다 얼마나 더 많은 자원을 필요로 하는지 계산할 수 있습니다:

print('초과:',
      [naive_depths[i] - tridi_depths[i] for i in range(0, len(naive_depths))])

# 가까운 미래에는 `qiskit.circuit.library.arithmetics.PiecewiseChebyshev`를 통합하여 조건부 회전의 다항식 구현을 얻을 계획입니다.
#
# 이제 관측 가능 주제로 돌아가서 `observable`과 `circuit_results` 속성이 무엇을 포함하는지 알아보겠습니다.
#
# 솔루션 벡터 $\mathbf{x}$의 함수를 계산하는 방법은 `.solve()` 메서드에 `LinearSystemObservable`을 입력으로 제공하는 것입니다. 사용할 수 있는 두 가지 유형의 `LinearSystemObservable`이 있으며 입력으로 제공할 수 있습니다:

from linear_solvers.observables import AbsoluteAverage, MatrixFunctional

# 벡터 `{latex} \mathbf{x}=(x_1,...,x_N)`에 대해, `AbsoluteAverage` 관측 가능은 $|\frac{1}{N}\sum_{i=1}^{N}x_i|$를 계산합니다.

# +
NUM_QUBITS = 1
MATRIX_SIZE = 2 ** NUM_QUBITS
# 삼중 대각선 Toeplitz 대칭 행렬의 항목
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

print('양자 평균:', average_solution.observable)
print('고전 평균:', classical_average.observable)
print('양자 회로 결과:', average_solution.circuit_results)
# -

# `MatrixFunctional` 관측 가능은 벡터 $\mathbf{x}$와 삼중 대각선 대칭 Toeplitz 행렬 $B$에 대해 $\mathbf{x}^T B \mathbf{x}$를 계산합니다. 클래스는 행렬의 주 대각선 및 비대각선 값을 생성자 메서드에 사용합니다.

# +
observable = MatrixFunctional(1, 1 / 2)

functional_solution = HHL().solve(tridi_matrix, vector, observable)
classical_functional = NumPyLinearSolver(
                          ).solve(matrix,
                                  vector / np.linalg.norm(vector),
                                  observable)

print('양자 함수:', functional_solution.observable)
print('고전 함수:', classical_functional.observable)
print('양자 회로 결과:', functional_solution.circuit_results)
# -

# 따라서, `observable`은 $\mathbf{x}$에 대한 함수의 최종 값을 포함하고, `circuit_results`는 회로에서 얻은 원시 값을 포함하며, `observable`의 결과를 처리하는 데 사용됩니다.
#
# 이 '결과를 처리하는 방법'은 `.solve()`가 어떤 인수를 받는지 살펴보면 더 잘 설명됩니다. `solve()` 메서드는 최대 다섯 개의 인수를 허용합니다:
# @@@python
# def solve(self, matrix: Union[np.ndarray, QuantumCircuit],
#           vector: Union[np.ndarray, QuantumCircuit],
#           observable: Optional[Union[LinearSystemObservable, BaseOperator,
#                                      List[BaseOperator]]] = None,
#           post_rotation: Optional[Union[QuantumCircuit, List[QuantumCircuit]]] = None,
#           post_processing: Optional[Callable[[Union[float, List[float]]],
#                                              Union[float, List[float]]]] = None) \
#         -> LinearSolverResult:
# @@@
# 첫 번째 두 개는 선형 시스템을 정의하는 행렬과 방정식의 오른쪽 벡터로, 이미 다루었습니다. 나머지 매개변수는 솔루션 벡터 $x$에서 계산할 (목록의) 관측 가능에 관한 것이며 두 가지 다른 방법으로 지정할 수 있습니다. 한 가지 옵션은 세 번째 및 마지막 매개변수로 (목록의) `LinearSystemObservable`(들)을 제공하는 것입니다. 대안으로, `observable`, `post_rotation` 및 `post_processing`의 자체 구현을 제공할 수 있으며, 여기서
# - `observable`은 관측 가능의 기대값을 계산하는 연산자이며, 예를 들어 `PauliSumOp`일 수 있습니다
# - `post_rotation`은 추가 게이트가 필요한 경우 솔루션에 적용할 회로입니다.
# - `post_processing`은 계산된 확률에서 관측 가능의 값을 계산하는 함수입니다.
#
# 즉, `circuit_results`는 `post_rotation` 회로만큼 있으며, `post_processing`은 우리가 `circuit_results`를 인쇄할 때 볼 수 있는 값을 사용하여 `observable`을 인쇄할 때 볼 수 있는 값을 얻는 방법을 알려줍니다.
#
# 마지막으로, `HHL` 클래스는 생성자 메서드에서 다음 매개변수를 허용합니다:
# - 오류 허용치: 해의 근사의 정확도, 기본값은 `1e-2`
# - 기대값: 기대값이 평가되는 방법, 기본값은 `PauliExpectation`
# - 양자 인스턴스: `QuantumInstance` 또는 백엔드, 기본값은 `Statevector` 시뮬레이션

# +
from qiskit import Aer

backend = Aer.get_backend('aer_simulator')
hhl = HHL(1e-3, quantum_instance=backend)

accurate_solution = hhl.solve(matrix, vector)
classical_solution = NumPyLinearSolver(
                    ).solve(matrix,
                            vector / np.linalg.norm(vector))

print(accurate_solution.euclidean_norm)
print(classical_solution.euclidean_norm)
# -

# ## B. 실제 양자 장치에서 HHL 실행: 최적화된 예제<a id='implementationdev'></a>
#
# 이전 섹션에서는 Qiskit에서 제공하는 표준 알고리즘을 실행했으며, 이는 $7$ 큐빗을 사용하고, 깊이가 약 $100$ 게이트이며, 총 $54$ CNOT 게이트가 필요하다는 것을 보았습니다. 이러한 수치는 현재 사용 가능한 하드웨어에 적합하지 않으므로 이러한 수치를 줄여야 합니다. 특히, CNOT의 수를 $5$배 줄이는 것이 목표가 될 것입니다. 이는 단일 큐빗 게이트보다 충실도가 낮기 때문입니다. 또한, 문제의 원래 진술이 $4$ 큐빗이었으므로 큐빗 수를 $4$로 줄일 수 있습니다: Qiskit 방법은 일반적인 문제를 위해 작성되었으며, 따라서 $3$개의 추가 보조 큐빗이 필요합니다.
#
# 그러나 게이트와 큐빗 수를 줄이는 것만으로는 실제 하드웨어에서 솔루션에 대한 좋은 근사치를 제공하지 않습니다. 이는 회로 실행 중 발생하는 오류와 판독 오류라는 두 가지 오류 원인이 있기 때문입니다.
#
# Qiskit은 모든 기저 상태를 개별적으로 준비하고 측정하여 판독 오류를 완화하는 모듈을 제공합니다. 자세한 내용은 Dewes et al.의 논문에서 찾을 수 있습니다<sup>[3](#readouterr)</sup>. 오류를 완화하기 위해, 각 CNOT 게이트를 $1$, $3$ 및 $5$ CNOT으로 대체하여 회로를 세 번 실행하여 오류를 0으로 계산하는 Richardson 외삽법을 사용할 수 있습니다<sup>[4](#richardson)</sup>. 아이디어는 이론적으로 세 회로가 동일한 결과를 생성해야 하지만, 실제 하드웨어에서 CNOT을 추가하면 오류가 증폭된다는 것입니다. 증폭된 오류로 결과를 얻었음을 알고 있으며, 각 경우에 오류가 얼마나 증폭되었는지 추정할 수 있으므로, 양을 재조합하여 분석적 솔루션에 더 가까운 새로운 결과를 얻을 수 있습니다.
#
# 아래는 다음과 같은 형태의 문제에 사용할 수 있는 최적화된 회로입니다
# $$A = \begin{pmatrix}a & b\\b & a \end{pmatrix}\quad \text{and} \quad |b\rangle=\begin{pmatrix}\cos(\theta) \\ \sin(\theta)\end{pmatrix},\quad a,b,\theta\in\mathbb{R}$$
#
# 다음 최적화는 삼중 대각선 대칭 행렬에 대한 HHL에 대한 작업에서 추출되었으며<sup>[[5]](#tridi)</sup>, 이 특정 회로는 UniversalQCompiler 소프트웨어의 도움으로 유도되었습니다<sup>[[6]](#qcompiler)</sup>.
#

# +
from qiskit import QuantumRegister, QuantumCircuit
import numpy as np

t = 2  # 최적이 아님; 연습으로, 최상의 결과를 얻을 수 있는 값을 설정하십시오. 솔루션은 섹션 8에 있습니다.

NUM_QUBITS = 4  # 총 큐빗 수
nb = 1  # 해를 나타내는 큐빗 수
nl = 2  # 고유값을 나타내는 큐빗 수

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


# Controlled e^{iAt} on \lambda_{1}:
params=b*t

qc.p(np.pi/2,qrb[0])
qc.cx(qrl[0],qrb[0])
qc.ry(params,qrb[0])
qc.cx(qrl[0],qrb[0])
qc.ry(-params,qrb[0])
qc.p(3*np.pi/2,qrb[0])

# Controlled e^{2iAt} on \lambda_{2}:
params = b*t*2

qc.p(np.pi/2,qrb[0])
qc.cx(qrl[1],qrb[0])
qc.ry(params,qrb[0])
qc.cx(qrl[1],qrb[0])
qc.ry(-params,qrb[0])
qc.p(3*np.pi/2,qrb[0])

# Inverse QFT
qc.h(qrl[1])
qc.rz(-np.pi/4,qrl[1])
qc.cx(qrl[0],qrl[1])
qc.rz(np.pi/4,qrl[1])
qc.cx(qrl[0],qrl[1])
qc.rz(-np.pi/4,qrl[0])
qc.h(qrl[0])

# Eigenvalue rotation
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

print(f"깊이: {qc.depth()}")
print(f"CNOTS: {qc.count_ops()['cx']}")
qc.draw(fold=-1)
# -

# 아래 코드는 회로, 실제 하드웨어 백엔드 및 사용할 큐빗 세트를 입력으로 받아 지정된 장치에서 실행할 수 있는 인스턴스를 반환합니다. $3$ 및 $5$ CNOT으로 회로를 생성하는 것은 올바른 양자 회로로 `transpile` 메서드를 호출하는 것과 동일합니다.
#
# 실제 하드웨어 장치는 정기적으로 재보정이 필요하며, 특정 큐빗 또는 게이트의 충실도는 시간이 지남에 따라 변경될 수 있습니다. 또한, 다른 칩은 다른 큐빗 연결성을 가지고 있습니다. 지정된 장치에서 연결되지 않은 두 큐빗 사이에 2-큐빗 게이트를 수행하는 회로를 실행하려고 하면, 트랜스파일러는 SWAP 게이트를 추가합니다. 따라서 다음 코드를 실행하기 전에 IBM Quantum Experience 웹페이지<sup>[[7]](#qexperience)</sup>를 확인하고 주어진 시간에 올바른 연결성과 가장 낮은 오류율을 가진 큐빗 세트를 선택하는 것이 좋습니다.

# + tags=["uses-hardware"]
from qiskit import IBMQ, transpile
from qiskit.utils.mitigation import complete_meas_cal

provider = IBMQ.load_account()

backend = provider.get_backend('ibmq_quito') # 실제 하드웨어를 사용하여 보정
layout = [2,3,0,4]
chip_qubits = 5

# 실제 하드웨어를 위한 트랜스파일된 회로
qc_qa_cx = transpile(qc, backend=backend, initial_layout=layout)
# -

# 다음 단계는 판독 오류를 완화하기 위해 추가 회로를 생성하는 것입니다<sup>[[3]](#readouterr)</sup>.

# + tags=["uses-hardware"]
meas_cals, state_labels = complete_meas_cal(qubit_list=layout,
                                            qr=QuantumRegister(chip_qubits))
qcs = meas_cals + [qc_qa_cx]

job = backend.run(qcs, shots=10)
# -

# 다음 그림<sup>[[5]](#tridi)</sup>은 $10$개의 다른 초기 상태에 대해 실제 하드웨어에서 회로를 실행한 결과를 보여줍니다. $x$-축은 각 경우의 초기 상태를 정의하는 각도 $\theta$를 나타냅니다. 결과는 판독 오류를 완화한 후 $1$, $3$ 및 $5$ CNOT으로 회로를 사용하여 오류를 외삽한 후 얻은 것입니다.
#
# <img src="images/norm_public.png">
#
# 오류 완화 및 CNOT에서의 외삽 없이 얻은 결과와 비교하십시오<sup>[5](#tridi)</sup>.
#
# <img src="images/noerrmit_public.png">

# ## 8. 문제<a id='problems'></a>
#
# ##### 실제 하드웨어:
#
# 1. 최적화된 예제의 시간 매개변수를 설정하십시오.
#
# <details>
#     <summary> 솔루션 (확장하려면 클릭)</summary>
#     t = 2.344915690192344
#
# 가장 작은 고유값을 정확히 표현할 수 있도록 설정하는 것이 최상의 결과입니다. 이는 해에서 가장 큰 기여를 할 것입니다
# </details>
#
# 2. 주어진 회로 '`qc`'에서 $3$ 및 $5$ CNOT에 대한 트랜스파일된 회로를 생성하십시오. 회로를 생성할 때 `transpile()` 함수를 사용할 때 이러한 연속적인 CNOT 게이트가 취소되지 않도록 장벽을 추가해야 합니다.
# 3. 실제 하드웨어에서 회로를 실행하고 외삽된 값을 얻기 위해 CNOT에서의 외삽을 적용하십시오.

# ## 9. 참고문헌<a id='references'></a>
#
# <!-- vale off -->
#
# 1. J. R. Shewchuk. An Introduction to the Conjugate Gradient Method Without the Agonizing Pain. Technical Report CMU-CS-94-125, School of Computer Science, Carnegie Mellon University, Pittsburgh, Pennsylvania, March 1994.<a id='conjgrad'></a> 
# 2. A. W. Harrow, A. Hassidim, and S. Lloyd, "Quantum algorithm for linear systems of equations," Phys. Rev. Lett. 103.15 (2009), p. 150502.<a id='hhl'></a>
# 3. A. Dewes, F. R. Ong, V. Schmitt, R. Lauro, N. Boulant, P. Bertet, D. Vion, and D. Esteve, "Characterization of a two-transmon processor with individual single-shot qubit readout," Phys. Rev. Lett. 108, 057002 (2012). <a id='readouterr'></a>
# 4. N. Stamatopoulos, D. J. Egger, Y. Sun, C. Zoufal, R. Iten, N. Shen, and S. Woerner, "Option Pricing using Quantum Computers," arXiv:1905.02666 . <a id='richardson'></a>
# 5. A. Carrera Vazquez, A. Frisch, D. Steenken, H. S. Barowski, R. Hiptmair, and S. Woerner, "Enhancing Quantum Linear System Algorithm by Richardson Extrapolation," ACM Trans. Quantum Comput. 3 (2022).<a id='tridi'></a>
# 6. R. Iten, O. Reardon-Smith, L. Mondada, E. Redmond, R. Singh Kohli, R. Colbeck, "Introduction to UniversalQCompiler," arXiv:1904.01072 .<a id='qcompiler'></a>
# 7. https://quantum-computing.ibm.com/ .<a id='qexperience'></a>
# 8. D. Bucher, J. Mueggenburg, G. Kus, I. Haide, S. Deutschle, H. Barowski, D. Steenken, A. Frisch, "Qiskit Aqua: Solving linear systems of equations with the HHL algorithm" https://github.com/Qiskit/qiskit-tutorials/blob/master/legacy_tutorials/aqua/linear_systems_of_equations.ipynb

# pylint: disable=unused-import
import qiskit.tools.jupyter
# %qiskit_version_table
