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
# # 선형 방정식 시스템을 HHL 및 Qiskit 구현을 사용하여 해결하기
# -

# 이 튜토리얼에서는 HHL 알고리즘을 소개하고, 회로를 유도하며, Qiskit을 사용하여 구현합니다. 시뮬레이터와 5 큐빗 장치에서 HHL을 실행하는 방법을 보여줍니다.

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
#     1. [시뮬레이터에서 HHL 실행: 최적화된 예제](#implementationdev)
# 5. [문제](#problems)
# 6. [참고문헌](#references)
# -

# ## 1. 소개 <a id='introduction'></a>
#
# 우리는 다양한 분야의 많은 실제 응용 프로그램에서 선형 방정식 시스템을 봅니다. 예로는 부분 미분 방정식의 해, 금융 모델의 보정, 유체 시뮬레이션 또는 수치 필드 계산이 있습니다. 문제는 행렬 $A\in\mathbb{C}^{N\times N}$과 벡터 $\vec{b}\in\mathbb{C}^{N}$이 주어졌을 때, $A\vec{x}=\vec{b}$를 만족하는 $\vec{x}\in\mathbb{C}^{N}$을 찾는 것으로 정의할 수 있습니다.
#
# 예를 들어, $N=2$로 설정하면,
#
# $$A = \begin{pmatrix}1 & -1/3\\-1/3 & 1 \end{pmatrix},\quad \vec{x}=\begin{pmatrix} x_{1}\\ x_{2}\end{pmatrix}\quad \text{and} \quad \vec{b}=\begin{pmatrix}1 \\ 0\end{pmatrix}$$
#
# 그러면 문제는 다음과 같이 쓸 수 있습니다: $x_{1}, x_{2}\in\mathbb{C}$을 찾아라
# $$\begin{cases}x_{1} - \frac{x_{2}}{3} = 1 \\ -\frac{x_{1}}{3} + x_{2} = 0\end{cases} $$
#
# 선형 방정식 시스템은 $A$가 행 또는 열당 최대 $s$개의 0이 아닌 항목을 갖는 경우 $s$-희소라고 합니다. 고전 컴퓨터를 사용하여 크기 $N$의 $s$-희소 시스템을 해결하려면 공액 기울기 방법을 사용하여 $\mathcal{ O }(Ns\kappa\log(1/\epsilon))$ 실행 시간이 필요합니다 <sup>[1](#conjgrad)</sup>. 여기서 $\kappa$는 시스템의 조건 수를 나타내고 $\epsilon$은 근사의 정확도를 나타냅니다.
#
# HHL 알고리즘은 $\mathcal{ O }(\log(N)s^{2}\kappa^{2}/\epsilon)$ 실행 시간 복잡도로 해의 함수를 추정합니다<sup>[2](#hhl)</sup>. 행렬 $A$는 Hermitian이어야 하며, 데이터 로드, 해밀토니안 시뮬레이션 및 해의 함수 계산을 위한 효율적인 오라클이 있다고 가정합니다. 이는 시스템 크기에 대한 지수적 속도 향상으로, HHL은 해 벡터의 함수만 근사할 수 있는 반면, 고전적 알고리즘은 전체 해를 반환할 수 있습니다.

# ### A. 수학적 배경 <a id='mathbackground'></a>
#
# $A$가 $N\times N$ Hermitian 행렬이라고 가정합니다. 스펙트럼 정리에 따르면, $A$는 고유 벡터 $|u_{j}\rangle$와 고유값 $\lambda_{j}$에 대해 다음과 같이 분해될 수 있습니다:
# $$
# A=\sum_{j=0}^{N-1}\lambda_{j}|u_{j}\rangle\langle u_{j}|,\quad \lambda_{j}\in\mathbb{ R }
# $$
# 여기서 $|u_{j}\rangle$는 $A$의 $j^{th}$ 고유 벡터이며, 각각의 고유값은 $\lambda_{j}$입니다. 그러면,
# $$
# A^{-1}=\sum_{j=0}^{N-1}\lambda_{j}^{-1}|u_{j}\rangle\langle u_{j}|
# $$
# 시스템의 오른쪽 항은 $A$의 고유 기저로 다음과 같이 쓸 수 있습니다:
# $$
# |b\rangle=\sum_{j=0}^{N-1}b_{j}|u_{j}\rangle,\quad b_{j}\in\mathbb{ C }
# $$
# HHL의 목표는 알고리즘을 종료할 때 판독 레지스터가 다음 상태에 있도록 하는 것입니다:
# $$
# |x\rangle=A^{-1}|b\rangle=\sum_{j=0}^{N-1}\lambda_{j}^{-1}b_{j}|u_{j}\rangle
# $$
# 여기서 우리는 이미 양자 상태에 대해 암묵적인 정규화 상수를 가지고 있다는 점에 유의하십시오.

# ### B. HHL 알고리즘 설명 <a id='hhldescription'></a>
#
# 알고리즘은 모두 $|0\rangle $으로 설정된 세 개의 양자 레지스터를 사용합니다. 하나의 레지스터는 $A$의 고유값의 이진 표현을 저장하는 데 사용되며, 하위 인덱스 $n_{l}$로 표시합니다. 두 번째 레지스터는 벡터 해를 포함하며, 이제부터 $N=2^{n_{b}}$입니다. 계산의 중간 단계에서 사용되는 보조 큐빗을 위한 추가 레지스터가 있습니다. 다음 설명에서는 보조 큐빗을 무시할 수 있습니다. 계산의 시작 시점에 $|0\rangle $ 상태이며, 각 개별 작업의 끝에서 다시 $|0\rangle $로 복원됩니다.
#
# 다음은 HHL 알고리즘의 개요와 해당 회로의 고급 그림입니다. 간단히 하기 위해 모든 계산이 정확하다고 가정하며, 비정확한 경우에 대한 자세한 설명은 [2.D.](#qpe2) 섹션에 나와 있습니다.
#
# <img src="images/hhlcircuit.png" width = "75%" height = "75%">
#
# <!-- vale QiskitTextbook.Spelling = NO -->
#
# 1.  데이터 $|b\rangle\in\mathbb{ C }^{N}$을 로드합니다. 즉, 변환을 수행합니다:
#     $$ |0\rangle _{n_{b}} \mapsto |b\rangle _{n_{b}} $$
# 2. 양자 위상 추정 (QPE)을 적용합니다:
#     $$ U = e ^ { i A t } := \sum _{j=0}^{N-1}e ^ { i \lambda _ { j } t } |u_{j}\rangle\langle u_{j}| $$
# 	이제 레지스터의 양자 상태는 $A$의 고유 기저로 표현됩니다.

# \t이제 레지스터의 양자 상태는 $A$의 고유 기저로 표현됩니다:
#     $$ \sum_{j=0}^{N-1} b _ { j } |\lambda _ {j }\rangle_{n_{l}} |u_{j}\rangle_{n_{b}} $$
#     여기서 $|\lambda _ {j }\rangle_{n_{l}}$는 $\lambda _ {j }$의 $n_{l}$-비트 이진 표현입니다.
#     
# 3. 보조 큐빗을 추가하고 $|\lambda_{ j }\rangle$에 조건부로 회전을 적용합니다:
#     $$ \sum_{j=0}^{N-1} b _ { j } |\lambda _ { j }\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} \left( \sqrt { 1 - \frac { C^{2}  } { \lambda _ { j } ^ { 2 } } } |0\rangle + \frac { C } { \lambda _ { j } } |1\rangle \right) $$
# \t여기서 $C$는 정규화 상수이며, 위의 현재 형태로 표현된 것처럼 크기에서 가장 작은 고유값 $\lambda_{min}$보다 작아야 합니다, 즉 $|C| < \lambda_{min}$.
#     
# 4. QPE^{\dagger}를 적용합니다. QPE에서 발생할 수 있는 오류를 무시하면, 결과는 다음과 같습니다:
#     $$ \sum_{j=0}^{N-1} b _ { j } |0\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} \left( \sqrt { 1 - \frac {C^{2}  } { \lambda _ { j } ^ { 2 } } } |0\rangle + \frac { C } { \lambda _ { j } } |1\rangle \right) $$
#     
# 5. 계산 기저에서 보조 큐빗을 측정합니다. 결과가 $1$이면, 레지스터는 측정 후 상태에 있습니다:
#     $$ \left( \sqrt { \frac { 1 } { \sum_{j=0}^{N-1} \left| b _ { j } \right| ^ { 2 } / \left| \lambda _ { j } \right| ^ { 2 } } } \right) \sum _{j=0}^{N-1} \frac{b _ { j }}{\lambda _ { j }} |0\rangle_{n_{l}}|u_{j}\rangle_{n_{b}} $$
# \t정규화 계수에 따라 해에 해당합니다.
#
# 6. 관측 가능량 $M$을 적용하여 $F(x):=\langle x|M|x\rangle$을 계산합니다.

# ### C. HHL 내의 양자 위상 추정 (QPE) <a id='qpe'></a>
#
# 양자 위상 추정은 3장에서 더 자세히 설명됩니다. 그러나 이 양자 절차는 HHL 알고리즘의 핵심이기 때문에 여기서 정의를 상기합니다. 대략적으로 말하면, 이는 단위행렬 $U$와 고유 벡터 $|\psi\rangle_{m}$ 및 고유값 $e^{2\pi i\theta}$가 주어졌을 때 $\theta$를 찾는 양자 알고리즘입니다. 이를 공식적으로 다음과 같이 정의할 수 있습니다.
#
# **정의:** $U\in\mathbb{ C }^{2^{m}\times 2^{m}}$이 단위행렬이고 $|\psi\rangle_{m}\in\mathbb{ C }^{2^{m}}$이 고유값 $e^{2\pi i\theta}$를 갖는 고유 벡터 중 하나라고 하자. **양자 위상 추정** 알고리즘, 약칭 **QPE**,은 $U$의 단위 게이트와 상태 $|0\rangle_{n}|\psi\rangle_{m}`을 입력으로 받아 상태 $|\tilde{\theta}\rangle_{n}|\psi\rangle_{m}`을 반환합니다. 여기서 $\tilde{\theta}$는 $2^{n}\theta$의 이진 근사값을 나타내며, $n$ 하위 인덱스는 $n$ 자리로 잘렸음을 나타냅니다.
# $$
# \operatorname { QPE } ( U , |0\rangle_{n}|\psi\rangle_{m} ) = |\tilde{\theta}\rangle_{n}|\psi\rangle_{m}
# $$
#
# HHL을 위해 우리는 $U = e ^ { i A t }$로 QPE를 사용할 것입니다, 여기서 $A$는 우리가 해결하려는 시스템과 관련된 행렬입니다. 이 경우,
# $$
# e ^ { i A t } = \sum_{j=0}^{N-1}e^{i\lambda_{j}t}|u_{j}\rangle\langle u_{j}|
# $$
# 그러면 고유 벡터 $|u_{j}\rangle_{n_{b}}`는 고유값 $e ^ { i \lambda _ { j } t }`를 가지며, QPE는 $|\tilde{\lambda }_ { j }\rangle_{n_{l}}|u_{j}\rangle_{n_{b}}`를 출력합니다. 여기서 $\tilde{\lambda }_ { j }$는 $2^{n_l}\frac{\lambda_ { j }t}{2\pi}$의 $n_{l}$-비트 이진 근사값을 나타냅니다. 따라서 각 $\lambda_{j}$가 $n_{l}$ 비트로 정확하게 표현될 수 있다면,
