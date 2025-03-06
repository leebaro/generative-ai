# %%
!pip install pypandoc nbformat


# %%
!python convert.py ./hhl_tutorial.md hhl_tutorial.ipynb


# %% [markdown]
## HHL 및 Qiskit 구현을 사용한 선형 시스템 방정식 풀이

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

### B. HHL 알고리즘 설m명 <a id='hhldescription'></a>

이 알고리즘은 세 개의 양자 레지스터를 사용하며, 알고리즘 시작 시 모두 $|0\rangle$로 설정됩니다. 하나의 레지스터는 아래 첨자 $n_{l}$로 표시되며, $A$의 고유값의 이진 표현을 저장하는 데 사용됩니다. 두 번째 레지스터는 $n_{b}$로 표시되며, 벡터 해를 포함하고, 앞으로 $N=2^{n_{b}}$입니다. 계산의 중간 단계에 사용되는 보조 큐비트에 대한 추가 레지스터가 있습니다. 다음 설명에서는 각 계산 시작 시 $|0\rangle$이고 각 개별 연산이 끝날 때 다시 $|0\rangle$로 복원되므로 보조 레지스터는 무시할 수 있습니다.

다음은 해당 회로의 개략적인 그림과 함께 HHL 알고리즘의 개요입니다. 단순화를 위해 모든 계산은 이후 설명에서 정확하다고 가정하며, 정확하지 않은 경우에 대한 자세한 설명은 섹션 [2.D.](#qpe2)에 제공됩니다.



