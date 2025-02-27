
```language:docs/docs/tutorials/plan-and-execute/plan-and-execute_ko.ipynb
// ... existing code ...
    past_steps: Annotated[List[Tuple], operator.add] # past_steps 필드: 튜플 리스트 타입, operator.add로 어노테이트됨
// ... existing code ...
```

**코드 설명**

이 코드는 `PlanExecute`라는 `TypedDict` 내에서 `past_steps` 필드를 정의합니다. 이 필드는 에이전트가 이전에 수행한 단계들을 추적하는 데 사용됩니다. 각 단계는 튜플로 표현되며, 이 튜플은 단계의 설명과 그 단계의 결과를 포함합니다.

*   **`past_steps`**: 이 필드는 에이전트가 이전에 실행한 단계들을 저장하는 리스트입니다.
*   **`List[Tuple]`**: 이 타입 힌트는 `past_steps`가 튜플들의 리스트임을 나타냅니다. 각 튜플은 개별 단계를 나타냅니다.
*   **`Tuple`**: 각 튜플은 단계의 설명과 그 단계의 결과를 포함합니다. 예를 들어, `("검색 엔진을 사용하여 '미국 오픈 우승자' 검색", "2023년 미국 오픈 우승자는 노박 조코비치입니다.")`와 같은 형태일 수 있습니다.
*   **`Annotated[List[Tuple], operator.add]`**: `Annotated`는 타입 힌트에 추가적인 메타데이터를 추가하는 데 사용됩니다. 여기서는 `operator.add`를 사용하여 `past_steps` 리스트가 업데이트될 때마다 이전 단계들이 유지되도록 합니다. 즉, 새로운 단계가 추가될 때마다 이전 단계들이 삭제되지 않고 리스트에 계속 누적됩니다.

**예시**

다음은 `past_steps` 필드의 예시입니다.

```python
past_steps: List[Tuple] = [
    ("검색 엔진을 사용하여 '호주 오픈 우승자' 검색", "2024년 호주 오픈 남자 단식 우승자는 얀니크 시너입니다."),
    ("검색 엔진을 사용하여 '얀니크 시너 고향' 검색", "얀니크 시너의 고향은 이탈리아의 산 칸디도입니다.")
]
```

이 예시에서 `past_steps`는 두 개의 튜플을 포함하는 리스트입니다. 각 튜플은 에이전트가 수행한 단계를 나타냅니다. 첫 번째 튜플은 "검색 엔진을 사용하여 '호주 오픈 우승자' 검색" 단계를 나타내고, 그 결과는 "2024년 호주 오픈 남자 단식 우승자는 얀니크 시너입니다."입니다. 두 번째 튜플은 "검색 엔진을 사용하여 '얀니크 시너 고향' 검색" 단계를 나타내고, 그 결과는 "얀니크 시너의 고향은 이탈리아의 산 칸디도입니다."입니다.

**`operator.add`의 역할**

`operator.add`는 `Annotated`를 사용하여 `past_steps` 필드에 추가적인 의미를 부여합니다. `operator.add`는 리스트에 새로운 항목이 추가될 때마다 이전 항목들을 유지하도록 합니다. 이는 에이전트가 이전 단계들을 기억하고 이를 바탕으로 계획을 재조정하거나 최종 응답을 생성하는 데 유용합니다.

예를 들어, `past_steps`가 다음과 같이 초기화되었다고 가정해 보겠습니다.

```python
past_steps: List[Tuple] = [("A", "B")]
```

그리고 새로운 단계가 추가되면 다음과 같이 됩니다.

```python
past_steps += [("C", "D")]
```

`operator.add`가 없으면 `past_steps`는 `[("C", "D")]`로 덮어쓰여질 수 있습니다. 그러나 `operator.add`를 사용하면 `past_steps`는 `[("A", "B"), ("C", "D")]`가 되어 이전 단계들이 유지됩니다.

이러한 방식으로 `past_steps` 필드는 에이전트가 이전에 수행한 모든 단계들을 추적하고, 이를 바탕으로 더 나은 결정을 내릴 수 있도록 돕습니다.

알겠습니다. `PlanExecute` TypedDict를 사용하는 예시 코드를 작성해 드리겠습니다.

```python
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
import operator


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


# PlanExecute 타입의 인스턴스 생성
example: PlanExecute = {
    "input": "2024년 호주 오픈 남자 단식 우승자는 누구이며, 그의 고향은 어디인가?",
    "plan": [
        "2024년 호주 오픈 남자 단식 우승자 검색",
        "우승자의 고향 검색"
    ],
    "past_steps": [],
    "response": ""
}

# 초기 상태 출력
print("초기 상태:", example)

# 첫 번째 단계 실행 후 상태 업데이트
example["past_steps"].append(("2024년 호주 오픈 남자 단식 우승자 검색", "얀니크 시너"))
example["plan"].pop(0)  # 첫 번째 단계 완료 후 계획에서 제거

print("\n첫 번째 단계 실행 후 상태:", example)

# 두 번째 단계 실행 후 상태 업데이트
example["past_steps"].append(("얀니크 시너의 고향 검색", "이탈리아 산 칸디도"))
example["plan"].pop(0)  # 두 번째 단계 완료 후 계획에서 제거
example["response"] = "2024년 호주 오픈 남자 단식 우승자는 얀니크 시너이며, 그의 고향은 이탈리아 산 칸디도입니다."

print("\n두 번째 단계 실행 후 상태:", example)
```

**코드 설명:**

1.  **`PlanExecute` TypedDict 정의:**
    *   `PlanExecute`는 `input`, `plan`, `past_steps`, `response` 필드를 가지는 TypedDict로 정의됩니다.
    *   `past_steps`는 `Annotated`를 사용하여 `operator.add`로 어노테이트되어, 단계가 추가될 때마다 이전 단계들이 유지되도록 합니다.
2.  **`PlanExecute` 인스턴스 생성:**
    *   `example` 변수는 `PlanExecute` 타입의 인스턴스로 초기화됩니다.
    *   `input`은 사용자 질문, `plan`은 계획된 단계 리스트, `past_steps`는 빈 리스트, `response`는 빈 문자열로 설정됩니다.
3.  **단계 실행 및 상태 업데이트:**
    *   각 단계가 실행된 후 `past_steps`에 단계 설명과 결과가 튜플로 추가됩니다.
    *   `plan`에서 완료된 단계는 `pop(0)`을 사용하여 제거됩니다.
    *   마지막 단계에서는 `response`에 최종 답변이 저장됩니다.
4.  **상태 출력:**
    *   각 단계 실행 후 `example` 변수의 상태가 출력되어, 상태 변화를 확인할 수 있습니다.

이 예시 코드는 `PlanExecute` TypedDict를 사용하여 에이전트의 상태를 관리하고, 각 단계를 실행하면서 상태를 업데이트하는 방법을 보여줍니다. `past_steps`는 에이전트가 이전에 수행한 단계들을 추적하는 데 사용되며, `operator.add` 어노테이션은 이전 단계들이 유지되도록 보장합니다.
