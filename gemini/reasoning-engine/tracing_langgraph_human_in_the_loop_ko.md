```python
!jupyter nbconvert --to markdown ./tracing_langgraph_human_in_the_loop_ko.ipynb

```

    usage: jupyter [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir]
                   [--paths] [--json] [--debug]
                   [subcommand]
    
    Jupyter: Interactive Computing
    
    positional arguments:
      subcommand     the subcommand to launch
    
    options:
      -h, --help     show this help message and exit
      --version      show the versions of core jupyter packages and exit
      --config-dir   show Jupyter config dir
      --data-dir     show Jupyter data dir
      --runtime-dir  show Jupyter runtime dir
      --paths        show all Jupyter paths. Add --json for machine-readable
                     format.
      --json         output paths as machine-readable json
      --debug        output debug information about paths
    
    Available subcommands: kernel kernelspec migrate run troubleshoot
    
    Jupyter command `jupyter-nbconvert` not found.



```python
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Building and Deploying a Human-in-the-Loop LangGraph Application with Reasoning Engine on Vertex AI

<table align="left">
  <td style="text-align: center">
    <a href="https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/langgraph_human_in_the_loop.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/colab-logo-32px.png" alt="Google Colaboratory logo"><br> Open in Colab
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fgenerative-ai%2Fmain%2Fgemini%2Freasoning-engine%2Flanggraph_human_in_the_loop.ipynb">
      <img width="32px" src="https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png" alt="Google Cloud Colab Enterprise logo"><br> Open in Colab Enterprise
    </a>
  </td>    
  <td style="text-align: center">
    <a href="https://console.cloud.google.com/vertex-ai/workbench/deploy-notebook?download_url=https://raw.githubusercontent.com/GoogleCloudPlatform/generative-ai/main/gemini/reasoning-engine/langgraph_human_in_the_loop.ipynb">
      <img src="https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32" alt="Vertex AI logo"><br> Open in Workbench
    </a>
  </td>
  <td style="text-align: center">
    <a href="https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/langgraph_human_in_the_loop.ipynb">
      <img src="https://cloud.google.com/ml-engine/images/github-logo-32px.png" alt="GitHub logo"><br> View on GitHub
    </a>
  </td>
</table>

<div style="clear: both;"></div>

<b>Share to:</b>

<a href="https://www.linkedin.com/sharing/share-offsite/?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/langgraph_human_in_the_loop.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" alt="LinkedIn logo">
</a>

<a href="https://bsky.app/intent/compose?text=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/langgraph_human_in_the_loop.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/7/7a/Bluesky_Logo.svg" alt="Bluesky logo">
</a>

<a href="https://twitter.com/intent/tweet?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/langgraph_human_in_the_loop.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/53/X_logo_2023_original.svg" alt="X logo">
</a>

<a href="https://reddit.com/submit?url=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/langgraph_human_in_the_loop.ipynb" target="_blank">
  <img width="20px" src="https://redditinc.com/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" alt="Reddit logo">
</a>

<a href="https://www.facebook.com/sharer/sharer.php?u=https%3A//github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/langgraph_human_in_the_loop.ipynb" target="_blank">
  <img width="20px" src="https://upload.wikimedia.org/wikipedia/commons/5/51/Facebook_f_logo_%282019%29.svg" alt="Facebook logo">
</a>            

| | |
|-|-|
| Author(s) | [Xiaolong Yang](https://github.com/shawn-yang-google) |


```python

```

## 개요

<mark>[Reasoning Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview) (Vertex AI의 LangChain)은 에이전트 추론 프레임워크(agent reasoning frameworks)를 구축하고 배포하는 데 도움이 되는 관리형 서비스</mark>입니다. [LangGraph](https://langchain-ai.github.io/langgraph/)는 LLM을 사용하여 상태 저장(stateful) 다중 액터 애플리케이션(multi-actor applications)을 구성하기 위한 라이브러리로서, 정교한 에이전트 및 다중 에이전트 워크플로(multi-agent workflows) 생성을 가능하게 합니다.

이 노트북은 Vertex AI의 [Reasoning Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview)을 사용하여 Human-in-the-Loop LangGraph 애플리케이션을 구축, 배포 및 테스트하는 방법을 보여줍니다. LangGraph의 강력한 워크플로 오케스트레이션(workflow orchestration)과 Vertex AI의 확장성을 결합하여 Human-in-the-Loop 생성형 AI 애플리케이션을 구축하는 방법을 배우게 됩니다.

이전 [노트북](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/reasoning-engine/intro_reasoning_engine.ipynb)에서는 도구 정의(Defining Tools), 라우터 정의(Defining a Router), LangGraph 애플리케이션 구축(Building a LangGraph Application), 로컬 테스트(Local Testing), Vertex AI에 배포(Deploying to Vertex AI), 원격 테스트(Remote Testing) 및 리소스 정리(Cleaning Up Resources)를 다루었습니다.

이 노트북은 이러한 개념을 확장하고 다음과 같은 Human-in-the-Loop 기능을 살펴봅니다.

- **도구 호출 검토 (Reviewing Tool Calls):** 도구 사용 후 사람의 감독을 구현하여 진행하기 전에 작업의 확인 및 수정을 허용합니다.
- **상태 기록 가져오기 (Fetching State History):** 감사, 분석 및 잠재적인 상태 복원을 위해 LangGraph 애플리케이션의 전체 실행 기록을 검색합니다.
- **시간 여행 (Time Travel):** 과거 결정을 이해하기 위해 특정 시점의 에이전트 상태를 검사합니다.
- **재생 (Replay):** 수정 없이 특정 체크포인트에서 실행을 다시 시작하여 일관된 결과를 보장합니다.
- **분기 (Branching):** 과거 상태를 기반으로 대체 실행 경로를 생성하여 에이전트가 다양한 가능성을 탐색하거나 이전 오류를 수정할 수 있도록 합니다.

이 노트북이 끝나면 LangGraph, Reasoning Engine 및 Vertex AI를 사용하여 사용자 정의된 Human-in-the-Loop 생성형 AI 애플리케이션을 구축하고 배포하는 기술을 갖추게 됩니다.

## 시작하기 (Get started)

### Vertex AI SDK 및 필수 패키지 설치 (Install the Vertex AI SDK and Required Packages)


```python
%pip install --upgrade --user --quiet \
    "google-cloud-aiplatform[langchain,reasoningengine]" \
    requests --force-reinstall
```

### Restart runtime

To use the newly installed packages in this Jupyter runtime, you must restart the runtime. You can do this by running the cell below, which restarts the current kernel.

The restart might take a minute or longer. After it's restarted, continue to the next step.


```python
import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```

<div class="alert alert-block alert-warning">
<b>⚠️ The kernel is going to restart. Wait until it's finished before continuing to the next step. ⚠️</b>
</div>

### Authenticate your notebook environment (Colab only)

If you're running this notebook on Google Colab, run the cell below to authenticate your environment.


```python
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
```

### Google Cloud 프로젝트 정보 설정 및 Vertex AI SDK 초기화

Vertex AI를 사용하기 전에 기존 Google Cloud 프로젝트가 있는지 확인하고 [Vertex AI API를 활성화](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com)했는지 확인하세요.

자세한 내용은 [프로젝트 및 개발 환경 설정](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)에 대한 문서를 참조하세요.


```python
import os
import dotenv
from google.cloud import trace_v1 as trace
import pandas as pd
from vertexai.preview import reasoning_engines
from vertexai.reasoning_engines._reasoning_engines import _utils

dotenv.load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATOIN")

MODEL_NAME = "gemini-2.0-flash-001"
```


```python
# PROJECT_ID = "[your-project-id]"  # @param {type:"string"}
# LOCATION = "us-central1"  # @param {type:"string"}
STAGING_BUCKET = "gs://[your-staging-bucket]"  # @param {type:"string"}

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
```

## Reasoning Engine에서 LangGraph 앱 구축 및 배포

다음 섹션에서는 Vertex AI의 Reasoning Engine을 사용하여 LangGraph 애플리케이션을 구축하고 배포하는 방법을 안내합니다. 

### 라이브러리 가져오기 (Import Libraries)

필요한 Python 라이브러리를 가져옵니다. 이러한 라이브러리는 LangGraph, Vertex AI 및 애플리케이션의 다른 구성 요소와 상호 작용하는 데 필요한 도구를 제공합니다.


```python
from langchain.load import load as langchain_load
from vertexai.preview import reasoning_engines

```

### 도구 정의 (Define Tools)

LangGraph 애플리케이션을 위한 도구를 정의하는 것으로 시작하겠습니다. 여기서는 에이전트 애플리케이션 내에서 도구 역할을 하는 사용자 정의 Python 함수를 정의합니다.

이 예에서는 <mark>사용자가 요청한 환율을 검색하는 간단한 도구</mark>를 만듭니다. 실제로 API와 상호 작용하거나, 데이터베이스를 쿼리하거나, 에이전트가 실행해야 할 수 있는 다른 작업을 수행하는 함수를 정의할 수 있습니다.

<details>
    <summary>tool의 docstring 사용 여부</summary>
Langchain에서 Tool로 function을 사용할 때 docstring도 사용하는가?

네, Langchain에서 Tool로 function을 사용할 때 docstring을 사용합니다. Langchain은 function의 docstring을 사용하여 Tool의 이름, 설명, 입력 매개변수 등을 자동으로 생성합니다. 이는 Tool을 정의하고 사용하는 과정을 간소화하며, 에이전트가 Tool을 올바르게 사용하는 데 도움이 됩니다.

예를 들어, 아래 `get_exchange_rate` 함수는 docstring을 포함하고 있습니다. Langchain은 이 docstring을 활용하여 Tool에 대한 메타데이터를 생성할 수 있습니다.

```python
def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "KRW",
    currency_date: str = "latest",
):
    """Retrieves the exchange rate between two currencies on a specified date.

    Uses the Frankfurter API (https://api.frankfurter.app/) to obtain
    exchange rate data.

    Args:
        currency_from: The base currency (3-letter currency code).
            Defaults to "USD" (US Dollar).
        currency_to: The target currency (3-letter currency code).
            Defaults to "KRW" (South Korea Won).
        currency_date: The date for which to retrieve the exchange rate.
            Defaults to "latest" for the most recent exchange rate data.
            Can be specified in YYYY-MM-DD format for historical rates.

    Returns:
        dict: A dictionary containing the exchange rate information.
            Example: {"amount": 1.0, "base": "USD", "date": "2023-11-24",
                "rates": {"KRW": 1300}}
    """
```

Langchain은 docstring을 사용하여 Tool의 `name`, `description`, `args` (입력 스키마)를 자동으로 생성합니다.  이는 에이전트가 어떤 Tool을 사용할 수 있는지, 그리고 각 Tool을 어떻게 사용해야 하는지 이해하는 데 매우 중요합니다.  따라서 Tool을 정의할 때 명확하고 유용한 docstring을 작성하는 것이 좋습니다.



</details>



```python
import requests


def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "KRW",
    currency_date: str = "latest",
):
    """Retrieves the exchange rate between two currencies on a specified date.

    Uses the Frankfurter API (https://api.frankfurter.app/) to obtain
    exchange rate data.

    Args:
        currency_from: The base currency (3-letter currency code).
            Defaults to "USD" (US Dollar).
        currency_to: The target currency (3-letter currency code).
            Defaults to "KRW" (South Korea Won).
        currency_date: The date for which to retrieve the exchange rate.
            Defaults to "latest" for the most recent exchange rate data.
            Can be specified in YYYY-MM-DD format for historical rates.

    Returns:
        dict: A dictionary containing the exchange rate information.
            Example: {"amount": 1.0, "base": "USD", "date": "2023-11-24",
                "rates": {"KRW": 1300}}
    """

    response = requests.get(
        f"https://api.frankfurter.app/{currency_date}",
        params={"from": currency_from, "to": currency_to},
    )
    return response.json()
```


```python
get_exchange_rate("USD", "KRW", "2023-11-24")
```




    {'amount': 1.0, 'base': 'USD', 'date': '2023-11-24', 'rates': {'KRW': 1305.98}}



### [`Checkpointers`](https://langchain-ai.github.io/langgraph/concepts/persistence/) 정의

LangGraph에서 [메모리(memory)는 체크포인팅/지속성(checkpointing/persistence)](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9290376)입니다. <mark style="background-color:#B18904">체크포인팅은 그래프의 각 노드에서 에이전트 실행의 [상태(state)](https://langchain-ai.github.io/langgraph/concepts/low_level/#checkpointer-state)를 저장하며, 이는 `실행 재개(Resuming execution)`, `디버깅 및 검사(Debugging and Inspection)`, 그리고 `비동기 작업(Asynchronous Operations)`에 매우 중요</mark>합니다.

LangGraph는 상태를 저장하고 로드하는 메서드를 정의하는 [체크포인터 인터페이스(Checkpointer Interface)](https://langchain-ai.github.io/langgraph/concepts/persistence/#checkpointer-interface)를 제공합니다.  
이 인터페이스를 구현하기 위해 여러 내장 체크포인터(built-in checkpointers)를 사용할 수 있습니다.

.put - 체크포인트(checkpoint)의 구성 및 메타데이터(metadata)를 저장합니다.  
.put_writes - 체크포인트에 연결된 중간 쓰기(intermediate writes)를 저장합니다 (즉, 보류 중인 쓰기). (Store intermediate writes linked to a checkpoint (i.e. pending writes).)  
.get_tuple - 주어진 구성(스레드 ID(thread_id) 및 체크포인트 ID(checkpoint_id))에 대해 체크포인트 튜플(checkpoint tuple)을 가져옵니다. 이는 그래프(graph)의 StateSnapshot을 채우는 데 사용됩니다.  
.list - 주어진 구성 및 필터 기준과 일치하는 체크포인트를 나열합니다. 이는 그래프(graph)의 상태 기록(state history)을 채우는 데 사용됩니다 (graph.get_state_history()).

다음으로, LangGraph 애플리케이션의 체크포인터에 대한 인수를 정의하고 체크포인터 빌더(checkpointer builder) 역할을 하는 사용자 정의 Python 함수를 만듭니다. 이 경우 간단한 `In Memory` 체크포인터를 정의합니다.


```python
checkpointer_kwargs = None

def checkpointer_builder(**kwargs):
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()
```

### Human-in-the-Loop LangGraph 애플리케이션 정의

이제 모든 구성 요소를 통합하여 Reasoning Engine 내에서 Human-in-the-Loop LangGraph 애플리케이션을 정의합니다.

이 애플리케이션은 사용자가 정의한 도구(tools)와 체크포인터(checkpointer)를 활용합니다. LangGraph는 이러한 상호 작용을 구조화하고 LLM의 기능을 활용하기 위한 강력한 프레임워크를 제공합니다.


```python
agent = reasoning_engines.LanggraphAgent(
    model=MODEL_NAME,
    tools=[get_exchange_rate],
    model_kwargs={"temperature": 0, "max_retries": 6},
    checkpointer_kwargs=checkpointer_kwargs,
    checkpointer_builder=checkpointer_builder,
    enable_tracing=True,
)
```

### Local Testing

This section covers local testing of your LangGraph application before deployment to ensure it behaves as expected.


```python
agent.set_up()
```


```python
inputs = {
    "messages": [
        ("user", "미국 달러에서 한국 통화로의 환율은 얼마인가요?")
    ]
}
```


```python
inputs
```




    {'messages': [('user', '미국 달러에서 한국 통화로의 환율은 얼마인가요?')]}




```python
response = agent.query(
    input=inputs,
    config={"configurable": {"thread_id": "synchronous-thread-id"}},
)

response["messages"][-1]["kwargs"]["content"]
```


```python
import pprint

pprint.pprint(response["messages"])
```

    [{'id': ['langchain', 'schema', 'messages', 'HumanMessage'],
      'kwargs': {'content': '미국 달러에서 한국 통화로의 환율은 얼마인가요?',
                 'id': 'df2873ec-789d-4543-aedf-44b277c3d250',
                 'type': 'human'},
      'lc': 1,
      'type': 'constructor'},
     {'id': ['langchain', 'schema', 'messages', 'AIMessage'],
      'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                      '"USD", '
                                                                      '"currency_to": '
                                                                      '"KRW"}',
                                                         'name': 'get_exchange_rate'}},
                 'content': '',
                 'id': 'run-282a2239-ead9-429e-8d81-3f3a0b505a2e-0',
                 'invalid_tool_calls': [],
                 'response_metadata': {'avg_logprobs': -0.0051815759922776905,
                                       'finish_reason': 'STOP',
                                       'is_blocked': False,
                                       'safety_ratings': [],
                                       'usage_metadata': {'cache_tokens_details': [],
                                                          'cached_content_token_count': 0,
                                                          'candidates_token_count': 14,
                                                          'candidates_tokens_details': [{'modality': 1,
                                                                                         'token_count': 14}],
                                                          'prompt_token_count': 144,
                                                          'prompt_tokens_details': [{'modality': 1,
                                                                                     'token_count': 144}],
                                                          'total_token_count': 158}},
                 'tool_calls': [{'args': {'currency_from': 'USD',
                                          'currency_to': 'KRW'},
                                 'id': '0819fc10-5a93-46ad-8b97-4aa609c013ec',
                                 'name': 'get_exchange_rate',
                                 'type': 'tool_call'}],
                 'type': 'ai',
                 'usage_metadata': {'input_tokens': 144,
                                    'output_tokens': 14,
                                    'total_tokens': 158}},
      'lc': 1,
      'type': 'constructor'},
     {'id': ['langchain', 'schema', 'messages', 'ToolMessage'],
      'kwargs': {'content': '{"amount": 1.0, "base": "USD", "date": "2025-02-18", '
                            '"rates": {"KRW": 1441.87}}',
                 'id': '2d34d18e-d6f9-4c0b-8727-5acb8d6b828b',
                 'name': 'get_exchange_rate',
                 'status': 'success',
                 'tool_call_id': '0819fc10-5a93-46ad-8b97-4aa609c013ec',
                 'type': 'tool'},
      'lc': 1,
      'type': 'constructor'},
     {'id': ['langchain', 'schema', 'messages', 'AIMessage'],
      'kwargs': {'content': '현재 미국 달러 환율은 1,441.87 한국 원입니다.',
                 'id': 'run-d38527f8-0efd-40fd-96f5-84b8e3db3a05-0',
                 'invalid_tool_calls': [],
                 'response_metadata': {'avg_logprobs': -0.059743279502505346,
                                       'finish_reason': 'STOP',
                                       'is_blocked': False,
                                       'safety_ratings': [],
                                       'usage_metadata': {'cache_tokens_details': [],
                                                          'cached_content_token_count': 0,
                                                          'candidates_token_count': 21,
                                                          'candidates_tokens_details': [{'modality': 1,
                                                                                         'token_count': 21}],
                                                          'prompt_token_count': 180,
                                                          'prompt_tokens_details': [{'modality': 1,
                                                                                     'token_count': 180}],
                                                          'total_token_count': 201}},
                 'tool_calls': [],
                 'type': 'ai',
                 'usage_metadata': {'input_tokens': 180,
                                    'output_tokens': 21,
                                    'total_tokens': 201}},
      'lc': 1,
      'type': 'constructor'}]


## trace 하기


```python
client = trace.TraceServiceClient()
```


```python
result = [
    r
    for r in client.list_traces(
        request=trace.types.ListTracesRequest(
            project_id=PROJECT_ID,
            # Return all traces containing `labels {key: "openinference.span.kind" value: "AGENT"}`
            filter="openinference.span.kind:AGENT",
        )
    )
]
```


```python
trace_data = client.get_trace(project_id=PROJECT_ID, trace_id=result[0].trace_id).spans[0]
trace_data
```




    span_id: 11561621918614542820
    name: "LangGraph"
    start_time {
      seconds: 1739948605
      nanos: 667520000
    }
    end_time {
      seconds: 1739948619
      nanos: 419518976
    }
    labels {
      key: "session.id"
      value: "synchronous-thread-id"
    }
    labels {
      key: "output.value"
      value: "{\"messages\": [\"content=\'미국 달러에서 한국 통화로의 환율은 얼마인가요?\' additional_kwargs={} response_metadata={} id=\'d34311bf-e868-4b68-901f-9f538e74e560\'\", \"content=\'\' additional_kwargs={\'function_call\': {\'name\': \'get_exchange_rate\', "
    }
    labels {
      key: "output.mime_type"
      value: "application/json"
    }
    labels {
      key: "openinference.span.kind"
      value: "CHAIN"
    }
    labels {
      key: "metadata"
      value: "{\"thread_id\": \"synchronous-thread-id\"}"
    }
    labels {
      key: "input.value"
      value: "{\"messages\": [[\"user\", \"미국 달러에서 한국 통화로의 환율은 얼마인가요?\"]]}"
    }
    labels {
      key: "input.mime_type"
      value: "application/json"
    }
    labels {
      key: "g.co/agent"
      value: "opentelemetry-python 1.28.2; google-cloud-trace-exporter 1.7.0"
    }



또한 [스트리밍](https://langchain-ai.github.io/langgraph/how-tos/stream-values/) 모드(streaming mode)를 활용하여 각 노드 실행 후 그래프의 전체 상태를 나타내는 그래프의 `values`를 스트리밍 방식으로 다시 가져올 수 있습니다.


```python
import pprint

for state_values in agent.stream_query(
    input=inputs,
    stream_mode="values",
    config={"configurable": {"thread_id": "streaming-thread-values"}},
):
    pprint.pprint(state_values)
    print("*"*30)
    
```

    {'messages': [{'id': ['langchain', 'schema', 'messages', 'HumanMessage'],
                   'kwargs': {'content': '미국 달러에서 한국 통화로의 환율은 얼마인가요?',
                              'id': '4af8ad3d-7fa0-4256-9c0c-3851f7152da1',
                              'type': 'human'},
                   'lc': 1,
                   'type': 'constructor'}]}
    ******************************
    {'messages': [{'id': ['langchain', 'schema', 'messages', 'HumanMessage'],
                   'kwargs': {'content': '미국 달러에서 한국 통화로의 환율은 얼마인가요?',
                              'id': '4af8ad3d-7fa0-4256-9c0c-3851f7152da1',
                              'type': 'human'},
                   'lc': 1,
                   'type': 'constructor'},
                  {'id': ['langchain', 'schema', 'messages', 'AIMessage'],
                   'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                   '"USD", '
                                                                                   '"currency_to": '
                                                                                   '"KRW"}',
                                                                      'name': 'get_exchange_rate'}},
                              'content': '',
                              'id': 'run-64e8b928-8576-481a-bf5f-4ad8165f1729-0',
                              'invalid_tool_calls': [],
                              'response_metadata': {'avg_logprobs': -0.0051815759922776905,
                                                    'finish_reason': 'STOP',
                                                    'is_blocked': False,
                                                    'safety_ratings': [],
                                                    'usage_metadata': {'cache_tokens_details': [],
                                                                       'cached_content_token_count': 0,
                                                                       'candidates_token_count': 14,
                                                                       'candidates_tokens_details': [{'modality': 1,
                                                                                                      'token_count': 14}],
                                                                       'prompt_token_count': 144,
                                                                       'prompt_tokens_details': [{'modality': 1,
                                                                                                  'token_count': 144}],
                                                                       'total_token_count': 158}},
                              'tool_calls': [{'args': {'currency_from': 'USD',
                                                       'currency_to': 'KRW'},
                                              'id': '76ca091f-f936-486b-8874-6d66b99d7672',
                                              'name': 'get_exchange_rate',
                                              'type': 'tool_call'}],
                              'type': 'ai',
                              'usage_metadata': {'input_tokens': 144,
                                                 'output_tokens': 14,
                                                 'total_tokens': 158}},
                   'lc': 1,
                   'type': 'constructor'}]}
    ******************************
    {'messages': [{'id': ['langchain', 'schema', 'messages', 'HumanMessage'],
                   'kwargs': {'content': '미국 달러에서 한국 통화로의 환율은 얼마인가요?',
                              'id': '4af8ad3d-7fa0-4256-9c0c-3851f7152da1',
                              'type': 'human'},
                   'lc': 1,
                   'type': 'constructor'},
                  {'id': ['langchain', 'schema', 'messages', 'AIMessage'],
                   'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                   '"USD", '
                                                                                   '"currency_to": '
                                                                                   '"KRW"}',
                                                                      'name': 'get_exchange_rate'}},
                              'content': '',
                              'id': 'run-64e8b928-8576-481a-bf5f-4ad8165f1729-0',
                              'invalid_tool_calls': [],
                              'response_metadata': {'avg_logprobs': -0.0051815759922776905,
                                                    'finish_reason': 'STOP',
                                                    'is_blocked': False,
                                                    'safety_ratings': [],
                                                    'usage_metadata': {'cache_tokens_details': [],
                                                                       'cached_content_token_count': 0,
                                                                       'candidates_token_count': 14,
                                                                       'candidates_tokens_details': [{'modality': 1,
                                                                                                      'token_count': 14}],
                                                                       'prompt_token_count': 144,
                                                                       'prompt_tokens_details': [{'modality': 1,
                                                                                                  'token_count': 144}],
                                                                       'total_token_count': 158}},
                              'tool_calls': [{'args': {'currency_from': 'USD',
                                                       'currency_to': 'KRW'},
                                              'id': '76ca091f-f936-486b-8874-6d66b99d7672',
                                              'name': 'get_exchange_rate',
                                              'type': 'tool_call'}],
                              'type': 'ai',
                              'usage_metadata': {'input_tokens': 144,
                                                 'output_tokens': 14,
                                                 'total_tokens': 158}},
                   'lc': 1,
                   'type': 'constructor'},
                  {'id': ['langchain', 'schema', 'messages', 'ToolMessage'],
                   'kwargs': {'content': '{"amount": 1.0, "base": "USD", "date": '
                                         '"2025-02-18", "rates": {"KRW": 1441.87}}',
                              'id': '1645a0b6-40ce-4968-966c-ec4133864dc2',
                              'name': 'get_exchange_rate',
                              'status': 'success',
                              'tool_call_id': '76ca091f-f936-486b-8874-6d66b99d7672',
                              'type': 'tool'},
                   'lc': 1,
                   'type': 'constructor'}]}
    ******************************
    {'messages': [{'id': ['langchain', 'schema', 'messages', 'HumanMessage'],
                   'kwargs': {'content': '미국 달러에서 한국 통화로의 환율은 얼마인가요?',
                              'id': '4af8ad3d-7fa0-4256-9c0c-3851f7152da1',
                              'type': 'human'},
                   'lc': 1,
                   'type': 'constructor'},
                  {'id': ['langchain', 'schema', 'messages', 'AIMessage'],
                   'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                   '"USD", '
                                                                                   '"currency_to": '
                                                                                   '"KRW"}',
                                                                      'name': 'get_exchange_rate'}},
                              'content': '',
                              'id': 'run-64e8b928-8576-481a-bf5f-4ad8165f1729-0',
                              'invalid_tool_calls': [],
                              'response_metadata': {'avg_logprobs': -0.0051815759922776905,
                                                    'finish_reason': 'STOP',
                                                    'is_blocked': False,
                                                    'safety_ratings': [],
                                                    'usage_metadata': {'cache_tokens_details': [],
                                                                       'cached_content_token_count': 0,
                                                                       'candidates_token_count': 14,
                                                                       'candidates_tokens_details': [{'modality': 1,
                                                                                                      'token_count': 14}],
                                                                       'prompt_token_count': 144,
                                                                       'prompt_tokens_details': [{'modality': 1,
                                                                                                  'token_count': 144}],
                                                                       'total_token_count': 158}},
                              'tool_calls': [{'args': {'currency_from': 'USD',
                                                       'currency_to': 'KRW'},
                                              'id': '76ca091f-f936-486b-8874-6d66b99d7672',
                                              'name': 'get_exchange_rate',
                                              'type': 'tool_call'}],
                              'type': 'ai',
                              'usage_metadata': {'input_tokens': 144,
                                                 'output_tokens': 14,
                                                 'total_tokens': 158}},
                   'lc': 1,
                   'type': 'constructor'},
                  {'id': ['langchain', 'schema', 'messages', 'ToolMessage'],
                   'kwargs': {'content': '{"amount": 1.0, "base": "USD", "date": '
                                         '"2025-02-18", "rates": {"KRW": 1441.87}}',
                              'id': '1645a0b6-40ce-4968-966c-ec4133864dc2',
                              'name': 'get_exchange_rate',
                              'status': 'success',
                              'tool_call_id': '76ca091f-f936-486b-8874-6d66b99d7672',
                              'type': 'tool'},
                   'lc': 1,
                   'type': 'constructor'},
                  {'id': ['langchain', 'schema', 'messages', 'AIMessage'],
                   'kwargs': {'content': '현재 미국 달러 환율은 1,441.87 한국 원입니다.',
                              'id': 'run-42e47169-1b65-41df-a8c5-c87e96ebb95b-0',
                              'invalid_tool_calls': [],
                              'response_metadata': {'avg_logprobs': -0.059743279502505346,
                                                    'finish_reason': 'STOP',
                                                    'is_blocked': False,
                                                    'safety_ratings': [],
                                                    'usage_metadata': {'cache_tokens_details': [],
                                                                       'cached_content_token_count': 0,
                                                                       'candidates_token_count': 21,
                                                                       'candidates_tokens_details': [{'modality': 1,
                                                                                                      'token_count': 21}],
                                                                       'prompt_token_count': 180,
                                                                       'prompt_tokens_details': [{'modality': 1,
                                                                                                  'token_count': 180}],
                                                                       'total_token_count': 201}},
                              'tool_calls': [],
                              'type': 'ai',
                              'usage_metadata': {'input_tokens': 180,
                                                 'output_tokens': 21,
                                                 'total_tokens': 201}},
                   'lc': 1,
                   'type': 'constructor'}]}
    ******************************


또는 그래프의 `updates`를 스트리밍 방식으로 다시 가져올 수 있습니다. 이는 각 노드가 실행된 후 상태의 변경 사항을 나타냅니다.


```python
import pprint

for state_updates in agent.stream_query(
    input=inputs,
    stream_mode="updates",
    config={"configurable": {"thread_id": "streaming-thread-updates"}},
):
    pprint.pprint(state_updates)
    print("*"*30)    
```

    {'agent': {'messages': [{'id': ['langchain', 'schema', 'messages', 'AIMessage'],
                             'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                             '"USD", '
                                                                                             '"currency_to": '
                                                                                             '"KRW"}',
                                                                                'name': 'get_exchange_rate'}},
                                        'content': '',
                                        'id': 'run-d9ca1686-8292-4bed-8095-9aa0ff442533-0',
                                        'invalid_tool_calls': [],
                                        'response_metadata': {'avg_logprobs': -0.00133492039250476,
                                                              'finish_reason': 'STOP',
                                                              'is_blocked': False,
                                                              'safety_ratings': [],
                                                              'usage_metadata': {'cache_tokens_details': [],
                                                                                 'cached_content_token_count': 0,
                                                                                 'candidates_token_count': 14,
                                                                                 'candidates_tokens_details': [{'modality': 1,
                                                                                                                'token_count': 14}],
                                                                                 'prompt_token_count': 144,
                                                                                 'prompt_tokens_details': [{'modality': 1,
                                                                                                            'token_count': 144}],
                                                                                 'total_token_count': 158}},
                                        'tool_calls': [{'args': {'currency_from': 'USD',
                                                                 'currency_to': 'KRW'},
                                                        'id': '220fd5f1-8add-4e88-8f04-e6611f399738',
                                                        'name': 'get_exchange_rate',
                                                        'type': 'tool_call'}],
                                        'type': 'ai',
                                        'usage_metadata': {'input_tokens': 144,
                                                           'output_tokens': 14,
                                                           'total_tokens': 158}},
                             'lc': 1,
                             'type': 'constructor'}]}}
    ******************************
    {'tools': {'messages': [{'id': ['langchain',
                                    'schema',
                                    'messages',
                                    'ToolMessage'],
                             'kwargs': {'content': '{"amount": 1.0, "base": "USD", '
                                                   '"date": "2025-02-18", "rates": '
                                                   '{"KRW": 1441.87}}',
                                        'id': '8362f513-163c-41d1-a1be-1335e97c6561',
                                        'name': 'get_exchange_rate',
                                        'status': 'success',
                                        'tool_call_id': '220fd5f1-8add-4e88-8f04-e6611f399738',
                                        'type': 'tool'},
                             'lc': 1,
                             'type': 'constructor'}]}}
    ******************************
    {'agent': {'messages': [{'id': ['langchain', 'schema', 'messages', 'AIMessage'],
                             'kwargs': {'content': '현재 미국 달러 환율은 1,441.87 한국 원입니다.',
                                        'id': 'run-bac8c742-69c6-4e29-a9e1-8ca2260ec6f7-0',
                                        'invalid_tool_calls': [],
                                        'response_metadata': {'avg_logprobs': -0.059743279502505346,
                                                              'finish_reason': 'STOP',
                                                              'is_blocked': False,
                                                              'safety_ratings': [],
                                                              'usage_metadata': {'cache_tokens_details': [],
                                                                                 'cached_content_token_count': 0,
                                                                                 'candidates_token_count': 21,
                                                                                 'candidates_tokens_details': [{'modality': 1,
                                                                                                                'token_count': 21}],
                                                                                 'prompt_token_count': 180,
                                                                                 'prompt_tokens_details': [{'modality': 1,
                                                                                                            'token_count': 180}],
                                                                                 'total_token_count': 201}},
                                        'tool_calls': [],
                                        'type': 'ai',
                                        'usage_metadata': {'input_tokens': 180,
                                                           'output_tokens': 21,
                                                           'total_tokens': 201}},
                             'lc': 1,
                             'type': 'constructor'}]}}
    ******************************


## 디버그 모드로 조회하기


```python
import pprint

for state_updates in agent.stream_query(
    input=inputs,
    stream_mode="debug",
    config={"configurable": {"thread_id": "streaming-thread-updates-1"}},
):
    pprint.pprint(state_updates)
    print("*"*30)    
```

    {'payload': {'config': {'callbacks': None,
                            'configurable': {'checkpoint_id': '1efee983-2eeb-6e0c-bfff-e63e282df7c7',
                                             'checkpoint_ns': '',
                                             'thread_id': 'streaming-thread-updates-1'},
                            'metadata': {'id': ['collections', 'ChainMap'],
                                         'lc': 1,
                                         'repr': "ChainMap({'thread_id': "
                                                 "'streaming-thread-updates-1'})",
                                         'type': 'not_implemented'},
                            'recursion_limit': 25,
                            'tags': []},
                 'metadata': {'parents': {},
                              'source': 'input',
                              'step': -1,
                              'thread_id': 'streaming-thread-updates-1',
                              'writes': {'__start__': {'messages': [['user',
                                                                     '미국 달러에서 한국 '
                                                                     '통화로의 환율은 '
                                                                     '얼마인가요?']]}}},
                 'next': ['__start__'],
                 'parent_config': None,
                 'tasks': [{'id': 'b90f551d-d417-3d21-b572-f15364b5863d',
                            'interrupts': [],
                            'name': '__start__',
                            'state': None}],
                 'values': {'messages': []}},
     'step': -1,
     'timestamp': '2025-02-19T08:04:52.872139+00:00',
     'type': 'checkpoint'}
    ******************************
    {'payload': {'config': {'callbacks': None,
                            'configurable': {'checkpoint_id': '1efee983-384c-6d98-8000-6cd06711dedc',
                                             'checkpoint_ns': '',
                                             'thread_id': 'streaming-thread-updates-1'},
                            'metadata': {'id': ['collections', 'ChainMap'],
                                         'lc': 1,
                                         'repr': "ChainMap({'thread_id': "
                                                 "'streaming-thread-updates-1'})",
                                         'type': 'not_implemented'},
                            'recursion_limit': 25,
                            'tags': []},
                 'metadata': {'parents': {},
                              'source': 'loop',
                              'step': 0,
                              'thread_id': 'streaming-thread-updates-1',
                              'writes': None},
                 'next': ['agent'],
                 'parent_config': {'callbacks': None,
                                   'configurable': {'checkpoint_id': '1efee983-2eeb-6e0c-bfff-e63e282df7c7',
                                                    'checkpoint_ns': '',
                                                    'thread_id': 'streaming-thread-updates-1'},
                                   'metadata': {'id': ['collections', 'ChainMap'],
                                                'lc': 1,
                                                'repr': "ChainMap({'thread_id': "
                                                        "'streaming-thread-updates-1'})",
                                                'type': 'not_implemented'},
                                   'recursion_limit': 25,
                                   'tags': []},
                 'tasks': [{'id': 'eab75cde-b095-4de7-6d31-f6afd0d97031',
                            'interrupts': [],
                            'name': 'agent',
                            'state': None}],
                 'values': {'messages': [{'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'HumanMessage'],
                                          'kwargs': {'content': '미국 달러에서 한국 통화로의 '
                                                                '환율은 얼마인가요?',
                                                     'id': '674ab326-0d2b-4921-950d-cebd5b2f5cc9',
                                                     'type': 'human'},
                                          'lc': 1,
                                          'type': 'constructor'}]}},
     'step': 0,
     'timestamp': '2025-02-19T08:04:53.855567+00:00',
     'type': 'checkpoint'}
    ******************************
    {'payload': {'id': 'eab75cde-b095-4de7-6d31-f6afd0d97031',
                 'input': {'is_last_step': False,
                           'messages': [{'id': ['langchain',
                                                'schema',
                                                'messages',
                                                'HumanMessage'],
                                         'kwargs': {'content': '미국 달러에서 한국 통화로의 '
                                                               '환율은 얼마인가요?',
                                                    'id': '674ab326-0d2b-4921-950d-cebd5b2f5cc9',
                                                    'type': 'human'},
                                         'lc': 1,
                                         'type': 'constructor'}],
                           'remaining_steps': 24},
                 'name': 'agent',
                 'triggers': ['start:agent']},
     'step': 1,
     'timestamp': '2025-02-19T08:04:53.856489+00:00',
     'type': 'task'}
    ******************************
    {'payload': {'error': None,
                 'id': 'eab75cde-b095-4de7-6d31-f6afd0d97031',
                 'interrupts': [],
                 'name': 'agent',
                 'result': [['messages',
                             [{'id': ['langchain',
                                      'schema',
                                      'messages',
                                      'AIMessage'],
                               'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                               '"USD", '
                                                                                               '"currency_to": '
                                                                                               '"KRW"}',
                                                                                  'name': 'get_exchange_rate'}},
                                          'content': '',
                                          'id': 'run-7e05efec-d382-43fd-a73b-1f880e2d8243-0',
                                          'invalid_tool_calls': [],
                                          'response_metadata': {'avg_logprobs': -0.00133492039250476,
                                                                'finish_reason': 'STOP',
                                                                'is_blocked': False,
                                                                'safety_ratings': [],
                                                                'usage_metadata': {'cache_tokens_details': [],
                                                                                   'cached_content_token_count': 0,
                                                                                   'candidates_token_count': 14,
                                                                                   'candidates_tokens_details': [{'modality': 1,
                                                                                                                  'token_count': 14}],
                                                                                   'prompt_token_count': 144,
                                                                                   'prompt_tokens_details': [{'modality': 1,
                                                                                                              'token_count': 144}],
                                                                                   'total_token_count': 158}},
                                          'tool_calls': [{'args': {'currency_from': 'USD',
                                                                   'currency_to': 'KRW'},
                                                          'id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                          'name': 'get_exchange_rate',
                                                          'type': 'tool_call'}],
                                          'type': 'ai',
                                          'usage_metadata': {'input_tokens': 144,
                                                             'output_tokens': 14,
                                                             'total_tokens': 158}},
                               'lc': 1,
                               'type': 'constructor'}]]]},
     'step': 1,
     'timestamp': '2025-02-19T08:04:57.155056+00:00',
     'type': 'task_result'}
    ******************************
    {'payload': {'config': {'callbacks': None,
                            'configurable': {'checkpoint_id': '1efee983-57c4-67fc-8001-7289089e9f0c',
                                             'checkpoint_ns': '',
                                             'thread_id': 'streaming-thread-updates-1'},
                            'metadata': {'id': ['collections', 'ChainMap'],
                                         'lc': 1,
                                         'repr': "ChainMap({'thread_id': "
                                                 "'streaming-thread-updates-1'})",
                                         'type': 'not_implemented'},
                            'recursion_limit': 25,
                            'tags': []},
                 'metadata': {'parents': {},
                              'source': 'loop',
                              'step': 1,
                              'thread_id': 'streaming-thread-updates-1',
                              'writes': {'agent': {'messages': [{'id': ['langchain',
                                                                        'schema',
                                                                        'messages',
                                                                        'AIMessage'],
                                                                 'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                                                                 '"USD", '
                                                                                                                                 '"currency_to": '
                                                                                                                                 '"KRW"}',
                                                                                                                    'name': 'get_exchange_rate'}},
                                                                            'content': '',
                                                                            'id': 'run-7e05efec-d382-43fd-a73b-1f880e2d8243-0',
                                                                            'invalid_tool_calls': [],
                                                                            'response_metadata': {'avg_logprobs': -0.00133492039250476,
                                                                                                  'finish_reason': 'STOP',
                                                                                                  'is_blocked': False,
                                                                                                  'safety_ratings': [],
                                                                                                  'usage_metadata': {'cache_tokens_details': [],
                                                                                                                     'cached_content_token_count': 0,
                                                                                                                     'candidates_token_count': 14,
                                                                                                                     'candidates_tokens_details': [{'modality': 1,
                                                                                                                                                    'token_count': 14}],
                                                                                                                     'prompt_token_count': 144,
                                                                                                                     'prompt_tokens_details': [{'modality': 1,
                                                                                                                                                'token_count': 144}],
                                                                                                                     'total_token_count': 158}},
                                                                            'tool_calls': [{'args': {'currency_from': 'USD',
                                                                                                     'currency_to': 'KRW'},
                                                                                            'id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                                                            'name': 'get_exchange_rate',
                                                                                            'type': 'tool_call'}],
                                                                            'type': 'ai',
                                                                            'usage_metadata': {'input_tokens': 144,
                                                                                               'output_tokens': 14,
                                                                                               'total_tokens': 158}},
                                                                 'lc': 1,
                                                                 'type': 'constructor'}]}}},
                 'next': ['tools'],
                 'parent_config': {'callbacks': None,
                                   'configurable': {'checkpoint_id': '1efee983-384c-6d98-8000-6cd06711dedc',
                                                    'checkpoint_ns': '',
                                                    'thread_id': 'streaming-thread-updates-1'},
                                   'metadata': {'id': ['collections', 'ChainMap'],
                                                'lc': 1,
                                                'repr': "ChainMap({'thread_id': "
                                                        "'streaming-thread-updates-1'})",
                                                'type': 'not_implemented'},
                                   'recursion_limit': 25,
                                   'tags': []},
                 'tasks': [{'id': 'b9f1b0c7-0f3d-8062-c604-ab5c2ef83de0',
                            'interrupts': [],
                            'name': 'tools',
                            'state': None}],
                 'values': {'messages': [{'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'HumanMessage'],
                                          'kwargs': {'content': '미국 달러에서 한국 통화로의 '
                                                                '환율은 얼마인가요?',
                                                     'id': '674ab326-0d2b-4921-950d-cebd5b2f5cc9',
                                                     'type': 'human'},
                                          'lc': 1,
                                          'type': 'constructor'},
                                         {'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'AIMessage'],
                                          'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                                          '"USD", '
                                                                                                          '"currency_to": '
                                                                                                          '"KRW"}',
                                                                                             'name': 'get_exchange_rate'}},
                                                     'content': '',
                                                     'id': 'run-7e05efec-d382-43fd-a73b-1f880e2d8243-0',
                                                     'invalid_tool_calls': [],
                                                     'response_metadata': {'avg_logprobs': -0.00133492039250476,
                                                                           'finish_reason': 'STOP',
                                                                           'is_blocked': False,
                                                                           'safety_ratings': [],
                                                                           'usage_metadata': {'cache_tokens_details': [],
                                                                                              'cached_content_token_count': 0,
                                                                                              'candidates_token_count': 14,
                                                                                              'candidates_tokens_details': [{'modality': 1,
                                                                                                                             'token_count': 14}],
                                                                                              'prompt_token_count': 144,
                                                                                              'prompt_tokens_details': [{'modality': 1,
                                                                                                                         'token_count': 144}],
                                                                                              'total_token_count': 158}},
                                                     'tool_calls': [{'args': {'currency_from': 'USD',
                                                                              'currency_to': 'KRW'},
                                                                     'id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                                     'name': 'get_exchange_rate',
                                                                     'type': 'tool_call'}],
                                                     'type': 'ai',
                                                     'usage_metadata': {'input_tokens': 144,
                                                                        'output_tokens': 14,
                                                                        'total_tokens': 158}},
                                          'lc': 1,
                                          'type': 'constructor'}]}},
     'step': 1,
     'timestamp': '2025-02-19T08:04:57.155164+00:00',
     'type': 'checkpoint'}
    ******************************
    {'payload': {'id': 'b9f1b0c7-0f3d-8062-c604-ab5c2ef83de0',
                 'input': {'is_last_step': False,
                           'messages': [{'id': ['langchain',
                                                'schema',
                                                'messages',
                                                'HumanMessage'],
                                         'kwargs': {'content': '미국 달러에서 한국 통화로의 '
                                                               '환율은 얼마인가요?',
                                                    'id': '674ab326-0d2b-4921-950d-cebd5b2f5cc9',
                                                    'type': 'human'},
                                         'lc': 1,
                                         'type': 'constructor'},
                                        {'id': ['langchain',
                                                'schema',
                                                'messages',
                                                'AIMessage'],
                                         'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                                         '"USD", '
                                                                                                         '"currency_to": '
                                                                                                         '"KRW"}',
                                                                                            'name': 'get_exchange_rate'}},
                                                    'content': '',
                                                    'id': 'run-7e05efec-d382-43fd-a73b-1f880e2d8243-0',
                                                    'invalid_tool_calls': [],
                                                    'response_metadata': {'avg_logprobs': -0.00133492039250476,
                                                                          'finish_reason': 'STOP',
                                                                          'is_blocked': False,
                                                                          'safety_ratings': [],
                                                                          'usage_metadata': {'cache_tokens_details': [],
                                                                                             'cached_content_token_count': 0,
                                                                                             'candidates_token_count': 14,
                                                                                             'candidates_tokens_details': [{'modality': 1,
                                                                                                                            'token_count': 14}],
                                                                                             'prompt_token_count': 144,
                                                                                             'prompt_tokens_details': [{'modality': 1,
                                                                                                                        'token_count': 144}],
                                                                                             'total_token_count': 158}},
                                                    'tool_calls': [{'args': {'currency_from': 'USD',
                                                                             'currency_to': 'KRW'},
                                                                    'id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                                    'name': 'get_exchange_rate',
                                                                    'type': 'tool_call'}],
                                                    'type': 'ai',
                                                    'usage_metadata': {'input_tokens': 144,
                                                                       'output_tokens': 14,
                                                                       'total_tokens': 158}},
                                         'lc': 1,
                                         'type': 'constructor'}],
                           'remaining_steps': 23},
                 'name': 'tools',
                 'triggers': ['branch:agent:should_continue:tools']},
     'step': 2,
     'timestamp': '2025-02-19T08:04:57.155301+00:00',
     'type': 'task'}
    ******************************
    {'payload': {'error': None,
                 'id': 'b9f1b0c7-0f3d-8062-c604-ab5c2ef83de0',
                 'interrupts': [],
                 'name': 'tools',
                 'result': [['messages',
                             [{'id': ['langchain',
                                      'schema',
                                      'messages',
                                      'ToolMessage'],
                               'kwargs': {'content': '{"amount": 1.0, "base": '
                                                     '"USD", "date": "2025-02-18", '
                                                     '"rates": {"KRW": 1441.87}}',
                                          'id': 'b99eba67-2a1b-415f-8ab2-87eca61da0fb',
                                          'name': 'get_exchange_rate',
                                          'status': 'success',
                                          'tool_call_id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                          'type': 'tool'},
                               'lc': 1,
                               'type': 'constructor'}]]]},
     'step': 2,
     'timestamp': '2025-02-19T08:04:59.032698+00:00',
     'type': 'task_result'}
    ******************************
    {'payload': {'config': {'callbacks': None,
                            'configurable': {'checkpoint_id': '1efee983-69ac-68d4-8002-333069251d49',
                                             'checkpoint_ns': '',
                                             'thread_id': 'streaming-thread-updates-1'},
                            'metadata': {'id': ['collections', 'ChainMap'],
                                         'lc': 1,
                                         'repr': "ChainMap({'thread_id': "
                                                 "'streaming-thread-updates-1'})",
                                         'type': 'not_implemented'},
                            'recursion_limit': 25,
                            'tags': []},
                 'metadata': {'parents': {},
                              'source': 'loop',
                              'step': 2,
                              'thread_id': 'streaming-thread-updates-1',
                              'writes': {'tools': {'messages': [{'id': ['langchain',
                                                                        'schema',
                                                                        'messages',
                                                                        'ToolMessage'],
                                                                 'kwargs': {'content': '{"amount": '
                                                                                       '1.0, '
                                                                                       '"base": '
                                                                                       '"USD", '
                                                                                       '"date": '
                                                                                       '"2025-02-18", '
                                                                                       '"rates": '
                                                                                       '{"KRW": '
                                                                                       '1441.87}}',
                                                                            'id': 'b99eba67-2a1b-415f-8ab2-87eca61da0fb',
                                                                            'name': 'get_exchange_rate',
                                                                            'status': 'success',
                                                                            'tool_call_id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                                            'type': 'tool'},
                                                                 'lc': 1,
                                                                 'type': 'constructor'}]}}},
                 'next': ['agent'],
                 'parent_config': {'callbacks': None,
                                   'configurable': {'checkpoint_id': '1efee983-57c4-67fc-8001-7289089e9f0c',
                                                    'checkpoint_ns': '',
                                                    'thread_id': 'streaming-thread-updates-1'},
                                   'metadata': {'id': ['collections', 'ChainMap'],
                                                'lc': 1,
                                                'repr': "ChainMap({'thread_id': "
                                                        "'streaming-thread-updates-1'})",
                                                'type': 'not_implemented'},
                                   'recursion_limit': 25,
                                   'tags': []},
                 'tasks': [{'id': '6903fb95-79c7-5b90-feee-9b99e6ed9956',
                            'interrupts': [],
                            'name': 'agent',
                            'state': None}],
                 'values': {'messages': [{'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'HumanMessage'],
                                          'kwargs': {'content': '미국 달러에서 한국 통화로의 '
                                                                '환율은 얼마인가요?',
                                                     'id': '674ab326-0d2b-4921-950d-cebd5b2f5cc9',
                                                     'type': 'human'},
                                          'lc': 1,
                                          'type': 'constructor'},
                                         {'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'AIMessage'],
                                          'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                                          '"USD", '
                                                                                                          '"currency_to": '
                                                                                                          '"KRW"}',
                                                                                             'name': 'get_exchange_rate'}},
                                                     'content': '',
                                                     'id': 'run-7e05efec-d382-43fd-a73b-1f880e2d8243-0',
                                                     'invalid_tool_calls': [],
                                                     'response_metadata': {'avg_logprobs': -0.00133492039250476,
                                                                           'finish_reason': 'STOP',
                                                                           'is_blocked': False,
                                                                           'safety_ratings': [],
                                                                           'usage_metadata': {'cache_tokens_details': [],
                                                                                              'cached_content_token_count': 0,
                                                                                              'candidates_token_count': 14,
                                                                                              'candidates_tokens_details': [{'modality': 1,
                                                                                                                             'token_count': 14}],
                                                                                              'prompt_token_count': 144,
                                                                                              'prompt_tokens_details': [{'modality': 1,
                                                                                                                         'token_count': 144}],
                                                                                              'total_token_count': 158}},
                                                     'tool_calls': [{'args': {'currency_from': 'USD',
                                                                              'currency_to': 'KRW'},
                                                                     'id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                                     'name': 'get_exchange_rate',
                                                                     'type': 'tool_call'}],
                                                     'type': 'ai',
                                                     'usage_metadata': {'input_tokens': 144,
                                                                        'output_tokens': 14,
                                                                        'total_tokens': 158}},
                                          'lc': 1,
                                          'type': 'constructor'},
                                         {'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'ToolMessage'],
                                          'kwargs': {'content': '{"amount": 1.0, '
                                                                '"base": "USD", '
                                                                '"date": '
                                                                '"2025-02-18", '
                                                                '"rates": {"KRW": '
                                                                '1441.87}}',
                                                     'id': 'b99eba67-2a1b-415f-8ab2-87eca61da0fb',
                                                     'name': 'get_exchange_rate',
                                                     'status': 'success',
                                                     'tool_call_id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                     'type': 'tool'},
                                          'lc': 1,
                                          'type': 'constructor'}]}},
     'step': 2,
     'timestamp': '2025-02-19T08:04:59.032793+00:00',
     'type': 'checkpoint'}
    ******************************
    {'payload': {'id': '6903fb95-79c7-5b90-feee-9b99e6ed9956',
                 'input': {'is_last_step': False,
                           'messages': [{'id': ['langchain',
                                                'schema',
                                                'messages',
                                                'HumanMessage'],
                                         'kwargs': {'content': '미국 달러에서 한국 통화로의 '
                                                               '환율은 얼마인가요?',
                                                    'id': '674ab326-0d2b-4921-950d-cebd5b2f5cc9',
                                                    'type': 'human'},
                                         'lc': 1,
                                         'type': 'constructor'},
                                        {'id': ['langchain',
                                                'schema',
                                                'messages',
                                                'AIMessage'],
                                         'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                                         '"USD", '
                                                                                                         '"currency_to": '
                                                                                                         '"KRW"}',
                                                                                            'name': 'get_exchange_rate'}},
                                                    'content': '',
                                                    'id': 'run-7e05efec-d382-43fd-a73b-1f880e2d8243-0',
                                                    'invalid_tool_calls': [],
                                                    'response_metadata': {'avg_logprobs': -0.00133492039250476,
                                                                          'finish_reason': 'STOP',
                                                                          'is_blocked': False,
                                                                          'safety_ratings': [],
                                                                          'usage_metadata': {'cache_tokens_details': [],
                                                                                             'cached_content_token_count': 0,
                                                                                             'candidates_token_count': 14,
                                                                                             'candidates_tokens_details': [{'modality': 1,
                                                                                                                            'token_count': 14}],
                                                                                             'prompt_token_count': 144,
                                                                                             'prompt_tokens_details': [{'modality': 1,
                                                                                                                        'token_count': 144}],
                                                                                             'total_token_count': 158}},
                                                    'tool_calls': [{'args': {'currency_from': 'USD',
                                                                             'currency_to': 'KRW'},
                                                                    'id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                                    'name': 'get_exchange_rate',
                                                                    'type': 'tool_call'}],
                                                    'type': 'ai',
                                                    'usage_metadata': {'input_tokens': 144,
                                                                       'output_tokens': 14,
                                                                       'total_tokens': 158}},
                                         'lc': 1,
                                         'type': 'constructor'},
                                        {'id': ['langchain',
                                                'schema',
                                                'messages',
                                                'ToolMessage'],
                                         'kwargs': {'content': '{"amount": 1.0, '
                                                               '"base": "USD", '
                                                               '"date": '
                                                               '"2025-02-18", '
                                                               '"rates": {"KRW": '
                                                               '1441.87}}',
                                                    'id': 'b99eba67-2a1b-415f-8ab2-87eca61da0fb',
                                                    'name': 'get_exchange_rate',
                                                    'status': 'success',
                                                    'tool_call_id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                    'type': 'tool'},
                                         'lc': 1,
                                         'type': 'constructor'}],
                           'remaining_steps': 22},
                 'name': 'agent',
                 'triggers': ['tools']},
     'step': 3,
     'timestamp': '2025-02-19T08:04:59.032908+00:00',
     'type': 'task'}
    ******************************
    {'payload': {'error': None,
                 'id': '6903fb95-79c7-5b90-feee-9b99e6ed9956',
                 'interrupts': [],
                 'name': 'agent',
                 'result': [['messages',
                             [{'id': ['langchain',
                                      'schema',
                                      'messages',
                                      'AIMessage'],
                               'kwargs': {'content': '현재 미국 달러에서 한국 통화로의 환율은 '
                                                     '1441.87입니다. 즉, 1 미국 달러는 '
                                                     '1441.87 한국 원입니다.',
                                          'id': 'run-274f149d-3463-469c-8087-eba492206019-0',
                                          'invalid_tool_calls': [],
                                          'response_metadata': {'avg_logprobs': -0.048041518529256186,
                                                                'finish_reason': 'STOP',
                                                                'is_blocked': False,
                                                                'safety_ratings': [],
                                                                'usage_metadata': {'cache_tokens_details': [],
                                                                                   'cached_content_token_count': 0,
                                                                                   'candidates_token_count': 45,
                                                                                   'candidates_tokens_details': [{'modality': 1,
                                                                                                                  'token_count': 45}],
                                                                                   'prompt_token_count': 180,
                                                                                   'prompt_tokens_details': [{'modality': 1,
                                                                                                              'token_count': 180}],
                                                                                   'total_token_count': 225}},
                                          'tool_calls': [],
                                          'type': 'ai',
                                          'usage_metadata': {'input_tokens': 180,
                                                             'output_tokens': 45,
                                                             'total_tokens': 225}},
                               'lc': 1,
                               'type': 'constructor'}]]]},
     'step': 3,
     'timestamp': '2025-02-19T08:05:01.944166+00:00',
     'type': 'task_result'}
    ******************************
    {'payload': {'config': {'callbacks': None,
                            'configurable': {'checkpoint_id': '1efee983-8570-69e4-8003-9fa66b3396d8',
                                             'checkpoint_ns': '',
                                             'thread_id': 'streaming-thread-updates-1'},
                            'metadata': {'id': ['collections', 'ChainMap'],
                                         'lc': 1,
                                         'repr': "ChainMap({'thread_id': "
                                                 "'streaming-thread-updates-1'})",
                                         'type': 'not_implemented'},
                            'recursion_limit': 25,
                            'tags': []},
                 'metadata': {'parents': {},
                              'source': 'loop',
                              'step': 3,
                              'thread_id': 'streaming-thread-updates-1',
                              'writes': {'agent': {'messages': [{'id': ['langchain',
                                                                        'schema',
                                                                        'messages',
                                                                        'AIMessage'],
                                                                 'kwargs': {'content': '현재 '
                                                                                       '미국 '
                                                                                       '달러에서 '
                                                                                       '한국 '
                                                                                       '통화로의 '
                                                                                       '환율은 '
                                                                                       '1441.87입니다. '
                                                                                       '즉, '
                                                                                       '1 '
                                                                                       '미국 '
                                                                                       '달러는 '
                                                                                       '1441.87 '
                                                                                       '한국 '
                                                                                       '원입니다.',
                                                                            'id': 'run-274f149d-3463-469c-8087-eba492206019-0',
                                                                            'invalid_tool_calls': [],
                                                                            'response_metadata': {'avg_logprobs': -0.048041518529256186,
                                                                                                  'finish_reason': 'STOP',
                                                                                                  'is_blocked': False,
                                                                                                  'safety_ratings': [],
                                                                                                  'usage_metadata': {'cache_tokens_details': [],
                                                                                                                     'cached_content_token_count': 0,
                                                                                                                     'candidates_token_count': 45,
                                                                                                                     'candidates_tokens_details': [{'modality': 1,
                                                                                                                                                    'token_count': 45}],
                                                                                                                     'prompt_token_count': 180,
                                                                                                                     'prompt_tokens_details': [{'modality': 1,
                                                                                                                                                'token_count': 180}],
                                                                                                                     'total_token_count': 225}},
                                                                            'tool_calls': [],
                                                                            'type': 'ai',
                                                                            'usage_metadata': {'input_tokens': 180,
                                                                                               'output_tokens': 45,
                                                                                               'total_tokens': 225}},
                                                                 'lc': 1,
                                                                 'type': 'constructor'}]}}},
                 'next': [],
                 'parent_config': {'callbacks': None,
                                   'configurable': {'checkpoint_id': '1efee983-69ac-68d4-8002-333069251d49',
                                                    'checkpoint_ns': '',
                                                    'thread_id': 'streaming-thread-updates-1'},
                                   'metadata': {'id': ['collections', 'ChainMap'],
                                                'lc': 1,
                                                'repr': "ChainMap({'thread_id': "
                                                        "'streaming-thread-updates-1'})",
                                                'type': 'not_implemented'},
                                   'recursion_limit': 25,
                                   'tags': []},
                 'tasks': [],
                 'values': {'messages': [{'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'HumanMessage'],
                                          'kwargs': {'content': '미국 달러에서 한국 통화로의 '
                                                                '환율은 얼마인가요?',
                                                     'id': '674ab326-0d2b-4921-950d-cebd5b2f5cc9',
                                                     'type': 'human'},
                                          'lc': 1,
                                          'type': 'constructor'},
                                         {'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'AIMessage'],
                                          'kwargs': {'additional_kwargs': {'function_call': {'arguments': '{"currency_from": '
                                                                                                          '"USD", '
                                                                                                          '"currency_to": '
                                                                                                          '"KRW"}',
                                                                                             'name': 'get_exchange_rate'}},
                                                     'content': '',
                                                     'id': 'run-7e05efec-d382-43fd-a73b-1f880e2d8243-0',
                                                     'invalid_tool_calls': [],
                                                     'response_metadata': {'avg_logprobs': -0.00133492039250476,
                                                                           'finish_reason': 'STOP',
                                                                           'is_blocked': False,
                                                                           'safety_ratings': [],
                                                                           'usage_metadata': {'cache_tokens_details': [],
                                                                                              'cached_content_token_count': 0,
                                                                                              'candidates_token_count': 14,
                                                                                              'candidates_tokens_details': [{'modality': 1,
                                                                                                                             'token_count': 14}],
                                                                                              'prompt_token_count': 144,
                                                                                              'prompt_tokens_details': [{'modality': 1,
                                                                                                                         'token_count': 144}],
                                                                                              'total_token_count': 158}},
                                                     'tool_calls': [{'args': {'currency_from': 'USD',
                                                                              'currency_to': 'KRW'},
                                                                     'id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                                     'name': 'get_exchange_rate',
                                                                     'type': 'tool_call'}],
                                                     'type': 'ai',
                                                     'usage_metadata': {'input_tokens': 144,
                                                                        'output_tokens': 14,
                                                                        'total_tokens': 158}},
                                          'lc': 1,
                                          'type': 'constructor'},
                                         {'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'ToolMessage'],
                                          'kwargs': {'content': '{"amount": 1.0, '
                                                                '"base": "USD", '
                                                                '"date": '
                                                                '"2025-02-18", '
                                                                '"rates": {"KRW": '
                                                                '1441.87}}',
                                                     'id': 'b99eba67-2a1b-415f-8ab2-87eca61da0fb',
                                                     'name': 'get_exchange_rate',
                                                     'status': 'success',
                                                     'tool_call_id': '0281375f-df9d-4ac3-b4dc-26271685a0e4',
                                                     'type': 'tool'},
                                          'lc': 1,
                                          'type': 'constructor'},
                                         {'id': ['langchain',
                                                 'schema',
                                                 'messages',
                                                 'AIMessage'],
                                          'kwargs': {'content': '현재 미국 달러에서 한국 '
                                                                '통화로의 환율은 '
                                                                '1441.87입니다. 즉, 1 '
                                                                '미국 달러는 1441.87 한국 '
                                                                '원입니다.',
                                                     'id': 'run-274f149d-3463-469c-8087-eba492206019-0',
                                                     'invalid_tool_calls': [],
                                                     'response_metadata': {'avg_logprobs': -0.048041518529256186,
                                                                           'finish_reason': 'STOP',
                                                                           'is_blocked': False,
                                                                           'safety_ratings': [],
                                                                           'usage_metadata': {'cache_tokens_details': [],
                                                                                              'cached_content_token_count': 0,
                                                                                              'candidates_token_count': 45,
                                                                                              'candidates_tokens_details': [{'modality': 1,
                                                                                                                             'token_count': 45}],
                                                                                              'prompt_token_count': 180,
                                                                                              'prompt_tokens_details': [{'modality': 1,
                                                                                                                         'token_count': 180}],
                                                                                              'total_token_count': 225}},
                                                     'tool_calls': [],
                                                     'type': 'ai',
                                                     'usage_metadata': {'input_tokens': 180,
                                                                        'output_tokens': 45,
                                                                        'total_tokens': 225}},
                                          'lc': 1,
                                          'type': 'constructor'}]}},
     'step': 3,
     'timestamp': '2025-02-19T08:05:01.944256+00:00',
     'type': 'checkpoint'}
    ******************************


## Human-in-the-loop

### 도구 호출 검토 (Reviewing Tool Calls)

LangGraph의 Human-in-the-Loop 기능은 에이전트 워크플로우(상태 머신 (state machines))에 인간의 개입 및 감독을 통합하기 위한 다양한 [사용 사례 (use cases)](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/#use-cases)를 제공합니다. 이 노트북은 도구 호출 검토 (Reviewing Tool Calls) 사용 사례에 중점을 둡니다.

<mark style='background-color:#B18904'>이를 달성하기 위해 에이전트는 다음 시나리오에서 실행을 [interrupt](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/#interrupt) (중단)해야 합니다.</mark>

* 도구를 호출하기 전  (LLM이 도구 호출 AI Message를 생성할 때).
* 도구 응답을 받은 후.

<details>
    <summary>langchain_load</summary>
langchain_load는 LangChain 객체를 로드하는 데 사용되는 함수입니다. 이 함수는 직렬화된 LangChain 객체(예: JSON 또는 YAML 파일)를 Python 객체로 다시 변환합니다. 이를 통해 저장된 체인, 에이전트 또는 기타 LangChain 구성 요소를 다시 사용할 수 있습니다.
</details>


```python
inputs
```




    {'messages': [('user', '미국 달러에서 한국 통화로의 환율은 얼마인가요?')]}




```python
response = agent.query(
    input=inputs,
    interrupt_before=["tools"],  # Before invoking the tool.
    interrupt_after=["tools"],  # After getting a tool message.
    config={"configurable": {"thread_id": "human-in-the-loop-deepdive"}},
)
langchain_load(response["messages"][-1]).pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_exchange_rate (c41411f8-ee62-4f97-807d-ac325e06471e)
     Call ID: c41411f8-ee62-4f97-807d-ac325e06471e
      Args:
        currency_from: USD
        currency_to: KRW


    /var/folders/g4/xm7xh85s4jd3swbhwyd0q0bh0000gn/T/ipykernel_5462/1559279857.py:7: LangChainBetaWarning: The function `load` is in beta. It is actively being worked on, so the API may change.
      langchain_load(response["messages"][-1]).pretty_print()


프로세스가 *도구를 호출하기 전 (before invoking the tool)*에 중단되었습니다.

검토 후, LLM이 생성한 도구 호출 (`AI Message`)이 올바르다고 가정하고 실행을 재개합니다.


```python
response = agent.query(
    input=None,  # Resume (continue with the tool call AI Message).
    interrupt_before=["tools"],
    interrupt_after=["tools"],
    config={"configurable": {"thread_id": "human-in-the-loop-deepdive"}},
)
langchain_load(response["messages"][-1]).pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    현재 미국 달러에서 한국 통화로의 환율은 1441.87입니다. 즉, 1 미국 달러는 1441.87 한국 원입니다.


프로세스가 *도구 메시지를 받은 후 (after receiving the tool message)* 다시 중단되었습니다.

검토 후, LLM이 생성한 `Tool Message`가 올바르다고 판단되면 실행을 재개할 수 있습니다.


```python
response = agent.query(
    input=None,  # Resume (continue with the Tool Message).
    interrupt_before=["tools"],
    interrupt_after=["tools"],
    config={"configurable": {"thread_id": "human-in-the-loop-deepdive"}},
)
langchain_load(response["messages"][-1]).pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    
    현재 미국 달러 환율은 1,441.87 한국 원입니다.


### 상태 기록 가져오기 (Fetching State History)

`.get_state_history`를 호출하여 상태 기록을 가져올 수 있습니다.


```python
for state_snapshot in agent.get_state_history(
    config={"configurable": {"thread_id": "human-in-the-loop-deepdive"}},
):
    if state_snapshot["metadata"]["step"] >= 0:
        print(f'step {state_snapshot["metadata"]["step"]}: {state_snapshot["config"]}')
        state_snapshot["values"]["messages"][-1].pretty_print()
        print("\n")
```

    step 3: {'configurable': {'thread_id': 'human-in-the-loop-deepdive', 'checkpoint_ns': '', 'checkpoint_id': '1efee6b9-e7e0-6150-8003-d80324748251'}}
    ==================================[1m Ai Message [0m==================================
    
    현재 미국 달러 환율은 1,441.87 한국 원입니다.
    
    
    step 2: {'configurable': {'thread_id': 'human-in-the-loop-deepdive', 'checkpoint_ns': '', 'checkpoint_id': '1efee6b9-0d50-6e0e-8002-eaf7df76dfc1'}}
    =================================[1m Tool Message [0m=================================
    Name: get_exchange_rate
    
    {"amount": 1.0, "base": "USD", "date": "2025-02-18", "rates": {"KRW": 1441.87}}
    
    
    step 1: {'configurable': {'thread_id': 'human-in-the-loop-deepdive', 'checkpoint_ns': '', 'checkpoint_id': '1efee6b4-7a93-64e4-8001-ae8b0443a9a6'}}
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_exchange_rate (55ab34a9-dad9-4e25-ae92-83608b19c70a)
     Call ID: 55ab34a9-dad9-4e25-ae92-83608b19c70a
      Args:
        currency_from: USD
        currency_to: KRW
    
    
    step 0: {'configurable': {'thread_id': 'human-in-the-loop-deepdive', 'checkpoint_ns': '', 'checkpoint_id': '1efee6b4-7423-6f3c-8000-20d519478171'}}
    ================================[1m Human Message [0m=================================
    
    미국 달러에서 한국 통화로의 환율은 얼마인가요?
    
    


### Time Travel

LangGraph의 <mark style='background-color:#B18904'>[Time Travel](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/)은 지속적인 메모리(persistent memory)를 통해 대화형 에이전트(conversational agent)를 구축하고, 사람이 개입하여 과거의 행동을 수정할 수 있도록 하는 방법</mark>을 보여줍니다. 기본적으로, 이는 대화를 이전 상태로 "되감고(rewinds)", 실수를 수정할 수 있도록 하며, 수정된 지점부터 에이전트가 계속 진행할 수 있도록 합니다.

`.get_state`를 호출하여 "시간 여행(time travel)"을 할 수 있습니다. 기본적으로 에이전트는 `latest state(최신 상태)`를 검색합니다.


```python
state = agent.get_state(
    config={
        "configurable": {
            "thread_id": "human-in-the-loop-deepdive",
        }
    }
)

print(f'step {state["metadata"]["step"]}: {state["config"]}')
state["values"]["messages"][-1].pretty_print()
```

    step 3: {'configurable': {'thread_id': 'human-in-the-loop-deepdive', 'checkpoint_ns': '', 'checkpoint_id': '1efee6b9-e7e0-6150-8003-d80324748251'}}
    ==================================[1m Ai Message [0m==================================
    
    현재 미국 달러 환율은 1,441.87 한국 원입니다.


이전 상태를 검색하려면 `checkpoint_id` (및 `checkpoint_ns`)를 지정해야 합니다.


```python
snapshot_config = {}
for state_snapshot in agent.get_state_history(
    config={"configurable": {"thread_id": "human-in-the-loop-deepdive"}},
):
    if state_snapshot["metadata"]["step"] == 1:
        snapshot_config = state_snapshot["config"]
        break

snapshot_config
```




    {'configurable': {'thread_id': 'human-in-the-loop-deepdive',
      'checkpoint_ns': '',
      'checkpoint_id': '1efee6b4-7a93-64e4-8001-ae8b0443a9a6'}}




```python
state = agent.get_state(config=snapshot_config)
print(f'step {state["metadata"]["step"]}: {state["config"]}')
state["values"]["messages"][-1].pretty_print()
```

    step 1: {'configurable': {'thread_id': 'human-in-the-loop-deepdive', 'checkpoint_ns': '', 'checkpoint_id': '1efee6b4-7a93-64e4-8001-ae8b0443a9a6'}}
    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_exchange_rate (55ab34a9-dad9-4e25-ae92-83608b19c70a)
     Call ID: 55ab34a9-dad9-4e25-ae92-83608b19c70a
      Args:
        currency_from: USD
        currency_to: KRW



```python
state
```




    {'values': {'messages': [HumanMessage(content='미국 달러에서 한국 통화로의 환율은 얼마인가요?', additional_kwargs={}, response_metadata={}, id='6332bdd7-e7d3-455a-97fa-1d5e66fedf26'),
       AIMessage(content='', additional_kwargs={'function_call': {'name': 'get_exchange_rate', 'arguments': '{"currency_from": "USD", "currency_to": "KRW"}'}}, response_metadata={'is_blocked': False, 'safety_ratings': [], 'usage_metadata': {'prompt_token_count': 144, 'candidates_token_count': 14, 'total_token_count': 158, 'prompt_tokens_details': [{'modality': 1, 'token_count': 144}], 'candidates_tokens_details': [{'modality': 1, 'token_count': 14}], 'cached_content_token_count': 0, 'cache_tokens_details': []}, 'finish_reason': 'STOP', 'avg_logprobs': -0.00133492039250476}, id='run-8dee9ed3-5631-42f3-8129-ba9a6f759108-0', tool_calls=[{'name': 'get_exchange_rate', 'args': {'currency_from': 'USD', 'currency_to': 'KRW'}, 'id': '55ab34a9-dad9-4e25-ae92-83608b19c70a', 'type': 'tool_call'}], usage_metadata={'input_tokens': 144, 'output_tokens': 14, 'total_tokens': 158})]},
     'next': ('tools',),
     'config': {'configurable': {'thread_id': 'human-in-the-loop-deepdive',
       'checkpoint_ns': '',
       'checkpoint_id': '1efee6b4-7a93-64e4-8001-ae8b0443a9a6'}},
     'metadata': {'source': 'loop',
      'writes': {'agent': {'messages': [AIMessage(content='', additional_kwargs={'function_call': {'name': 'get_exchange_rate', 'arguments': '{"currency_from": "USD", "currency_to": "KRW"}'}}, response_metadata={'is_blocked': False, 'safety_ratings': [], 'usage_metadata': {'prompt_token_count': 144, 'candidates_token_count': 14, 'total_token_count': 158, 'prompt_tokens_details': [{'modality': 1, 'token_count': 144}], 'candidates_tokens_details': [{'modality': 1, 'token_count': 14}], 'cached_content_token_count': 0, 'cache_tokens_details': []}, 'finish_reason': 'STOP', 'avg_logprobs': -0.00133492039250476}, id='run-8dee9ed3-5631-42f3-8129-ba9a6f759108-0', tool_calls=[{'name': 'get_exchange_rate', 'args': {'currency_from': 'USD', 'currency_to': 'KRW'}, 'id': '55ab34a9-dad9-4e25-ae92-83608b19c70a', 'type': 'tool_call'}], usage_metadata={'input_tokens': 144, 'output_tokens': 14, 'total_tokens': 158})]}},
      'thread_id': 'human-in-the-loop-deepdive',
      'step': 1,
      'parents': {}},
     'created_at': '2025-02-19T02:43:20.295745+00:00',
     'parent_config': {'configurable': {'thread_id': 'human-in-the-loop-deepdive',
       'checkpoint_ns': '',
       'checkpoint_id': '1efee6b4-7423-6f3c-8000-20d519478171'}},
     'tasks': (PregelTask(id='3de06a83-5948-c2b1-cf24-40408fa8763a', name='tools', path=('__pregel_pull', 'tools'), error=None, interrupts=(), state=None, result={'messages': [ToolMessage(content='{"amount": 1.0, "base": "USD", "date": "2025-02-18", "rates": {"KRW": 1441.87}}', name='get_exchange_rate', tool_call_id='55ab34a9-dad9-4e25-ae92-83608b19c70a')]}),)}



### Replay (재생)

LangGraph의 [Replay](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/#replay-a-state) 기능은 대화 기록의 특정 시점부터 대화를 재개하거나 다시 재생할 수 있도록 합니다.

`state["config"]`를 에이전트에게 다시 전달하여 재생을 시작할 수 있습니다. 실행은 중단된 지점에서 정확히 재개되어 tool call (도구 호출)을 실행합니다.


```python
state["config"]
```




    {'configurable': {'thread_id': 'human-in-the-loop-deepdive',
      'checkpoint_ns': '',
      'checkpoint_id': '1efee6b4-7a93-64e4-8001-ae8b0443a9a6'}}




```python
for state_values in agent.stream_query(
    input=None,  # resume
    stream_mode="values",
    config=state["config"],
):
    langchain_load(state_values["messages"][-1]).pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_exchange_rate (55ab34a9-dad9-4e25-ae92-83608b19c70a)
     Call ID: 55ab34a9-dad9-4e25-ae92-83608b19c70a
      Args:
        currency_from: USD
        currency_to: KRW
    =================================[1m Tool Message [0m=================================
    Name: get_exchange_rate
    
    {"amount": 1.0, "base": "USD", "date": "2025-02-18", "rates": {"KRW": 1441.87}}
    ==================================[1m Ai Message [0m==================================
    
    현재 미국 달러 환율은 1,441.87 한국 원입니다.


### 분기 (Branching)

LangGraph의 [분기 (Branching)](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/#branch-off-a-past-state) 기능은 기록의 특정 시점부터 LangGraph 대화를 수정하고 다시 실행할 수 있도록 합니다 (최신 상태에서만 다시 시작하는 것이 아니라). 이를 통해 에이전트는 대체 궤적을 탐색하거나 사용자가 워크플로의 변경 사항을 "버전 제어 (version control)"할 수 있습니다.

이 예제에서는 다음을 수행합니다.
* 이전 단계의 도구 호출 (tool calls)을 업데이트합니다.
* `.update_state`를 호출하여 업데이트된 구성으로 단계를 다시 실행합니다.


```python
last_message = state["values"]["messages"][-1]
print(last_message)
print(last_message.tool_calls)
```

    content='' additional_kwargs={'function_call': {'name': 'get_exchange_rate', 'arguments': '{"currency_from": "USD", "currency_to": "KRW"}'}} response_metadata={'is_blocked': False, 'safety_ratings': [], 'usage_metadata': {'prompt_token_count': 144, 'candidates_token_count': 14, 'total_token_count': 158, 'prompt_tokens_details': [{'modality': 1, 'token_count': 144}], 'candidates_tokens_details': [{'modality': 1, 'token_count': 14}], 'cached_content_token_count': 0, 'cache_tokens_details': []}, 'finish_reason': 'STOP', 'avg_logprobs': -0.00133492039250476} id='run-8dee9ed3-5631-42f3-8129-ba9a6f759108-0' tool_calls=[{'name': 'get_exchange_rate', 'args': {'currency_from': 'USD', 'currency_to': 'KRW'}, 'id': '55ab34a9-dad9-4e25-ae92-83608b19c70a', 'type': 'tool_call'}] usage_metadata={'input_tokens': 144, 'output_tokens': 14, 'total_tokens': 158}
    [{'name': 'get_exchange_rate', 'args': {'currency_from': 'USD', 'currency_to': 'KRW'}, 'id': '55ab34a9-dad9-4e25-ae92-83608b19c70a', 'type': 'tool_call'}]


이전 단계의 도구 호출 (tool calls)을 업데이트합니다.


```python
last_message.tool_calls[0]["args"]["currency_date"] = "2024-09-01"
last_message.tool_calls
```




    [{'name': 'get_exchange_rate',
      'args': {'currency_from': 'USD',
       'currency_to': 'KRW',
       'currency_date': '2024-09-01'},
      'id': '55ab34a9-dad9-4e25-ae92-83608b19c70a',
      'type': 'tool_call'}]



`.update_state`를 호출하여 업데이트된 구성으로 단계를 다시 실행합니다.


```python
branch_config = agent.update_state(
    config=state["config"],
    values={"messages": [last_message]},  # the update we want to make
)
branch_config
```




    {'configurable': {'thread_id': 'human-in-the-loop-deepdive',
      'checkpoint_ns': '',
      'checkpoint_id': '1efee6be-a197-6d4c-8002-2d6ffc4105b0'}}




```python
for state_values in agent.stream_query(
    input=None,  # resume
    stream_mode="values",
    config=branch_config,
):
    langchain_load(state_values["messages"][-1]).pretty_print()
```

    ==================================[1m Ai Message [0m==================================
    Tool Calls:
      get_exchange_rate (55ab34a9-dad9-4e25-ae92-83608b19c70a)
     Call ID: 55ab34a9-dad9-4e25-ae92-83608b19c70a
      Args:
        currency_from: USD
        currency_to: KRW
        currency_date: 2024-09-01
    =================================[1m Tool Message [0m=================================
    Name: get_exchange_rate
    
    {"amount": 1.0, "base": "USD", "date": "2024-08-30", "rates": {"KRW": 1333.63}}
    ==================================[1m Ai Message [0m==================================
    
    2024-08-30 기준 미국 달러에서 한국 통화로의 환율은 1333.63입니다.


## Deploying the Agent


```python
remote_agent = reasoning_engines.ReasoningEngine.create(
    reasoning_engines.LanggraphAgent(
        model="gemini-1.5-pro",
        tools=[get_exchange_rate],
        model_kwargs={"temperature": 0, "max_retries": 6},
        checkpointer_kwargs=checkpointer_kwargs,
        checkpointer_builder=checkpointer_builder,
    ),
    requirements=[
        "google-cloud-aiplatform[reasoningengine,langchain]",
        "requests",
    ],
)

remote_agent
```

## Querying the Remote Agent

### Remote testing


```python
for state_updates in remote_agent.stream_query(
    input=inputs,
    stream_mode="updates",
    config={"configurable": {"thread_id": "remote-streaming-thread-updates"}},
):
    print(state_updates)
```


```python
for state_values in remote_agent.stream_query(
    input=inputs,
    stream_mode="values",
    config={"configurable": {"thread_id": "remote-human-in-the-loop-overall"}},
):
    print(state_values)
```

### Reviewing Tool Calls


```python
response = remote_agent.query(
    input=inputs,
    interrupt_before=["tools"],  # Before invoking the tool.
    interrupt_after=["tools"],  # After getting a tool message.
    config={"configurable": {"thread_id": "human-in-the-loop-deepdive"}},
)
langchain_load(response["messages"][-1]).pretty_print()
```


```python
response = remote_agent.query(
    input=None,  # Resume (continue with the tool call AI Message).
    interrupt_before=["tools"],
    interrupt_after=["tools"],
    config={"configurable": {"thread_id": "human-in-the-loop-deepdive"}},
)
langchain_load(response["messages"][-1]).pretty_print()
```


```python
response = agent.query(
    input=None,  # Resume (continue with the Tool Message).
    interrupt_before=["tools"],
    interrupt_after=["tools"],
    config={"configurable": {"thread_id": "human-in-the-loop-deepdive"}},
)
langchain_load(response["messages"][-1]).pretty_print()
```

## Cleaning up

After you've finished experimenting, it's a good practice to clean up your cloud resources. You can delete the deployed Reasoning Engine instance to avoid any unexpected charges on your Google Cloud account.


```python
remote_agent.delete()
```
