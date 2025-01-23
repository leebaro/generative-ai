# Vertex AI의 추론 엔진

[추론 엔진](https://cloud.google.com/vertex-ai/generative-ai/docs/reasoning-engine/overview)
(Vertex AI의 LangChain)은 에이전트 추론 프레임워크를 구축하고 배포하는 데 도움이 되는 관리형 서비스입니다. 이를 통해 LLM에 위임할 추론의 양과 사용자 정의 코드로 처리할 양을 유연하게 선택할 수 있습니다. Gemini 함수 호출을 통해 도구로 사용되는 Python 함수를 정의할 수 있습니다.

추론 엔진은 Vertex AI의 Gemini 모델용 Python SDK와 긴밀하게 통합되어 있으며, 프롬프트, 에이전트 및 예제를 모듈 방식으로 관리할 수 있습니다. 추론 엔진은 LangChain, LlamaIndex 또는 기타 Python 프레임워크와 호환됩니다.

## 샘플 노트북

| 설명                                                                              | 샘플 이름                                                                            |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Vertex AI에서 추론 엔진을 사용하여 에이전트 구축 및 배포 소개              | [intro_reasoning_engine.ipynb](intro_reasoning_engine.ipynb)                           |
| 추론 엔진에서 에이전트 디버깅 및 최적화: 추적 가이드                  | [tracing_agents_in_reasoning_engine.ipynb](tracing_agents_in_reasoning_engine.ipynb)   |
| Vertex AI 검색에서 추론 엔진 및 RAG를 사용하여 대화형 검색 에이전트 구축 | [tutorial_vertex_ai_search_rag_agent.ipynb](tutorial_vertex_ai_search_rag_agent.ipynb) |
| 추론 엔진을 사용하여 Google Maps API 에이전트 구축 및 배포                     | [tutorial_google_maps_agent.ipynb](tutorial_google_maps_agent.ipynb)                   |
| Vertex AI에서 추론 엔진을 사용하여 LangGraph 애플리케이션 구축 및 배포        | [tutorial_langgraph.ipynb](tutorial_langgraph.ipynb)                                   |
| 추론 엔진을 사용하여 AlloyDB로 RAG 애플리케이션 배포                           | [tutorial_alloydb_rag_agent.ipynb](tutorial_alloydb_rag_agent.ipynb)                   |
| 추론 엔진을 사용하여 PostgreSQL용 Cloud SQL로 RAG 애플리케이션 배포          | [tutorial_cloud_sql_pg_rag_agent.ipynb](tutorial_cloud_sql_pg_rag_agent.ipynb)         |
