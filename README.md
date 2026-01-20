# Solar LangChain

Upstage Solar Chat API의 비공식 LangChain 통합 라이브러리입니다.

> **주의사항**
>
> 이 프로젝트는 테스트/학습 목적으로 제작되었습니다.
> [Solar Chat](https://solar-chat.upstage.ai/chat/)을 참고하여 LangChain 통합 테스트를 위해 만들어졌으며,
> Upstage의 공식 라이브러리가 아닙니다.
>
> 기본 기능(`invoke`, `stream`)만 확인되었으며, 다른 LangChain 기능들은 별도의 검증이 필요합니다.
>
> 과도한 API 요청은 제공 업체의 서비스에 부담을 줄 수 있으니 절제하여 사용해주세요.

## 소개

이 프로젝트는 Upstage의 최신 Solar 모델이 LangChain 환경에서 잘 동작하는지 검증하기 위해 제작되었습니다.
API 키 없이 공개 엔드포인트를 사용합니다.

## 기능

- **SolarLLM**: LangChain LLM 인터페이스 (`langchain_core.language_models.llms.LLM`)
- **SolarChatModel**: LangChain ChatModel 인터페이스 (`langchain_core.language_models.chat_models.BaseChatModel`)

## 설치

### 요구사항

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (권장) 또는 pip

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/yourusername/solar_langchain.git
cd solar_langchain

# uv 사용 시
uv sync

# pip 사용 시
pip install -e .
```

### 의존성

```toml
dependencies = [
    "httpx>=0.28.1",        # HTTP 클라이언트 (SSE 스트리밍 지원)
    "langchain-core>=0.3.0", # LangChain 핵심 라이브러리
]
```

## 사용법

### SolarLLM (텍스트 완성)

```python
from solar_langchain import SolarLLM

# 인스턴스 생성
llm = SolarLLM(reasoning_effort="medium")

# 단일 호출
response = llm.invoke("파이썬의 장점을 설명해주세요")
print(response)

# 스트리밍
for chunk in llm.stream("한국의 사계절에 대해 알려주세요"):
    print(chunk, end="", flush=True)
```

### SolarChatModel (대화형)

```python
from solar_langchain import SolarChatModel
from langchain_core.messages import HumanMessage, SystemMessage

# 인스턴스 생성
chat = SolarChatModel(reasoning_effort="high")

# 메시지 리스트로 대화
messages = [
    SystemMessage(content="당신은 친절한 AI 어시스턴트입니다."),
    HumanMessage(content="안녕하세요!")
]

response = chat.invoke(messages)
print(response.content)

# 스트리밍
for chunk in chat.stream([HumanMessage(content="짧은 이야기를 들려주세요")]):
    print(chunk.content, end="", flush=True)
```

### LangChain 체인과 함께 사용

```python
from solar_langchain import SolarChatModel
from langchain_core.prompts import ChatPromptTemplate

chat = SolarChatModel()

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role}입니다."),
    ("human", "{question}")
])

chain = prompt | chat

response = chain.invoke({
    "role": "요리 전문가",
    "question": "김치찌개 맛있게 끓이는 법을 알려주세요"
})
print(response.content)
```

## 설정 옵션

### reasoning_effort

Solar 모델의 추론 수준을 설정합니다.

| 값 | 설명 | 용도 |
|------|------|------|
| `low` | 빠른 응답 | 간단한 질문, 인사 |
| `medium` | 균형 잡힌 응답 (기본값) | 일반적인 대화 |
| `high` | 깊은 사고 | 복잡한 문제 해결, 분석 |

```python
# 복잡한 문제에는 high 사용
llm = SolarLLM(reasoning_effort="high")

# 간단한 응답에는 low 사용
llm = SolarLLM(reasoning_effort="low")
```

### timeout

API 요청 타임아웃 (초 단위, 기본값: 120.0)

```python
llm = SolarLLM(timeout=60.0)  # 60초 타임아웃
```

## 파일 구조

```
solar_langchain/
├── __init__.py      # 패키지 초기화, 버전 정보
├── llm.py           # SolarLLM 클래스
├── chat_model.py    # SolarChatModel 클래스
├── example.py       # 사용 예시
├── pyproject.toml   # 프로젝트 설정
└── README.md
```

## 제한사항

- API 키 인증을 사용하지 않는 공개 엔드포인트 사용
- 프로덕션 환경에서의 사용은 권장하지 않음
- Upstage 공식 지원 라이브러리가 아님

---

# Solar LangChain (English)

Unofficial LangChain integration library for Upstage Solar Chat API.

> **Warning**
>
> This project is for testing and educational purposes only.
> It was created to test LangChain integration referencing [Solar Chat](https://solar-chat.upstage.ai/chat/),
> and is not an official Upstage library.
>
> Only basic features (`invoke`, `stream`) have been verified. Other LangChain features require separate testing.
>
> Excessive API requests may burden the service provider. Please use responsibly.

## Introduction

This project was created to verify that Upstage's latest Solar model works well in a LangChain environment.
It uses a public endpoint without API key authentication.

## Features

- **SolarLLM**: LangChain LLM interface (`langchain_core.language_models.llms.LLM`)
- **SolarChatModel**: LangChain ChatModel interface (`langchain_core.language_models.chat_models.BaseChatModel`)

## Installation

### Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install

```bash
# Clone repository
git clone https://github.com/yourusername/solar_langchain.git
cd solar_langchain

# Using uv
uv sync

# Using pip
pip install -e .
```

## Usage

### SolarLLM (Text Completion)

```python
from solar_langchain import SolarLLM

llm = SolarLLM(reasoning_effort="medium")

# Single call
response = llm.invoke("Explain the benefits of Python")
print(response)

# Streaming
for chunk in llm.stream("Tell me about the four seasons"):
    print(chunk, end="", flush=True)
```

### SolarChatModel (Conversational)

```python
from solar_langchain import SolarChatModel
from langchain_core.messages import HumanMessage, SystemMessage

chat = SolarChatModel(reasoning_effort="high")

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Hello!")
]

response = chat.invoke(messages)
print(response.content)
```

## Configuration Options

### reasoning_effort

Sets the reasoning level of the Solar model.

| Value | Description | Use Case |
|-------|-------------|----------|
| `low` | Fast response | Simple questions, greetings |
| `medium` | Balanced response (default) | General conversation |
| `high` | Deep thinking | Complex problem solving, analysis |

## Limitations

- Uses public endpoint without API key authentication
- Not recommended for production use
- Not an official Upstage-supported library
