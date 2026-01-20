#!/usr/bin/env python3
"""Solar LangChain Integration Examples

README.md의 예시와 동일한 코드입니다.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from solar_langchain import SolarChatModel, SolarLLM


def example_llm():
    """SolarLLM (텍스트 완성) 예시"""
    print("=" * 60)
    print("SolarLLM (텍스트 완성)")
    print("=" * 60)

    # 인스턴스 생성
    llm = SolarLLM(reasoning_effort="medium")

    # 단일 호출
    print("\n[단일 호출]")
    response = llm.invoke("파이썬의 장점을 설명해주세요")
    print(response)

    # 스트리밍
    print("\n[스트리밍]")
    for chunk in llm.stream("한국의 사계절에 대해 알려주세요"):
        print(chunk, end="", flush=True)
    print()


def example_chat_model():
    """SolarChatModel (대화형) 예시"""
    print("\n" + "=" * 60)
    print("SolarChatModel (대화형)")
    print("=" * 60)

    # 인스턴스 생성
    chat = SolarChatModel(reasoning_effort="high")

    # 메시지 리스트로 대화
    print("\n[메시지 리스트로 대화]")
    messages = [
        SystemMessage(content="당신은 친절한 AI 어시스턴트입니다."),
        HumanMessage(content="안녕하세요!"),
    ]

    response = chat.invoke(messages)
    print(response.content)

    # 스트리밍
    print("\n[스트리밍]")
    for chunk in chat.stream([HumanMessage(content="짧은 이야기를 들려주세요")]):
        print(chunk.content, end="", flush=True)
    print()


def example_chain():
    """LangChain 체인과 함께 사용 예시"""
    print("\n" + "=" * 60)
    print("LangChain 체인과 함께 사용")
    print("=" * 60)

    chat = SolarChatModel()

    prompt = ChatPromptTemplate.from_messages(
        [("system", "당신은 {role}입니다."), ("human", "{question}")]
    )

    chain = prompt | chat

    response = chain.invoke(
        {"role": "요리 전문가", "question": "김치찌개 맛있게 끓이는 법을 알려주세요"}
    )
    print(response.content)


def example_reasoning_effort():
    """reasoning_effort 설정 예시"""
    print("\n" + "=" * 60)
    print("reasoning_effort 설정")
    print("=" * 60)

    # 복잡한 문제에는 high 사용
    print("\n[high - 복잡한 문제]")
    llm_high = SolarLLM(reasoning_effort="high")
    response = llm_high.invoke("양자역학의 불확정성 원리를 설명해주세요")
    print(response)

    # 간단한 응답에는 low 사용
    print("\n[low - 간단한 응답]")
    llm_low = SolarLLM(reasoning_effort="low")
    response = llm_low.invoke("안녕하세요!")
    print(response)


def main():
    """모든 예시 실행"""
    print("Solar LangChain Integration 예시")
    print("=" * 60)

    example_llm()
    example_chat_model()
    example_chain()
    example_reasoning_effort()

    print("\n" + "=" * 60)
    print("예시 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
