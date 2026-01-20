#!/usr/bin/env python3
"""Solar LangChain Integration Examples"""

from langchain_core.messages import HumanMessage, SystemMessage

from solar_langchain import SolarChatModel, SolarLLM


def example_llm():
    """SolarLLM 기본 사용 예시"""
    print("=" * 60)
    print("SolarLLM 예시")
    print("=" * 60)

    llm = SolarLLM(reasoning_effort="medium")

    # 기본 호출
    print("\n[기본 호출]")
    response = llm.invoke("파이썬의 장점을 한 문장으로 설명해주세요.")
    print(f"응답: {response}")

    # 스트리밍 호출
    print("\n[스트리밍 호출]")
    print("응답: ", end="")
    for chunk in llm.stream("LangChain이 무엇인지 간단히 설명해주세요."):
        print(chunk, end="", flush=True)
    print()


def example_chat_model():
    """SolarChatModel 기본 사용 예시"""
    print("\n" + "=" * 60)
    print("SolarChatModel 예시")
    print("=" * 60)

    chat = SolarChatModel(reasoning_effort="medium")

    # 기본 호출
    print("\n[기본 호출]")
    messages = [HumanMessage(content="안녕하세요! 자기소개를 해주세요.")]
    response = chat.invoke(messages)
    print(f"응답: {response.content}")

    # 대화 기록 유지
    print("\n[대화 기록 유지]")
    messages = [
        SystemMessage(content="당신은 친절한 한국어 AI 어시스턴트입니다."),
        HumanMessage(content="내 이름은 철수야."),
    ]
    response1 = chat.invoke(messages)
    print(f"응답 1: {response1.content}")

    messages.append(response1)
    messages.append(HumanMessage(content="내 이름이 뭐라고 했지?"))
    response2 = chat.invoke(messages)
    print(f"응답 2: {response2.content}")

    # 스트리밍 호출
    print("\n[스트리밍 호출]")
    print("응답: ", end="")
    for chunk in chat.stream([HumanMessage(content="1부터 5까지 세어주세요.")]):
        print(chunk.content, end="", flush=True)
    print()


def example_reasoning_effort():
    """추론 수준별 비교 예시"""
    print("\n" + "=" * 60)
    print("추론 수준별 비교")
    print("=" * 60)

    prompt = "2+2*3의 결과는?"

    for effort in ["low", "medium", "high"]:
        print(f"\n[reasoning_effort={effort}]")
        llm = SolarLLM(reasoning_effort=effort)
        response = llm.invoke(prompt)
        print(f"응답: {response}")


def main():
    """모든 예시 실행"""
    print("Solar LangChain Integration 예시")
    print("=" * 60)

    example_llm()
    example_chat_model()
    example_reasoning_effort()

    print("\n" + "=" * 60)
    print("예시 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
