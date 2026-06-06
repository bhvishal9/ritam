import sys

from ritam.config.settings import get_settings
from ritam.llm.errors import LlmError
from ritam.llm.gemini_client import GeminiClient
from ritam.llm.types import LlmClient


def print_exit() -> None:
    print("\nGoodbye!")


def chat_loop(client: LlmClient) -> None:
    print("Welcome to the chat! Type '/exit' to quit.\n")
    while True:
        try:
            user_input = input("[You]: ").strip()
        except EOFError, KeyboardInterrupt:
            print_exit()
            return
        if user_input.lower() == "/exit":
            print_exit()
            return
        if not user_input:
            continue
        response = client.generate_response(user_input)
        print(f"[Gemini]: {response}")


def main() -> int:
    try:
        settings = get_settings()
        client = GeminiClient(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            embedding_model=settings.llm_embedding_model,
        )
        chat_loop(client)
    except ValueError as err:
        print(f"Config Error: {err}")
        return 1
    except LlmError as err:
        print(f"LLM Client Error: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
