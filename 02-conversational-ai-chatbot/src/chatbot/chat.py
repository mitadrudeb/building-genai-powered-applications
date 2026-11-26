
from chatbot.local_model import get_reply, load_model


def main():
    print("Loading model... (1-2 minutes the first time)")
    processor, model = load_model()
    history = []  # this list is the bot's memory

    print("Ready. Type 'reset' to clear memory, 'quit' to stop.")
    while True:
        user_text = input("\nYou: ").strip()
        if user_text == "quit":
            break
        if user_text == "reset":
            history = []
            print("(memory cleared)")
            continue
        if user_text == "":
            continue

        history.append({"role": "user", "content": user_text})
        answer = get_reply(processor, model, history)
        history.append({"role": "assistant", "content": answer})
        print("\nBot:", answer)


if __name__ == "__main__":
    main()