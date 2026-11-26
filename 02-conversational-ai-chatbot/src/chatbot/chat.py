"""Terminal chat with the local model.

This is the Stage 2 version of the chatbot. It runs in a plain terminal
(in Colab), with no web UI. It is a quick way to check that the model
loads and that it remembers earlier messages.
"""

from chatbot.local_model import get_reply, load_model


def main():
    """Load the model, then chat in a loop until the user types 'quit'.

    The whole conversation is kept in a list of messages. We send the full
    list to the model each turn, because the model itself has no memory.
    """
    print("Loading model... (1-2 minutes the first time)")
    processor, model = load_model()
    history = []  # the bot's memory: every message so far, in order

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
            continue  # skip empty input, so the model is not asked to answer nothing

        history.append({"role": "user", "content": user_text})
        answer = get_reply(processor, model, history)
        history.append({"role": "assistant", "content": answer})
        print("\nBot:", answer)


if __name__ == "__main__":
    main()