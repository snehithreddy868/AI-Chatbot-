from chatbot import BasicChatbot
import sys
import nltk

def main():
    try:
        # Initialize chatbot model
        bot = BasicChatbot("intents.json")
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)

    print("Chatbot initialized! Let's chat! (type 'quit' to exit)")
    
    while True:
        try:
            sentence = input("You: ")
            if sentence.lower() == "quit":
                break
            
            resp = bot.get_response(sentence)
            print(f"Bot: {resp}")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
