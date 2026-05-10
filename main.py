import uuid
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

from langchain_core.messages import HumanMessage
from llm import app

def main():
    print("Welcome to the Website Crawler Chat! Type 'quit' or 'exit' to stop.")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue

            result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            bot_response = result["messages"][-1].content
            print(f"\nBot: {bot_response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
