import asyncio
import os
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

load_dotenv()

server_path = os.path.join("server.py")
CONFIG = {
    "mcpServers": {
        "fii-demo": {
            "command": "D:/projets/NewsBot360/mcp/Scripts/python.exe",
            "args": [server_path]
        }
    }
}

async def run_chatbot():
    """Running a chat using MCPAgent's built-in conversation memory"""
    
    client = MCPClient.from_dict(CONFIG)
    # llm = ChatOllama(model="qwen3:8b")
    llm = ChatGroq(model="llama-3.3-70b-versatile")

    
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=20,
        memory_enabled=True,  # This enables conversation memory
        verbose=False
    )
    
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history\n")
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break
            
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.\n")
                continue
            
            if not user_input:
                continue
            
            print("\nAssistant: ", end="", flush=True)
            
            try:
                # Just call agent.run() - memory is handled internally
                response = await agent.run(user_input)
                print(response)
            except Exception as e:
                print(f"\nError: {e}")
    
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_chatbot())