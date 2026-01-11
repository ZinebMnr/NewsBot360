import asyncio
import threading
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

load_dotenv()
st.set_page_config(page_title="NewsBot360", page_icon="🤖")

# --- MCP (le minimum pour garder l'agent vivant) ---
class MCPChatSession:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()

        # 🔧 À ADAPTER
        self.config = {
            "mcpServers": {
                "fii-demo": {
                    "command": r"D:/projets/NewsBot360/mcp/Scripts/python.exe",
                    "args": ["server.py"],
                }
            }
        }

        asyncio.run_coroutine_threadsafe(self._init(), self.loop).result()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _init(self):
        client = MCPClient.from_dict(self.config)
        llm = ChatGroq(model="llama-3.3-70b-versatile")
        self.agent = MCPAgent(llm=llm, client=client, memory_enabled=True, verbose=False)

    def run(self, text: str):
        fut = asyncio.run_coroutine_threadsafe(self.agent.run(text), self.loop)
        return fut.result()

    def clear(self):
        self.agent.clear_conversation_history()


@st.cache_resource
def get_chat():
    return MCPChatSession()


# --- UI (chat Streamlit ultra simple) ---
st.title("🤖 NewsBot360")

chat = get_chat()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Salut 👋"}]

if st.button("Clear"):
    chat.clear()
    st.session_state.messages = [{"role": "assistant", "content": "OK, on repart de zéro."}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ton message…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            answer = chat.run(prompt)
        except Exception as e:
            answer = f"Erreur: {e}"
        st.markdown(str(answer) if answer else "_(Aucune réponse)_")

    st.session_state.messages.append({"role": "assistant", "content": str(answer)})
