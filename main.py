import uuid

from dotenv import load_dotenv
import os
from datetime import datetime
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import gradio as gr
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_tavily import TavilySearch
# from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_date():
    """Get the current date"""
    return datetime.now().strftime("%Y-%m-%d")

search_tool = TavilySearch()

conn = sqlite3.connect("chatbot_memory.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

llm = ChatOllama(model="qwen2.5:7b")

system_prompt = """
You are a helpful assistant.
Answer all user's queries.
Use the get_date tool ONLY when the user is explicitly asking for today's date.
Use the search_tool for answering questions that require up to date information.
"""

agent = create_agent(llm, tools=[get_date, search_tool], system_prompt=system_prompt, checkpointer=checkpointer)

# user_query = input("Enter a query ")


# print(response['messages'][-1].content)

def chat(message, history, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    last_response = response['messages'][-1].content
    return last_response


with gr.Blocks() as block:
    gr.Markdown("# AI Chatbot")
    thread_id = gr.State(value=lambda : str(uuid.uuid4()))
    gr.ChatInterface(fn=chat, additional_inputs=[thread_id])

block.launch()
