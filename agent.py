from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import hub
from tools import get_rag_tool, get_web_search_tool, generate_quiz
from dotenv import load_dotenv
import os

load_dotenv()

def create_study_agent(use_rag: bool = False):
    # 1. LLM — free Llama-3 via Groq
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # 2. Tools
    tools = [generate_quiz, get_web_search_tool()]
    if use_rag:
        tools.insert(0, get_rag_tool())

    # 3. Memory — remembers conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 4. ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react-chat")

    # 5. Create agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # 6. Executor — handles the Thought → Action → Observation loop
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )

    return agent_executor

def run_agent(agent_executor, user_question: str) -> dict:
    try:
        result = agent_executor.invoke({"input": user_question})
        return {
            "answer": result["output"],
            "steps": result.get("intermediate_steps", [])
        }
    except Exception as e:
        return {
            "answer": f"Something went wrong: {str(e)}",
            "steps": []
        }