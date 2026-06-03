import streamlit as st
import os
import tempfile
from ingest import create_vector_store
from agent import create_study_agent, run_agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="PDF Q&A chatbot", page_icon="📚", layout="wide")
st.title("AI Study Assistant Agent")
st.caption("Upload your notes → Ask questions → Get answers from your material or the web")

# Session state — persists data across Streamlit reruns
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# Sidebar
with st.sidebar:
    st.header("Setup")

    uploaded_file = st.file_uploader("Upload your study notes (PDF)", type=["pdf"])

    if uploaded_file is not None and not st.session_state.pdf_loaded:
        with st.spinner("Reading PDF and building knowledge base..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            create_vector_store(tmp_path)
            os.unlink(tmp_path)
            st.session_state.agent = create_study_agent(use_rag=True)
            st.session_state.pdf_loaded = True
        st.success(f"'{uploaded_file.name}' loaded!")

    if st.session_state.agent is None:
        st.session_state.agent = create_study_agent(use_rag=False)

    st.divider()
    st.subheader("Active tools")
    if st.session_state.pdf_loaded:
        st.success("Notes search (RAG)")
    st.success("Web search")
    st.success("Quiz generator")

    st.divider()
    st.subheader("Try asking:")
    examples = [
        "Summarise the key points from my notes",
        "Quiz me on machine learning",
        "What is backpropagation?",
        "Explain gradient descent simply"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": ex})
            st.rerun()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.agent = create_study_agent(use_rag=st.session_state.pdf_loaded)
        st.rerun()

# Chat history display
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("tools_used"):
            with st.expander("See agent's reasoning"):
                for tool_name, tool_input in msg["tools_used"]:
                    st.markdown(f"**Tool:** `{tool_name}`")
                    st.markdown(f"**Query:** {tool_input}")

# Chat input
if user_input := st.chat_input("Ask anything about your studies..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent(st.session_state.agent, user_input)
        st.write(result["answer"])

        tools_used = []
        for action, _ in result["steps"]:
            tools_used.append((action.tool, action.tool_input))

        if tools_used:
            with st.expander("See agent's reasoning"):
                for tool_name, tool_input in tools_used:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"`{tool_name}`")
                    with col2:
                        st.markdown(str(tool_input)[:200])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "tools_used": tools_used
        })