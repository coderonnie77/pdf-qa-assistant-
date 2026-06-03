from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from ingest import load_vector_store
import os

# Tool 1: Search the user's uploaded PDF notes
def get_rag_tool():
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    @tool
    def search_notes(query: str) -> str:
        """
        Search the user's uploaded study notes for information.
        Use this tool FIRST whenever the user asks a question about their material.
        Returns relevant text chunks from their notes.
        """
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant content found in notes."
        results = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            results.append(f"[Chunk {i} - Page {page}]:\n{doc.page_content}")
        return "\n\n".join(results)

    return search_notes

# Tool 2: Search the web
def get_web_search_tool():
    tavily = TavilySearchResults(
        max_results=3,
        api_key=os.getenv("TAVILY_API_KEY")
    )

    @tool
    def search_web(query: str) -> str:
        """
        Search the internet for information NOT found in the user's notes.
        Use this when the notes don't contain the answer, or the user asks
        about topics outside their uploaded material.
        """
        results = tavily.invoke(query)
        if not results:
            return "No web results found."
        formatted = []
        for r in results:
            formatted.append(f"Source: {r['url']}\n{r['content']}")
        return "\n\n---\n\n".join(formatted)

    return search_web

# Tool 3: Generate quiz questions
@tool
def generate_quiz(topic: str) -> str:
    """
    Generate 5 multiple choice quiz questions on a given topic.
    Use this when the user asks to be quizzed, tested, or wants practice questions.
    """
    return f"""Please generate 5 multiple choice questions about: {topic}

Format each question like this:
Q1. [Question text]
A) [Option]  B) [Option]  C) [Option]  D) [Option]
Answer: [Correct letter]

Make questions clear and educational."""