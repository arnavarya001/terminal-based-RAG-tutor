import os
import sys
import warnings

# Suppress all deprecation and runtime warnings for a clean CLI output
warnings.simplefilter("ignore")
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check local, sibling, and parent directories for .env
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    load_dotenv(os.path.join(script_dir, ".env"))
    load_dotenv(os.path.join(script_dir, "..", "langchain", ".env"))
    load_dotenv(os.path.join(script_dir, "..", "tutor", ".env"))
    load_dotenv(os.path.join(script_dir, "..", ".env"))

# Google GenAI SDK prefers GOOGLE_API_KEY; keep only GOOGLE_API_KEY in environment
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    # Remove GEMINI_API_KEY from environment to avoid duplicate key notice
    os.environ.pop("GEMINI_API_KEY", None)
else:
    print("Error: No Gemini or Google API key found.")
    print("Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")
    sys.exit(1)

# 2. LangChain components imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


def resolve_db_directory(target_folder: str = "db") -> str:
    """Resolves the Chroma db folder path across common locations."""
    candidates = [
        target_folder,
        os.path.join(script_dir, target_folder),
        os.path.join(script_dir, "..", "langchain", target_folder),
    ]
    for path in candidates:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    return target_folder


def format_docs(docs) -> str:
    """Formats retrieved document chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(db_path: str = "db"):
    """
    1. Loads the Chroma database from the db folder.
    2. Creates a retriever that fetches the top 3 most relevant chunks.
    3. Builds an LCEL chain with automatic fallback models to avoid free-tier quota exhaustion.
    """
    resolved_path = resolve_db_directory(db_path)
    if not os.path.exists(resolved_path):
        print(f"Error: Vector database folder '{resolved_path}' not found.")
        print("Please run `python ingest.py` first to generate the Chroma database.")
        sys.exit(1)

    print(f"Loading Chroma database from: {os.path.abspath(resolved_path)}...")

    # Embeddings model (matching the ingestion configuration)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    # Load Chroma vector store
    vectorstore = Chroma(
        persist_directory=resolved_path,
        embedding_function=embeddings,
    )

    # Retriever fetching top 3 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # RAG prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful and accurate AI assistant answering questions based on the retrieved context.
Use ONLY the following pieces of retrieved context to answer the question. If the context does not contain enough information to answer, state clearly that you don't know based on the provided documents.

Context:
{context}

Question:
{question}

Answer:"""
    )

    # Primary chat model: gemini-3.5-flash-lite has fresh quota
    # Fallback model: gemini-3.1-flash-lite handles requests if primary reaches rate limits
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    primary_llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
        max_retries=1,
    )
    fallback_llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.2,
        max_retries=1,
    )
    llm = primary_llm.with_fallbacks([fallback_llm])

    # LangChain LCEL Chain:
    # 1. Takes {"context": retriever | format_docs, "question": RunnablePassthrough()}
    # 2. Passes dictionary to prompt
    # 3. Sends formatted prompt to llm
    # 4. Parses output to string
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def ask_and_print(rag_chain, question: str):
    """Executes the RAG chain and handles rate limits or API errors gracefully."""
    try:
        answer = rag_chain.invoke(question)
        print(f"\nAnswer:\n{answer}\n")
    except Exception as e:
        err_text = str(e)
        if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text:
            print("\n[Rate Limit Notice]")
            print("Your Google Gemini API free-tier quota (20 requests/day per model) was reached.")
            print("Tip: You can change the model in .env via GEMINI_MODEL=gemini-3.1-flash-lite or wait for the daily quota reset.")
        else:
            print(f"\n[Error generating answer]: {err_text}\n")


def main():
    rag_chain, retriever = create_rag_chain("db")

    # If question is passed via command-line argument:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        print(f"\nQuestion: {question}")
        print("-" * 50)
        retrieved_chunks = retriever.invoke(question)
        print(f"[Retrieved {len(retrieved_chunks)} relevant chunk(s) from database]")
        ask_and_print(rag_chain, question)
        return

    # Interactive terminal loop
    print("\n" + "=" * 55)
    print("  LangChain + Chroma RAG Assistant")
    print("  Ask questions about your documents!")
    print("  (Type 'exit', 'quit', or 'q' to end the session)")
    print("=" * 55)

    while True:
        try:
            question = input("\nAsk a question: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                break

            print("\nSearching database (top 3 chunks) and generating answer...")
            retrieved_chunks = retriever.invoke(question)
            print(f"[Retrieved {len(retrieved_chunks)} relevant chunk(s)]")

            ask_and_print(rag_chain, question)
            print("-" * 55)

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
