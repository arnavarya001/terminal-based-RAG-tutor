import os
import sys
import warnings
from dotenv import load_dotenv

# Suppress deprecation notices from underlying community packages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 1. Load environment variables
load_dotenv()
# Also check for .env in parent directories if not found locally
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "tutor", ".env"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Google GenAI SDK accepts GOOGLE_API_KEY; map GEMINI_API_KEY if present
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# 2. Imports for LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


def ingest_pdf(pdf_path: str, persist_directory: str = "db"):
    """
    Loads a PDF, splits it into chunks, converts chunks into embeddings,
    and stores them in a local Chroma vector database.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        sys.exit(1)

    print(f"\n[1/4] Loading PDF: {pdf_path}...")
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    print(f"  -> Successfully loaded {len(documents)} page(s).")

    print("\n[2/4] Splitting text into chunks (chunk_size=500, chunk_overlap=50)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)

    # Required print statement showing how many chunks were created
    print(f"  -> Created {len(chunks)} chunks from {len(documents)} page(s).")

    if not chunks:
        print("Warning: No text chunks extracted. Vector database will not be updated.")
        return

    # Print a quick preview of the first chunk
    print(f"  -> Sample chunk preview (first 120 chars):")
    sample_preview = chunks[0].page_content.replace("\n", " ").strip()[:120]
    print(f"     \"{sample_preview}...\"")

    print("\n[3/4] Initializing Google Generative AI Embeddings...")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: No Google / Gemini API key found.")
        print("Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")
        sys.exit(1)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    print(f"\n[4/4] Generating embeddings and storing in Chroma vector database at '{persist_directory}'...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # Note: In Chroma 0.4.x+, documents are automatically persisted to disk

    collection_count = vector_store._collection.count() if hasattr(vector_store, "_collection") else len(chunks)
    print(f"\nIngestion Complete!")
    print(f"  - Total chunks processed and stored: {len(chunks)}")
    print(f"  - Chroma database directory: {os.path.abspath(persist_directory)}")
    print(f"  - Total documents in Chroma collection: {collection_count}")


if __name__ == "__main__":
    # Determine the target PDF path
    if len(sys.argv) > 1:
        target_pdf = sys.argv[1]
    else:
        # Default to sample.pdf in the script's directory if available
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_sample = os.path.join(script_dir, "sample.pdf")
        if os.path.exists(default_sample):
            target_pdf = default_sample
            print(f"No PDF file specified. Defaulting to: {target_pdf}")
        else:
            print("Usage: python ingest.py <path_to_pdf>")
            sys.exit(1)

    ingest_pdf(pdf_path=target_pdf, persist_directory="db")
