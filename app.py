# Step 1 — Import all libraries
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
# Use local sentence-transformers embeddings when OpenAI quota is unavailable
class LocalEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embs = self.model.encode(texts, show_progress_bar=False)
        return [list(map(float, e)) for e in embs]
# Note: older `RetrievalQA` import isn't available in this environment.
# We'll perform a lightweight retrieval + LLM prompt flow instead of using RetrievalQA.
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import gradio as gr
print('Imported top-level libraries')
# Step 2 — Load API Key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"OPENAI_API_KEY set: {bool(api_key)}")

# Step 3 — Load all PDFs from data folder
print("Loading PDFs...")
loader = PyPDFDirectoryLoader("data/")
documents = loader.load()
print(f"Loaded {len(documents)} pages from PDFs")

# Step 4 — Split documents into smaller chunks
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Each chunk = 1000 characters
    chunk_overlap=200   # Overlap to avoid losing context
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")
print("Document splitting complete")
# Step 5 — Create Vector Database (ChromaDB)
print("Creating or loading vector database...")
# Prefer local embeddings to avoid external API quota issues
try:
    embeddings = LocalEmbeddings()
    print("Using local sentence-transformers embeddings")
except Exception as e:
    print(f"Local embeddings failed: {e}; falling back to OpenAIEmbeddings if available")
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(api_key=api_key)

# If a persisted Chroma DB exists, load it to avoid rebuilding embeddings
if os.path.exists("chroma_db") and any(os.scandir("chroma_db")):
    try:
        print("Found existing chroma_db, loading from disk...")
        vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        print("Loaded vector database from chroma_db")
    except Exception as e:
        print(f"Failed to load existing chroma_db: {e}; will create a new one")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        print("Vector database created!")
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print("Vector database created!")
# Step 6 — Set up Privacy Guardrails (Presidio)
try:
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    privacy_available = True
    print("Presidio analyzer and anonymizer initialized")
except Exception as e:
    print(f"Presidio initialization failed, continuing without privacy guardrails: {e}")
    analyzer = None
    anonymizer = None
    privacy_available = False

def remove_sensitive_data(text):
    # If Presidio failed to initialize, return text unchanged (no-op)
    if not privacy_available:
        return text

    # Detect sensitive information
    results = analyzer.analyze(
        text=text,
        entities=["PHONE_NUMBER", "EMAIL_ADDRESS",
                  "CREDIT_CARD", "DATE_TIME",
                  "PERSON", "IN_PAN"],  # Indian PAN card
        language="en"
    )
    # Mask sensitive information
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    return anonymized.text
# Step 7 — Set up AI Model and RAG Chain
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=api_key
)
print("LLM initialized")
# We'll not use RetrievalQA here (not available); use direct retrieval + LLM call
def _get_top_chunks(query, k=3):
    # Prefer vectorstore.similarity_search when available
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception:
        try:
            retr = vectorstore.as_retriever(search_kwargs={"k": k})
            # different retriever implementations expose different methods
            if hasattr(retr, "get_relevant_documents"):
                return retr.get_relevant_documents(query)
            if hasattr(retr, "retrieve"):
                return retr.retrieve(query)
        except Exception:
            return []
    return []
print("RAG chain ready")
# Step 8 — Chatbot Function
def insurance_chatbot(question):
    # First remove sensitive data from the question
    clean_question = remove_sensitive_data(question)
    # Retrieve top chunks
    docs = _get_top_chunks(clean_question, k=3)
    context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    if not api_key:
        # Avoid calling external LLM when API key not configured; return retrieved context.
        return "OPENAI_API_KEY not configured — retrieved context:\n\n" + (context or "(no context found)")

    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {clean_question}\nAnswer:"
    try:
        res = llm.invoke({"input": prompt})
        if isinstance(res, str):
            answer = res
        elif isinstance(res, dict):
            answer = res.get("output") or res.get("result") or str(res)
        else:
            answer = str(res)
    except Exception as e:
        answer = f"LLM call failed: {e}. Retrieved context length: {len(context)}"

    # Remove sensitive data from answer too
    clean_answer = remove_sensitive_data(answer)
    return clean_answer

# Step 9 — Build Gradio Web Interface
print("Launching chatbot...")
interface = gr.Interface(
    fn=insurance_chatbot,
    inputs=gr.Textbox(
        placeholder="Ask your insurance question here...",
        label="Your Question"
    ),
    outputs=gr.Textbox(label="Insurance Agent Answer"),
    title="🛡️ Insurance Agent RAG Chatbot",
    description="Ask questions about your insurance policies. sensitive data is automatically protected!",
)


if __name__ == "__main__":
    print("Starting Gradio app on http://127.0.0.1:7860")
    interface.launch(share=False, server_name="127.0.0.1", server_port=7860)
