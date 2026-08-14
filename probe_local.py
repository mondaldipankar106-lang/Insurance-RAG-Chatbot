import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

class LocalEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        embs = self.model.encode(texts, show_progress_bar=False)
        return [list(map(float, e)) for e in embs]

print('Loader...')
loader = PyPDFDirectoryLoader('data/')
start=time.time()
docs = loader.load()
print('Loaded', len(docs), 'pages in', time.time()-start)
print('Split...')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
start=time.time()
chunks = text_splitter.split_documents(docs)
print('Chunks', len(chunks), 'in', time.time()-start)
print('Init embeddings')
start=time.time()
emb = LocalEmbeddings()
print('Embeddings ready in', time.time()-start)
print('Creating Chroma...')
start=time.time()
vs = Chroma.from_documents(documents=chunks, embedding=emb, persist_directory='chroma_db_local')
print('Chroma created in', time.time()-start)
