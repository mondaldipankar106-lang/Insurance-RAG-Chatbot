import time
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
print('API key set:', bool(api_key))

print('Creating loader')
loader = PyPDFDirectoryLoader('data/')
print('Loader created; starting load')
start = time.time()
docs = loader.load()
print('Loaded docs in', time.time()-start, 'seconds; count=', len(docs))

print('Splitting...')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
start = time.time()
chunks = text_splitter.split_documents(docs)
print('Split in', time.time()-start, 'seconds; chunks=', len(chunks))

print('Embedding init...')
emb = OpenAIEmbeddings(api_key=api_key)
print('Embedding created')
print('Creating Chroma...')
start = time.time()
vs = Chroma.from_documents(documents=chunks, embedding=emb, persist_directory='chroma_db')
print('Chroma created in', time.time()-start)
print('Done')
