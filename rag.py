from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

File_Path = r"C:\Users\Kanhaiya\OneDrive\Desktop\local_projects\chatbot\data.txt"

with open(File_Path,"r",encoding="utf-8") as f:
    texts = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(texts)

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

bm25_retriver = BM25Retriever.from_texts(
    chunks,
    k=2
)
vectorstores = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="./ChromaDB"
)

retriver = vectorstores.as_retriever(
    search_kwargs={"k":3}
)
hybrid_search=EnsembleRetriever(
    retrievers=[bm25_retriver,retriver],
    weights=[0.5,0.5]
)

