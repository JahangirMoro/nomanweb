from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Set up embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings Initialized!")

# Step 2: Load documents from 'data/' folder
loader = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

# Step 3: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")

# Step 4: Create FAISS vector store
faiss_store = FAISS.from_documents(
    documents=texts,
    embedding=embeddings
)
print("FAISS Vector Store Created!")

# Step 5: Save FAISS vector store
faiss_store.save_local("faiss_vectors")
print("FAISS Vector Store Saved to 'faiss_vectors' Folder.")

# Step 6: Load FAISS vector store safely
try:
    loaded_faiss_store = FAISS.load_local(
        "faiss_vectors", 
        embeddings, 
        allow_dangerous_deserialization=True  # Explicitly enable this parameter
    )
    print("FAISS Vector Store Successfully Loaded!")
except Exception as e:
    print(f"Error Loading FAISS Vector Store: {e}")
