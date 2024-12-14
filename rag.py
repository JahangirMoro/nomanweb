from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain_community.llms import Ollama
import os
import time  # For measuring response time
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from langchain.llms import HuggingFaceLLM

# Initialize Hugging Face model
model_name = "distilbert-base-uncased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)



# Initialize FastAPI application
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template directory
templates = Jinja2Templates(directory="templates")

# ---------- Application Routes (Frontend) ---------- #

@app.get("/", response_class=HTMLResponse, name="home")
async def root():
    """
    Redirect to the index page.
    """
    return RedirectResponse(url="/index/")

@app.get("/index/", response_class=HTMLResponse, name="index")
async def index(request: Request):
    """
    Render the main index page.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/services/", response_class=HTMLResponse, name="services")
async def services(request: Request):
    """
    Render the services page.
    """
    return templates.TemplateResponse("services.html", {"request": request})

@app.get("/about/", response_class=HTMLResponse, name="about")
async def about(request: Request):
    """
    Render the about page.
    """
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact/", response_class=HTMLResponse, name="contact")
async def contact(request: Request):
    """
    Render the contact page.
    """
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/elements/", response_class=HTMLResponse, name="elements")
async def elements(request: Request):
    """
    Render the elements page.
    """
    return templates.TemplateResponse("elements.html", {"request": request})

# ---------- Chatbot Initialization (Backend) ---------- #

# Initialize Local LLM Model
local_llm =  HuggingFaceLLM(model=model, tokenizer=tokenizer

# Initialize embeddings model using SentenceTransformer
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize FAISS vector store
faiss_store = FAISS.load_local("faiss_vectors", embeddings, allow_dangerous_deserialization=True)
print("FAISS Vector Store Loaded Successfully!")

# Define the prompt template for chatbot responses
prompt_template = """
Use ONLY the information provided in the context below to answer the user's question.
Do not provide an answer if the information is not available in the context.
Also, write my mission when someone asks, "what is your mission".

Context: {context}
Question: {question}

Helpful answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ---------- Chatbot API Endpoint ---------- #

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    """
    Handle the chatbot query and return a response generated from the LLM.
    """
    start_time = time.time()  # Record the start time

    # Handle specific mission query
    if query.lower() == "what is your mission":
        return JSONResponse(content={"answer": "Our mission is to provide accurate and reliable information using cutting-edge AI."})

    # Set up the retriever
    retriever = faiss_store.as_retriever(search_kwargs={"k": 1})
    chain_type_kwargs = {"prompt": prompt}

    # Create a RetrievalQA pipeline with the specified prompt
    qa = RetrievalQA.from_chain_type(
        llm=local_llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    # Generate response from the query
    try:
        response = qa(query)
        answer = response.get("result", "No answer found.")
        source_document = (
            response["source_documents"][0].page_content
            if response["source_documents"]
            else "No context found."
        )
        doc = (
            response["source_documents"][0].metadata.get("source", "Unknown source.")
            if response["source_documents"]
            else "Unknown source."
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    # Calculate the response time
    end_time = time.time()
    response_time = end_time - start_time  # Time taken for the request to complete

    # Return JSON response with the generated answer, source, and response time
    response_data = {
        "answer": answer,
        "source_document": source_document,
        "doc": doc,
        "response_time": response_time,
    }

    return JSONResponse(content=response_data)

# ---------- Script to Run Without Command and Open Browser ---------- #

if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable
    PORT = int(os.environ.get("PORT", 8000))

    # Start the server
    uvicorn.run("rag:app", host="0.0.0.0", port=PORT, reload=True)
