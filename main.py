from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#from langchain.schema import Document

# Pinecone
from pinecone import Pinecone



import logging
import torch

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Medical Chatbot API",
    description="A medical chatbot powered by RAG using Pinecone, LangChain, and Gemini 2.5 Flash",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    conversation_id: str

class HealthCheckResponse(BaseModel):
    status: str
    message: str

# Global variables
embeddings = None
vectorstore = None
llm = None
qa_chain = None

def get_optimal_device():
    """Detect and return the optimal device for embeddings
      (GPU if available and working, otherwise CPU)
    """
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.current_device()
                return "cuda"
            except Exception as e:
                logger.warning(f"CUDA available but not working properly: {e}")
                return "cpu"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch not available, defaulting to CPU")
        return "cpu"

def initialize_components():
    """Initialize all the RAG components"""
    global embeddings, vectorstore, llm, qa_chain
    
    try:
        logger.info("Initializing embeddings...")
        # Auto-detect optimal device
        device = get_optimal_device()
        logger.info(f"Using device: {device}")
        
        # Initialize embeddings with fallback handling
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": device}
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize with {device}, trying CPU fallback: {e}")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            logger.info("Embeddings initialized with CPU fallback")
        
        logger.info("Connecting to Pinecone...")
        
        
        # Connect to Pinecone
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "medical-chatbot-index"
        index = pc.Index(index_name)
        
        # Initialize vector store
        vectorstore = PineconeVectorStore(
            embedding=embeddings,
            index=index
        )
        
        logger.info("Initializing Gemini LLM...")
        
        
        # Initialize Gemini LLM
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,
            max_tokens=1000
        )
        
        logger.info("Setting up QA chain...")
        
        
        
        # Prompt
        medical_prompt_template = """
        You are a helpful medical assistant. Use the following pieces of context to answer the user's question about medical topics.
        
        Important guidelines:
        - Provide accurate, evidence-based information
        - Always recommend consulting healthcare professionals for medical advice
        - Be clear about limitations and when professional consultation is needed
        - If you don't know something, say so clearly
        
        Context: {context}
        
        Question: {question}
        
        Answer: Provide a helpful and informative response based on the context provided. Always remind users to consult healthcare professionals for personal medical advice.
        """
        
        medical_prompt = PromptTemplate(
            template=medical_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": medical_prompt},
            return_source_documents=True
        )
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize components when the app starts"""
    initialize_components()

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test if components are working
        if qa_chain is None:
            raise Exception("QA chain not initialized")
        
        return HealthCheckResponse(
            status="healthy",
            message="Medical chatbot is running successfully"
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Main chat endpoint"""
    try:
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")
        
        # Get response from QA chain
        result = qa_chain.invoke({"query": request.message})
        
        # Extract source information
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source_info = f"Page {doc.metadata.get('page', 'N/A')}"
                if source_info not in sources:
                    sources.append(source_info)
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or f"conv_{hash(request.message) % 10000}"
        
        return ChatResponse(
            response=result["result"],
            sources=sources,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root(request: Request):
    """Serve the frontend page"""
    return templates.TemplateResponse("index.html", {"request": request})

# Additional utility endpoints
#@app.post("/add_document")
#async def add_document(file_path: str):
#    """Add a new PDF document to the vector store"""
#    try:
#        if not os.path.exists(file_path):
#            raise HTTPException(status_code=404, detail="File not found")
#        
        # Load and process new document
#        loader = PyPDFLoader(file_path)
#        documents = loader.load()
        
        # Split documents
#        text_splitter = RecursiveCharacterTextSplitter(
#            chunk_size=1000,
#            chunk_overlap=100,
#            length_function=len,
#        )
#        split_docs = text_splitter.split_documents(documents)
        
        # Add to vector store
#        vectorstore.add_documents(split_docs)
        
#        return {
#            "message": f"Successfully added {len(split_docs)} chunks from {file_path}",
#            "chunks_added": len(split_docs)
#        }
        
#    except Exception as e:
#        logger.error(f"Error adding document: {str(e)}")
#        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

if __name__ == "__main__":
    if not os.getenv("PINECONE_API_KEY"):
        os.environ["PINECONE_API_KEY"] = "pcsk_4bJoVS_RFnJwCjpEsJ6PcK5XArBJzXNMC1nTGRV9Z1YWrVHB6o1mNbZLSEZpL7APMGmTSn"
    if not os.getenv("PINECONE_ENV"):
        os.environ["PINECONE_ENV"] = "us-east-1"
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set. Please set it in your environment or .env file")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)