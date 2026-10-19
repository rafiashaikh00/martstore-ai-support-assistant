# ==========================================
# HOW THIS FILE IS CONNECTED
# ==========================================
#
#   polices/*.txt  --(section 3)-->  documents
#   documents      --(section 4)-->  chunks
#   chunks         --(section 5,6)-> embeddings -> Chroma DB (chroma_db folder)
#   Chroma DB      --(section 7)-->  retriever
#   question       --(section 10)--> retriever -> context -> prompt -> llm -> answer
#
#   data_back.py (policy_node) calls answer_policy_question() in section 10
#   and gets back {"answer": ..., "sources": [...]}


# ==========================================
# 1. IMPORTS
# ==========================================

import os
from pathlib import Path
from langchain_core.messages import AIMessage
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ==========================================
# 2. LOAD ENVIRONMENT VARIABLES AND PATHS
# ==========================================

# The .env file is OUTSIDE this folder (in a parent folder).
# find_dotenv() searches upward from this file until it finds it.
# If your .env is somewhere else, write its full path instead:
# load_dotenv(r"D:\gen ai\langraph\.env")
load_dotenv(find_dotenv())

# Folder containing this rag.py file
BASE_DIR = Path(__file__).resolve().parent

# Folder containing martStore policy text files
POLICY_PATH = BASE_DIR / "polices"

# Persistent Chroma database folder
CHROMA_PATH = BASE_DIR / "chroma_db"


# ==========================================
# 3. LOAD POLICY DOCUMENTS
# ==========================================

if not POLICY_PATH.is_dir():
    raise FileNotFoundError(f"Policy folder not found: {POLICY_PATH}")

# Load all .txt policy files from the polices folder
loader = DirectoryLoader(
    str(POLICY_PATH),
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

documents = loader.load()

if not documents:
    raise ValueError(f"No policy text files found in: {POLICY_PATH}")


# ==========================================
# 4. SPLIT DOCUMENTS INTO CHUNKS
# ==========================================

# Split long policies into smaller overlapping pieces
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# CONNECTION: "documents" from section 3 -> "chunks" used in section 6
chunks = text_splitter.split_documents(documents)


# ==========================================
# 5. CREATE EMBEDDINGS
# ==========================================

# Convert policy text into vectors for semantic search
# CONNECTION: used by the vector store (section 6)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ==========================================
# 6. CREATE PERSISTENT CHROMA VECTOR STORE
# ==========================================

# Open the existing database or create a new collection
vector_store = Chroma(
    collection_name="martstore_policies",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH)
)

# Add policy chunks only when the collection is empty
if vector_store._collection.count() == 0:
    vector_store.add_documents(chunks)


# ==========================================
# 7. CREATE RETRIEVER
# ==========================================

# Retrieve relevant policy chunks for each question
# CONNECTION: used inside answer_policy_question (section 10)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "fetch_k": 10,
        "k": 3,
        "lambda_mult": 0.5
    }
)


# ==========================================
# 8. INITIALIZE CHATGROQ
# ==========================================

# Use the Groq API key from the .env file
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY was not found in the .env file.")

# CONNECTION: this llm is separate from the llm in data_back.py,
# it is used only to write the final policy answer (section 10)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key
)


# ==========================================
# 9. CREATE POLICY-BASED PROMPT
# ==========================================

# The assistant must answer from retrieved policy context only
# CONNECTION: {context} = retrieved chunks, {question} = customer question
prompt = ChatPromptTemplate.from_template("""
You are a customer support assistant for martStore.

Answer using only the policy context provided below.
Do not invent policies or assume missing information.

If the context does not contain enough information, say:
"The information is not available in the provided policies."

Policy context:
{context}

Customer question:
{question}

Answer:
""")


# ==========================================
# 10. ANSWER POLICY QUESTIONS
# ==========================================

# CONNECTION: called by policy_node in data_back.py
def answer_policy_question(question: str) -> dict:
    # Find policy chunks relevant to the question
    retrieved_documents = retriever.invoke(question)

    # Combine retrieved chunks into one context
    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

    # Collect the names of the source policy files
    sources = sorted({
        Path(document.metadata["source"]).name
        for document in retrieved_documents
        if document.metadata.get("source")
    })

    # Generate an answer using the retrieved policy context
    response_chain = prompt | llm | StrOutputParser()

    answer = response_chain.invoke({
        "context": context,
        "question": question
    })

    # Return both the answer and its source filenames
    return {
        "answer": answer,
        "sources": sources
    }


# ==========================================
# 11. VERIFY THE RAG PIPELINE
# ==========================================

# Run this check only when rag.py is executed directly.
# Importing this file from LangGraph will not run this test.
if __name__ == "__main__":
    result = answer_policy_question("What is the return policy?")

    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
