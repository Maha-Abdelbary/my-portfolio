import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import os

# 1. Page Configuration
st.set_page_config(page_title="RAG PDF Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI PDF Assistant (Free RAG Chatbot)")
st.caption("Upload any PDF document and ask questions about its content in real-time.")

# Sidebar for File Upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a PDF and ask me anything about it."}]

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Cache Resource for Fast Loading
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# 2. Process Uploaded PDF
if uploaded_file:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing PDF & creating embeddings..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Load and split PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                splits = text_splitter.split_documents(docs)

                # Store vectors in FAISS
                vectorstore = FAISS.from_documents(splits, embeddings)

                st.session_state.vectorstore = vectorstore
                st.session_state.current_file = uploaded_file.name
                st.sidebar.success(f"Successfully processed `{uploaded_file.name}`!")

            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                os.remove(tmp_path)

# 3. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Handle User Input & Semantic Retrieval
if user_query := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if st.session_state.vectorstore is None:
        response_text = "Please upload a valid PDF file first."
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching document for matching text..."):
                # Semantic Similarity Search
                docs = st.session_state.vectorstore.similarity_search(user_query, k=3)
                
                if docs:
                    context_text = "\n\n".join([f"**From page {doc.metadata.get('page', 0) + 1}:**\n{doc.page_content}" for doc in docs])
                    response_text = f"Here are the relevant parts found in your document:\n\n{context_text}"
                else:
                    response_text = "No relevant information found in the uploaded document."

                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})