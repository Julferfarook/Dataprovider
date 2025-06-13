import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import tempfile

# 🔐 Set Gemini API key directly
GOOGLE_API_KEY = "AIzaSyCsINRQVql0DWC3uZiWmA47ZLwS-EGkdPE"  # Replace with your actual key

# Initialize Gemini LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# 🧠 Define ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that answers questions using only the provided context."),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}")
])

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit UI setup
st.set_page_config(page_title="📄 RAG with Gemini", page_icon="📄")
st.title("📄 RAG (Retrieval-Augmented Generation) with Gemini")

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
question = st.text_input("Ask a question based on the PDF:")

# Sidebar: Display chat history
with st.sidebar:
    st.subheader("🕘 Chat History")
    if st.session_state.chat_history:
        for idx, (q, a) in enumerate(st.session_state.chat_history[::-1]):
            st.markdown(f"**Q:** {q}\n\n**A:** {a}\n\n---")
    else:
        st.info("No chat history yet.")

if uploaded_file and question:
    if st.button("Ask Question"):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            # Load and split PDF
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(pages)

            # Create FAISS vector store
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()

            # Perform retrieval
            retrieved_docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])

            # RAG-style prompt chaining
            chain: Runnable = prompt | llm
            response = chain.invoke({"context": context, "question": question})

            # Save to session chat history
            st.session_state.chat_history.append((question, response.content))

            # Display result
            st.success("✅ Answer generated successfully!")
            st.markdown("### 📌 Answer:")
            st.markdown(response.content)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

elif uploaded_file and not question:
    st.info("📥 PDF uploaded. Now enter a question above.")
elif not uploaded_file:
    st.info("📄 Please upload a PDF file to get started.")
