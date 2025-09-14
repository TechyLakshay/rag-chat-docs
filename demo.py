import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=api_key
)

# Prompt template
prompt = """
You are a helpful assistant.
Answer the question based on the provided context.
Context: {context}
Question: {question}

do's-
-answer query in more detailed and structured way
-tone must be professional
-answer must be accurate

dont's-
-never answer query out of the context just say 'I DON'T KNOW'
"""
st.title("📄Chat with Your PDF (RAG + Gemini")
uploader = st.file_uploader("Upload a PDF file", type=["pdf"])
user_question = st.text_input("", placeholder="Ask something about the PDF...")

if uploader and user_question:
    with st.spinner("Processing..."):
        try:
            
            # Save uploaded file
            temp_path = "temp_uploaded.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploader.read())

            # Load PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            if not documents:
                st.error("❌ The PDF seems empty or unreadable. Please upload a valid PDF.")
                st.stop()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)
            texts = text_splitter.split_documents(documents)

            if not texts:
                st.error("❌ No text extracted from the PDF.")
                st.stop()

            # Embeddings + Vectorstore
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(texts, embeddings)

            # QA chain
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt}
            )
            vectorstore.save_local("attention_index")
            # Run query
            response = qa_chain.run(user_question)
            st.success(f"✅ Answer: {response}")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

