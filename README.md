# 📄💬 RAG Chat With Documents

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload documents and chat with them using natural language. Powered by **LLaMA 3**, **LangChain**, and **Streamlit**, this app processes documents, retrieves relevant information, and generates accurate, context-aware answers—all locally on your machine.

---

## 🚀 Features

- 📂 Upload multiple documents: PDF, TXT, DOCX
- 🔍 Automatically splits documents into manageable text chunks
- 🧠 Generates sentence embeddings using HuggingFace models
- 🗃️ Stores and retrieves chunks using FAISS vector search
- 🤖 Interacts with your documents using LLaMA 3 (via Ollama)
- 💬 Clean and simple chat interface built with Streamlit
- 🔐 100% local – your data never leaves your device

---

## 🛠️ Tech Stack Overview

| Category | Tool | Purpose |
|----------|------|---------|
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Lightweight web app for file upload and chat |
| **Language Model** | [LLaMA 3 via Ollama](https://ollama.com/library/llama3) | Local LLM that generates intelligent responses |
| **RAG Framework** | [LangChain](https://www.langchain.com/) | Ties together retrieval and generation pipelines |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) | Fast and efficient similarity search on embeddings |
| **Embeddings** | [HuggingFace Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Converts text chunks into semantic vectors |
| **Document Loaders** | LangChain Community Loaders | Parses PDF, TXT, and DOCX files into readable format |

---

## 📦 Installation

> ⚠️ Python 3.10 or above is recommended

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/rag-chat-app.git
   cd rag-chat-app

2. **Install dependencies**
pip install -r requirements.txt

3. **Install and run Ollama**
Download from: https://ollama.com/download

Pull the model:
ollama pull llama3

4. **Usage**
streamlit run app.py

📁 Supported File Types
1 .pdf
2 .txt
3 .docx

Each file should be less than 5MB for optimal performance.

**🧠 How It Works**

*Upload Documents*
→ Files are saved temporarily and loaded.

*Text Splitting*
→ Uses RecursiveCharacterTextSplitter to break documents into chunks.

*Embedding*
→ Each chunk is embedded with sentence-transformers/all-MiniLM-L6-v2.

*Vector Store (FAISS)*
→ Embeddings are stored and queried to find relevant chunks.

*RAG + LLM (Ollama)*
→ Relevant context is passed to llama3 to generate an answer.

**ScreenShots**
![Upload screen](screenshots/upload_screen.png)
![Chat screen](screenshots/chat.png)

**NOTE**
Only the first 100 chunks are used for faster testing.

Ollama must be installed and running in the background to use the LLaMA 3 model.

**📜 License**
This project is for educational use only. Feel free to adapt it for your own learning or internal use.

Credits:
Using open-source tools from LangChain, HuggingFace, Meta AI, and the Python community.