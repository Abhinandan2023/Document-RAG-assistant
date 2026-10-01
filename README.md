📄 Document RAG Assistant

A dynamic Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about their content using natural language.

 Features:

 Upload PDF, TXT, and DOCX documents
 Semantic search using vector embeddings
 ChromaDB for vector storage
 MMR-based document retrieval
 Mistral AI for answer generation
 Interactive Streamlit chat interface
 Upload a new document and start a new conversation
 Architecture
Document Upload
      ↓
Document Loading
      ↓
Text Chunking
      ↓
HuggingFace Embeddings
      ↓
ChromaDB
      ↓
MMR Retrieval
      ↓
Relevant Context
      ↓
Mistral AI
      ↓
Answer

 Tech Stack:

Python · Streamlit · LangChain · Mistral AI · HuggingFace · ChromaDB

 Run Locally:

Clone the repository:

git clone https://github.com/Abhinandan2023/document-rag-assistant.git
cd document-rag-assistant

Install dependencies:

pip install -r requirements.txt

Create a .env file:

MISTRAL_API_KEY=your_api_key

Run the application:

streamlit run app.py
📁 Project Structure
document-rag-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

 Security:

API keys are stored in .env and excluded from Git using .gitignore.

 Future Improvements:
Multiple document support
Source/page citations
Streaming responses
Additional document formats
Cloud deployment

 Author:

Abhinandan Maity