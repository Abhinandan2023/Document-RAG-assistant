import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Page configuration
# ============================================================

st.set_page_config(
    page_title="Document RAG Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document RAG Assistant")
st.caption("Upload a document and ask questions about it.")


# ============================================================
# 3. Initialize session state
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# 4. Load embedding model
# ============================================================

@st.cache_resource
def load_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# 5. Load LLM
# ============================================================

@st.cache_resource
def load_llm():

    return ChatMistralAI(
        model="codestral-2508"
    )


llm = load_llm()


# ============================================================
# 6. Prompt
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say exactly:

"I could not find the answer in the document."

Do not use outside knowledge.
"""
        ),
        (
            "human",
            """Context:

{context}

Question:

{question}
"""
        )
    ]
)


# ============================================================
# 7. Load document
# ============================================================

def load_document(uploaded_file):

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name


    try:

        if extension == ".pdf":

            loader = PyPDFLoader(
                temp_path
            )

        elif extension == ".txt":

            loader = TextLoader(
                temp_path,
                encoding="utf-8"
            )

        elif extension == ".docx":

            loader = Docx2txtLoader(
                temp_path
            )

        else:

            raise ValueError(
                "Unsupported file type."
            )


        documents = loader.load()

        return documents

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 8. Create retriever
# ============================================================

def create_retriever(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    return retriever


# ============================================================
# 9. Upload document
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=[
        "pdf",
        "txt",
        "docx"
    ]
)


# ============================================================
# 10. Process document ONLY when a new document is uploaded
# ============================================================

if uploaded_file is not None:

    new_file = (
        st.session_state.file_name
        != uploaded_file.name
    )


    if new_file:

        with st.spinner(
            "Processing document..."
        ):

            try:

                # Load document
                documents = load_document(
                    uploaded_file
                )

                # Create retriever
                retriever = create_retriever(
                    documents
                )

                # Store retriever
                st.session_state.retriever = (
                    retriever
                )

                # Store filename
                st.session_state.file_name = (
                    uploaded_file.name
                )

                # Clear previous chat
                st.session_state.messages = []

                st.success(
                    f"✅ {uploaded_file.name} is ready!"
                )

            except Exception as e:

                st.error(
                    f"Error processing document: {e}"
                )


# ============================================================
# 11. Show current document
# ============================================================

if st.session_state.retriever is not None:

    st.success(
        f"📄 Current document: "
        f"{st.session_state.file_name}"
    )


# ============================================================
# 12. Display chat history
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 13. Chat input
# ============================================================

if st.session_state.retriever is not None:

    query = st.chat_input(
        "Ask something about your document..."
    )

    if query:

        # ------------------------------------
        # User message
        # ------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):

            st.markdown(query)


        # ------------------------------------
        # Retrieve relevant chunks
        # ------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching document..."
            ):

                docs = (
                    st.session_state.retriever
                    .invoke(query)
                )


            # --------------------------------
            # Create context
            # --------------------------------

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )


            # --------------------------------
            # Create prompt
            # --------------------------------

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )


            # --------------------------------
            # Generate answer
            # --------------------------------

            with st.spinner(
                "Generating answer..."
            ):

                response = llm.invoke(
                    final_prompt
                )

                answer = response.content


            st.markdown(answer)


        # ------------------------------------
        # Save assistant response
        # ------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


else:

    st.info(
        "👆 Upload a document to start chatting."
    )
