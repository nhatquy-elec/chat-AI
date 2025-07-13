import os
import streamlit as st
import tempfile
from pathlib import Path

from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Cấu hình embedding model (qua OpenRouter)
embedding_model = OpenAIEmbeddings(
    openai_api_key=st.secrets["openai_api_key"],
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/text-embedding-ada-002"
)

# Cấu hình mô hình trả lời (LLM)
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=st.secrets["openai_api_key"],
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="mistralai/mistral-7b-instruct:free"
)

# Giao diện chính
st.title("📄 Chatbot từ tài liệu của bạn")

uploaded_files = st.file_uploader(
    "📂 Tải lên tài liệu (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

lang = st.selectbox("🌐 Chọn ngôn ngữ trả lời", ["Tiếng Việt", "English"])

# Hàm xử lý tài liệu upload
def load_uploaded_db(uploaded_files):
    docs = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            docs += PyPDFLoader(tmp_path).load()
        elif uploaded_file.name.endswith(".docx") or uploaded_file.name.endswith(".txt"):
            docs += UnstructuredFileLoader(tmp_path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    return FAISS.from_documents(splits, embedding_model)

# Khi người dùng upload
if uploaded_files:
    db = load_uploaded_db(uploaded_files)

    query = st.text_input("🤖 Câu hỏi của bạn:")

    if query:
        prefix = "Trả lời bằng tiếng Việt. " if lang == "Tiếng Việt" else ""
        full_query = prefix + query

        docs = db.similarity_search(query, k=3)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=full_query)
        st.success(answer)
