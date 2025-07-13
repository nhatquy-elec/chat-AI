import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

from langchain.docstore.document import Document
import tempfile


# 🚀 Tiêu đề giao diện
st.title("📄 Chatbot từ tài liệu GitHub – dùng OpenRouter")

# ✅ Khởi tạo mô hình nhúng văn bản (không dùng API)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ✅ Khởi tạo mô hình LLM từ OpenRouter
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=st.secrets["openai_api_key"],
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="mistralai/mistral-7b-instruct:free"
)

# ✅ Nạp và xử lý tài liệu
@st.cache_resource
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


db = load_uploaded_db()

# ✅ Giao diện đặt câu hỏi
query = st.text_input("❓ Câu hỏi của bạn:")
if query:
    docs = db.similarity_search(query, k=3)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    st.success(answer)
