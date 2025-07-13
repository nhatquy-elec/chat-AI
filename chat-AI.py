import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

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
    model_name="meta-llama/llama-4-maverick:free"
)

# ✅ Nạp và xử lý tài liệu
@st.cache_resource
def load_db():
    docs = []
    for filename in os.listdir("data"):
        path = os.path.join("data", filename)
        if filename.endswith(".pdf"):
            docs += PyPDFLoader(path).load()
        elif filename.endswith(".docx") or filename.endswith(".txt"):
            docs += UnstructuredFileLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    db = FAISS.from_documents(splits, embedding_model)
    return db

db = load_db()

# ✅ Giao diện đặt câu hỏi
query = st.text_input("❓ Câu hỏi của bạn:")
if query:
    docs = db.similarity_search(query, k=3)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    st.success(answer)
