import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Đặt tiêu đề ứng dụng
st.title("Chatbot từ tài liệu lưu trên GitHub")

# Đặt biến môi trường (nếu cần dùng các thư viện phụ trợ gọi OpenAI-style)
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# ✅ Embedding model từ OpenRouter
embedding_model = OpenAIEmbeddings(
    openai_api_key=st.secrets["openai_api_key"],
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/text-embedding-ada-002"
)

# ✅ LLM (LLAMA-4-Maverick) từ OpenRouter
llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=st.secrets["openai_api_key"],
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="meta-llama/llama-4-maverick:free"
)

# ✅ Load tài liệu và tạo FAISS DB
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
query = st.text_input("?_ Câu hỏi của bạn:")
if query:
    docs = db.similarity_search(query, k=3)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    st.success(answer)
