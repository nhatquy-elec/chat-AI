import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Load API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

st.title("📄 Chatbot từ tài liệu lưu trên GitHub")

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
    db = FAISS.from_documents(splits, OpenAIEmbeddings())
    return db

db = load_db()

query = st.text_input("🧠 Câu hỏi của bạn:")
if query:
    docs = db.similarity_search(query, k=3)
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    st.success(answer)
