import streamlit as st
from chatbot import load_documents, build_qa_chain

st.title("🤖 Chatbot AI với tài liệu GitHub")

# Load và xây dựng chatbot
docs = load_documents("data/")
qa_chain = build_qa_chain(docs)

# Nhập câu hỏi
query = st.text_input("Nhập câu hỏi của bạn:")
if query:
    answer = qa_chain.run(query)
    st.write("📘 Trả lời:")
    st.write(answer)
