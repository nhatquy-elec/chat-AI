from langchain.document_loaders import GitLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Tải dữ liệu từ repo GitHub
loader = GitLoader(repo_url='https://github.com/your-username/your-repo', branch='main')
documents = loader.load()

# Tạo embeddings và xây dựng bộ nhớ tìm kiếm
db = FAISS.from_documents(documents, OpenAIEmbeddings())

# Tạo chatbot với khả năng tìm kiếm dữ liệu
retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)

# Hỏi chatbot
question = "Nội dung chính của tài liệu là gì?"
answer = qa_chain.run(question)
print(answer)
