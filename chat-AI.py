# app.py
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

vectorstore = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_and_split(file_path):
    ext = file_path.rsplit('.', 1)[-1].lower()
    if ext == 'txt':
        loader = TextLoader(file_path)
    elif ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'docx':
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)


@app.route('/', methods=['GET', 'POST'])
def index():
    global vectorstore

    if request.method == 'POST':
        if 'file' in request.files:
            files = request.files.getlist('file')
            all_docs = []
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    all_docs.extend(load_and_split(filepath))

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            return render_template('index.html', message="Tải tài liệu thành công.")

        if 'question' in request.form and vectorstore:
            query = request.form['question']
            docs = vectorstore.similarity_search(query)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=query)
            return render_template('index.html', question=query, answer=answer)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
