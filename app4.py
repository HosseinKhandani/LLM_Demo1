import streamlit as st
import os
import tempfile
import io
import shutil
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# مسیر پایگاه داده برداری
VECTORSTORE_PATH = "./chroma_db"
MODEL_NAME = "partai/dorna-llama3:latest"

# تابع بارگذاری اسناد از فایل‌های TXT, PDF و Word
def load_documents(files):
    docs = []
    for file in files:
        try:
            file_extension = file.name.split('.')[-1].lower()
            file_content = file.read()
            if file_extension == "txt":
                # برای فایل‌های متنی، محتوا را مستقیم خوانده و تبدیل به سند می‌کنیم
                text = file_content.decode("utf-8", errors="replace")
                docs.append(Document(page_content=text, metadata={"source": file.name}))
            elif file_extension == "pdf":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                loader = PyPDFLoader(temp_file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = file.name
                docs.extend(loaded_docs)
                os.unlink(temp_file_path)
            elif file_extension in ["docx", "doc"]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                loader = Docx2txtLoader(temp_file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = file.name
                docs.extend(loaded_docs)
                os.unlink(temp_file_path)
            else:
                st.warning(f"فرمت فایل {file_extension} پشتیبانی نمی‌شود.")
        except Exception as e:
            st.error(f"خطا در بارگذاری فایل {file.name}: {str(e)}")
    return docs

# تقسیم اسناد به قطعات کوچکتر
def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    return text_splitter.split_documents(docs)

# مقداردهی اولیه مؤلفه‌ها: ساخت پایگاه داده برداری و ایجاد زنجیره پرسش و پاسخ
def initialize_components(docs, temperature, k):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    # اگر پایگاه داده وجود داشته باشد، اسناد جدید به آن اضافه می‌شوند؛ در غیر این صورت ایجاد می‌شود
    if os.path.exists(VECTORSTORE_PATH) and os.path.isdir(VECTORSTORE_PATH) and os.listdir(VECTORSTORE_PATH):
        vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)
        if docs:
            vectorstore.add_documents(docs)
    else:
        vectorstore = Chroma.from_documents(
            docs, 
            embedding=embeddings, 
            persist_directory=VECTORSTORE_PATH
        )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = Ollama(model=MODEL_NAME, temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# تابع اصلی اپلیکیشن
def main():
    st.set_page_config(page_title="اپلیکیشن تحلیل اطلاعات متن", layout="wide")
    st.title("اپلیکیشن تحلیل اطلاعات متن")
    st.markdown("این اپلیکیشن امکان جستجو و تحلیل اطلاعات را بر اساس ورودی‌های TXT, PDF و Word فراهم می‌کند.")
    
    with st.sidebar:
        st.header("تنظیمات")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
        k = st.number_input("تعداد اسناد مرجع (K)", min_value=1, value=3, step=1)
        files = st.file_uploader("آپلود فایل‌های ورودی (TXT, PDF, DOCX/DOC)", type=["txt", "pdf", "docx", "doc"], accept_multiple_files=True)
    
    if not files:
        st.info("لطفاً فایل‌های ورودی را بارگذاری کنید.")
        return
    
    # بارگذاری و پردازش اسناد
    docs = load_documents(files)
    if docs:
        docs = split_documents(docs)
        st.success("اسناد با موفقیت بارگذاری و پردازش شدند!")
    else:
        st.error("هیچ سندی از فایل‌های ورودی استخراج نشد.")
        return
    
    qa_chain = initialize_components(docs, temperature, k)
    
    query = st.text_input("سوال خود را وارد کنید:")
    if query:
        with st.spinner("در حال تولید پاسخ..."):
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            st.markdown("### پاسخ:")
            st.write(answer)
            if "source_documents" in result:
                st.markdown("#### منابع:")
                for doc in result["source_documents"]:
                    st.write(f"- {doc.metadata.get('source', 'نامشخص')}")
    
if __name__ == "__main__":
    main()
