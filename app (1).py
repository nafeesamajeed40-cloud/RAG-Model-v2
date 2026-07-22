import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Banking Division RAG Assistant")
st.title("Banking Division RAG Assistant")
st.write("Ask questions and get answers sourced from internal manuals, circulars, and emails.")

@st.cache_resource
def load_pipeline():
    pdf_folder = "docs"
    documents = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embedding_model)

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="conversational",
        temperature=0.3,
        max_new_tokens=512
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        "Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return chain

qa_chain = load_pipeline()

question = st.text_input("Ask a question about the manuals, circulars, or emails")

if st.button("Submit") and question:
    with st.spinner("Thinking..."):
        answer = qa_chain.invoke(question)
    st.subheader("Answer")
    st.write(answer)
