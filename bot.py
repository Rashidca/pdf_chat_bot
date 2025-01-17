import streamlit as st
from PyPDF2 import PdfReader
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()
genai.configure(api_key=os.getenv("google_api_key"))

# Optimized PDF Text Extraction (using pdfplumber for speed)
def get_pdf_text(pdf_docs):
    text = ""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        texts = list(executor.map(process_pdf, pdf_docs))
    return ''.join(texts)

def process_pdf(pdf):
    with pdfplumber.open(pdf) as pdf_reader:
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Optimized text chunking function
def text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Optimized function to create vector store with batching
def chunks_vector(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # st.write("working2.....")
    
    # Create a list of documents (with 'page_content')
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    # st.write("working3.....")
    
    # Create the FAISS vector store from the list of documents
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    # st.write("working4.....")
    vector_store.save_local("FAISS_index")
    # st.write("working5.....")


# Cached QA chain to improve performance
@st.cache_resource
def convo_chain():
    prompt_template = """
    You are a helpful assistant with access to the following context:

    Answer the user's question as detailed as possible based on the information provided in the context. If the answer is not directly available, respond with "answer is not available." Do not provide incorrect or misleading information.
    context:\n{context}\n
    User's Question:\n{question}\n

    Answer:"""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Optimized user input handling
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("FAISS_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = convo_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Pdf_assistant:", response["output_text"])

# Main Streamlit app function
def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")

    st.markdown("# PDF Chatbot", unsafe_allow_html=True)
    st.sidebar.header("Navigation")

    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        progress_bar = st.progress(0)
        total_steps = 3  # Update to match the actual steps

        # Step 1: Extracting text
        st.write("Step 1: Extracting text from PDFs...")
        pdf_text = get_pdf_text(uploaded_files)
        st.success("Text extracted successfully!")
        progress_bar.progress(1 / total_steps)

        # Step 2: Creating vector store
        st.write("Step 2: Creating vector store...")
        chunks = text_chunks(pdf_text)
        st.write("working1.....")
        chunks_vector(chunks)
        st.success("Vector store created successfully!")
        progress_bar.progress(2 / total_steps)

        # Step 3: Handle user query
        user_question = st.text_input("Step 3: Ask a question about the uploaded PDFs:")
        if user_question:
            st.write("Generating response...")
            user_input(user_question)
            st.success("Response generated successfully!")
            progress_bar.progress(1)  # Completed progress at 100%

    if uploaded_files:
        st.download_button(
            label="Download Extracted Text",
            data=pdf_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

