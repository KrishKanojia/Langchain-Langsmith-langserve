import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
load_dotenv()

## Load the GROQ key
os.environ['GROQ_API_KEY'] = os.getenv['GROQ_API_KEY']

st.title('Chatgroq with llama3 Demo')

llm = ChatGroq(model_name='llama3-8b-8192')

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on question
<context>
{context}
<context>
Questions:{input}
"""
)