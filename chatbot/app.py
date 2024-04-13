from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # Output Parser whenever LLM give any kind of response

import streamlit as st
import os
from dotenv import load_dotenv


# LangSmith Tracing
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question: {question}")
    ]
)

# Ollama LLM
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

# Streamlit Framework
st.title('Langchain Demo With Ollama API')
input_text = st.text_input("Search the topic u want")

if input_text:
    st.write_stream(chain.stream({'question': input_text}))
