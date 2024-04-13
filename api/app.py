from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

app = FastAPI(
    title='Langchain Server',
    version="1.0",
    description="A Simple Api Server"
)

llm = Ollama(model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

prompt = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")
chain = prompt|llm

add_routes(
    app,
    chain,
    path='/poem'
)
if __name__ == "__main__":
    # uvicorn.run(
    #     app,
    #     host="localhost",
    #     port=8000)
    chain.invoke({"topic": "machine learning"})
   
        