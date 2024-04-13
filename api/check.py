from fastapi import FastAPI, Body
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
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools, AgentType, initialize_agent
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from pydantic import BaseModel
import os
import asyncio

prompt = ChatPromptTemplate.from_template("Write me a poem about {topic} with 20 words")


# memory =  ConversationBufferWindowMemory(
#     memory_key="chat_history",
#     k=5,
#     return_messages=True,
#     output_key="output"
# )

# tools = load_tools(["llm-math"], llm=llm)

# agent = initialize_agent(
#     agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
#     tools=tools,
#     llm=llm,
#     memory=memory,
#     verbose=True,
#     max_iterations=3,
#     early_stopping_method="generate",
#     handle_parsing_errors=True,
#     return_intermediate_steps=False
# )


# request input format
class Query(BaseModel):
    text: str

app = FastAPI()


llm = Ollama(model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

prompt = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")

chain = prompt|llm

agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=[],
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    # memory=memory,
    return_intermediate_steps=False
)
@app.get("/health")
async def health():
    return {"status": "Hello"}


async def run_call(query: str, stream_it: AsyncIteratorCallbackHandler):
    # agent.agent.llm_chain.llm.callbacks = [stream_it]
    # now query
    await agent.acall(inputs={"input": query})

async def create_gen(query: str, stream_it: AsyncIteratorCallbackHandler):
    task = asyncio.create_task(run_call(query, stream_it))
    async for token in stream_it.aiter():
        yield token
    await task


@app.get("/chat")
async def chat(query: Query = Body(...)):
    # response = chain.stream({"topic": query.text})
    stream_it = AsyncIteratorCallbackHandler()
    gen = create_gen(query.text, stream_it)
    return StreamingResponse(gen, media_type="text/event-stream")



if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8000)
    # chain.invoke({"topic": "machine learning"})
    # abc = "What is the square root of 71?"
    # print(agent(abc))

    