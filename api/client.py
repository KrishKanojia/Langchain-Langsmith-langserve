import requests
import streamlit as st

def get_llama2_response(input_text):
    response = requests.get(
        "http://localhost:8000/chat",
        # json = {'input': {'topic': "machine"}}
        json={"text": "Hello"}
    )
    return response.json()
    # return response.json()['output']


# Streamlit Framework
# st.title('Langchain Poem Demo With Ollama API')
# input_text = st.text_input("Search the topic u want")

# if input_text:
#     st.write(get_llama2_response(input_text))
def get_stream(query: str):
    s = requests.Session()
    with s.get(
        "http://localhost:8000/chat",
        stream=True,
        json={"text": "hello"}
    ) as r:
        for line in r.iter_content():
            print(line.decode("utf-8"), end="")

if __name__ == "__main__":
    get_stream("flower")