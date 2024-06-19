from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langserve import add_routes
from fastapi import FastAPI
import uvicorn

app = FastAPI (
    title = 'Langchain Server',
    version = '1.0',
    description = 'Simple server that sesrves langchain'
)

llm = ChatOllama (model='llama3')

prompt = ChatPromptTemplate.from_template ('Provide an essay of 100 words on given {topic} respectively')

add_routes (
    app = app,
    runnable = prompt | llm,
    path = '/essay'
)

if __name__ == '__main__':
    uvicorn.run (app = app, host = 'localhost', port = 8000)