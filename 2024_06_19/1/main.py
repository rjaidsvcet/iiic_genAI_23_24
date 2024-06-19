from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

inputPrompt = PromptTemplate (
    input_variables = ['name'],
    template = 'Tell me about the celebrity {name}'
)

llm = Ollama (model = 'llama3')

chain = LLMChain (
    llm = llm,
    prompt = inputPrompt,
    verbose = True
)

response = chain.run ({'name' : 'Marlon Brando'})
print (response)