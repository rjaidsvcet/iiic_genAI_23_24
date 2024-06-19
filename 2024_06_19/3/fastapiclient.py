import streamlit as st
import requests

def getOllamaResponse (inputText):
    response = requests.post (
        url = 'http://localhost:8000/essay/invoke',
        json = {'input' : {'topic' : inputText}}
    )

    return response.json ()['output']['content']

st.title ('Testing FastAPI Application')
inputText = st.text_input ('Enter topic for your essay')

if inputText:
    st.write (getOllamaResponse (inputText))