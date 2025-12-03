import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant. Answer the user accurately."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    llm = ChatGroq(
        model=engine,
        groq_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    return answer

# App title
st.title("AI Buddy – Created by Bharat")

# Sidebar settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your API Key:", type="password")

# Select the Groq model
engine = "llama-3.1-8b-instant"

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.3, max_value=1.0, value=0.6)
max_tokens = st.sidebar.slider("Max Tokens", min_value=200, max_value=10000, value=800)

# User Interface
st.write("Ask anything to my buddy:")
user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.write(response)

elif user_input:
    st.write("Please enter Groq API KEY.")

else:
    st.write("Please provide your question.")
