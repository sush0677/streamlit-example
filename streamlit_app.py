import streamlit as st
from fpdf import FPDF
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Initialize the AzureChatOpenAI model
model = AzureChatOpenAI(
    deployment_name="exq-gpt-35",
    azure_endpoint="https://exquitech-openai-2.openai.azure.com/",
    api_key="4f00a70876a542a18b30f13570248cdb",
    temperature=0,
    openai_api_version="2024-02-15-preview"
)
def run_sequential_chains(review):
    seq_chain = SequentialChain(chains=[
        LLMChain(llm=model, prompt=ChatPromptTemplate("Provide me with the following English text :\n{review}"), output_key="english_text"),
        LLMChain(llm=model, prompt=ChatPromptTemplate("Translate the following text into Arabic text :\n{english_text}"), output_key="Arabic_text"),
        LLMChain(llm=model, prompt=ChatPromptTemplate("Summarize the following text in Arabic language :\n{Arabic_text}"), output_key="final_plan")
    ],
    input_variables=['review'],
    output_variables=['english_text', 'Arabic_text', 'final_plan'],
    verbose=True)
    output = seq_chain(review)
    return output

# Streamlit app interface
st.title("Language Processing with LangChain and Azure")
review_text = st.text_area("Enter text to process:", "Type your review here...")
if st.button("Process Text"):
    output = run_sequential_chains(review_text)
    st.write("## Processed Outputs")
    st.write("### English Text")
    st.write(output['english_text'])
    st.write("### Arabic Text")
    st.write(output['Arabic_text'])
    st.write("### Arabic Summary")
    st.write(output['final_plan'])
else:
    st.write("Enter text and click the 'Process Text' button.")