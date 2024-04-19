import streamlit as st
import os
from azure.identity import ChainedTokenCredential, ManagedIdentityCredential, AzureCliCredential
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain, SequentialChain
import streamlit as st
from io import BytesIO
from fpdf import FPDF

# Assuming your Azure/OpenAI SDK and specific API classes/methods are properly defined and imported
# Example setup for model and chains
model = AzureChatOpenAI(
    deployment_name="exq-gpt-35",
    azure_endpoint="https://exquitech-openai-2.openai.azure.com/",
    api_key="your_api_key",
    temperature=0,
    openai_api_version="2024-02-15-preview"
)

template1 = "Provide me with the following English text :\n{review}"
prompt1 = ChatPromptTemplate.from_template(template1)
chain_1 = LLMChain(llm=model, prompt=prompt1, output_key="english_text")

template2 = "Translate the following text into Arabic text :\n{english_text}"
prompt2 = ChatPromptTemplate.from_template(template2)
chain_2 = LLMChain(llm=model, prompt=prompt2, output_key="Arabic_text")

template3 = "Summarize the following text in Arabic language :\n{Arabic_text}"
prompt3 = ChatPromptTemplate.from_template(template3)
chain_3 = LLMChain(llm=model, prompt=prompt3, output_key="final_plan")

seq_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3],
    input_variables=['review'],
    output_variables=['english_text', 'Arabic_text', 'final_plan'],
    verbose=True
)

def process_text(review):
    # Execute the sequential chain and return results
    results = seq_chain(review)
    return results['english_text'], results['Arabic_text'], results['final_plan']

def save_pdf(english, arabic, summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="English Text:", ln=True)
    pdf.multi_cell(0, 10, english)
    pdf.cell(200, 10, txt="Arabic Translation:", ln=True)
    pdf.multi_cell(0, 10, arabic)
    pdf.cell(200, 10, txt="Summarized Arabic Text:", ln=True)
    pdf.multi_cell(0, 10, summary)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

st.title("Text Processing App")
review_text = st.text_area("Enter your review text here:")

if st.button("Process Text"):
    if review_text:
        english_text, arabic_text, summary = process_text(review_text)
        st.write("English Text:", english_text)
        st.write("Arabic Translation:", arabic_text)
        st.write("Arabic Summary:", summary)

        pdf = save_pdf(english_text, arabic_text, summary)
        st.download_button(label="Download Processed Texts as PDF", data=pdf, file_name="processed_texts.pdf", mime='application/pdf')
    else:
        st.error("Please enter text to process.")

