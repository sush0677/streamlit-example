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
from langchain.chains import SequentialChain
import streamlit as st
from fpdf import FPDF
import os

# Assuming the SequentialChain and LLMChain have been set up as described in your code snippet
# Create an instance of the AzureChatOpenAI model
model = AzureChatOpenAI(
    deployment_name="exq-gpt-35",
    azure_endpoint="https://exquitech-openai-2.openai.azure.com/",
    api_key="4f00a70876a542a18b30f13570248cdb",
    temperature=0,
    openai_api_version="2024-02-15-preview"
)

template1 = "Provide me with the following English text :\n{review}"
prompt1 = ChatPromptTemplate.from_template(template1)
chain_1 = LLMChain(llm=model,
                     prompt=prompt1,
                     output_key="english_text")
template2 = "Translate the following text into Arabic text  :\n{english_text}"
prompt2 = ChatPromptTemplate.from_template(template2)
chain_2 = LLMChain(llm=model,
                     prompt=prompt2,
                     output_key="Arabic_text")
template3 = "Summarize the following text in Arabic language :\n{Arabic_text}"
prompt3 = ChatPromptTemplate.from_template(template3)
chain_3 = LLMChain(llm=model,
                     prompt=prompt3,
                     output_key="final_plan")
seq_chain = SequentialChain(chains=[chain_1,chain_2,chain_3],
                            input_variables=['review'],
                            output_variables=['english_text','Arabic_text','final_plan'],
                            verbose=True)
                            
# Function to handle file upload and text input
def get_input_text():
    input_text = st.text_area("Enter text to process or upload a file:", height=150)
    uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])
    if uploaded_file is not None:
        # Read the content of the uploaded file
        input_text = str(uploaded_file.read(), 'utf-8')
    return input_text

# Function to create PDF from text data
def create_pdf(english_text, arabic_text, summarized_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="English Text:", ln=True)
    pdf.multi_cell(0, 10, english_text)
    pdf.cell(200, 10, txt="Arabic Translation:", ln=True)
    pdf.multi_cell(0, 10, arabic_text)
    pdf.cell(200, 10, txt="Summarized Arabic Text:", ln=True)
    pdf.multi_cell(0, 10, summarized_text)
    filename = 'Text_Summary.pdf'
    pdf.output(filename)
    return filename

def main():
    st.title("Text Processing App")

    # Retrieve user input
    user_input = get_input_text()

    if st.button("Translate to Arabic"):
        if user_input:
            # Process translation
            english_text = seq_chain.run(review=user_input)['english_text']
            st.text_area("Translated Text:", english_text, height=150)
        else:
            st.warning("Please enter text or upload a file to translate.")

    if st.button("Summarize in Arabic"):
        if user_input:
            # Process summarization
            result = seq_chain.run(review=user_input)
            arabic_text = result['Arabic_text']
            summarized_text = result['final_plan']
            st.text_area("Arabic Summary:", summarized_text, height=150)
        else:
            st.warning("Please enter text or upload a file to summarize.")

    if st.button("Download PDF"):
        if user_input:
            # Generate PDF
            result = seq_chain.run(review=user_input)
            english_text = result['english_text']
            arabic_text = result['Arabic_text']
            summarized_text = result['final_plan']
            pdf_filename = create_pdf(english_text, arabic_text, summarized_text)
            with open(pdf_filename, "rb") as file:
                btn = st.download_button(
                    label="Download Text Summary",
                    data=file,
                    file_name=pdf_filename,
                    mime="application/octet-stream"
                )
        else:
            st.warning("Please enter text or upload a file to generate PDF.")

if __name__ == "__main__":
    main()
