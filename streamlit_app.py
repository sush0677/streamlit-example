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

def main():
    st.title("Text Processing App")
    user_input = get_input_text()

    if st.button("Translate to Arabic"):
        if user_input:
            try:
                results = seq_chain.run(review=user_input)
                english_text = results['english_text']
                st.text_area("Translated Text:", english_text, height=150)
            except Exception as e:
                st.error(f"Failed to translate due to: {str(e)}")

    if st.button("Summarize in Arabic"):
        if user_input:
            try:
                results = seq_chain.run(review=user_input)
                arabic_text = results['Arabic_text']
                summarized_text = results['final_plan']
                st.text_area("Arabic Summary:", summarized_text, height=150)
            except Exception as e:
                st.error(f"Failed to summarize due to: {str(e)}")

    if st.button("Download PDF"):
        if user_input:
            try:
                results = seq_chain.run(review=user_input)
                english_text = results['english_text']
                arabic_text = results['Arabic_text']
                summarized_text = results['final_plan']
                pdf_filename = create_pdf(english_text, arabic_text, summarized_text)
                with open(pdf_filename, "rb") as file:
                    btn = st.download_button(
                        label="Download Text Summary",
                        data=file,
                        file_name=pdf_filename,
                        mime="application/octet-stream"
                    )
            except Exception as e:
                st.error(f"Failed to generate PDF due to: {str(e)}")

if __name__ == "__main__":
    main()
    