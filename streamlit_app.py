import streamlit as st
from fpdf import FPDF
from io import BytesIO
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
)# Templates and Chains
template1 = "Provide me with the following English text :\n{review}"
prompt1 = ChatPromptTemplate.from_template(template1)
chain_1 = LLMChain(llm=model, prompt=prompt1, output_key="english_text")

template2 = "Translate the following text into Arabic text  :\n{english_text}"
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
# Streamlit User Interface
st.title("Text Translation and Summary App")

# File upload and text input
uploaded_file = st.file_uploader("Upload your file", type=['txt'])
if uploaded_file is not None:
    raw_text = str(uploaded_file.read(), 'utf-8')  # Convert to string
else:
    raw_text = st.text_area("Or enter the text to translate and summarize", height=150)

# Process and display buttons
if st.button("Show Original Text"):
    if raw_text:
        st.write("Original Text:")
        st.write(raw_text)
    else:
        st.error("No text provided.")

if st.button("Translate to Arabic"):
    if raw_text:
        results = seq_chain({raw_text})
        st.write("Translated Arabic Text:")
        st.write(results["Arabic_text"])
    else:
        st.error("Please provide text to translate.")

if st.button("Summarization in Arabic"):
    if raw_text:
        results = seq_chain({raw_text})
        st.write("Summary in Arabic:")
        st.write(results["final_plan"])
    else:
        st.error("Please provide text to summarize.")

# Function to create and download a PDF
def create_downloadable_pdf(english_text, arabic_text, summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.add_page()

    # Adding English text
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, 'English Text:', 0, 1)
    pdf.multi_cell(0, 10, english_text)

    # Adding Arabic translation
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, 'Arabic Translated Text:', 0, 1)
    pdf.multi_cell(0, 10, arabic_text)

    # Adding Arabic summary
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, 'Arabic Summary:', 0, 1)
    pdf.multi_cell(0, 10, summary_text)

    pdf_output = BytesIO()
    pdf.output(pdf_output, 'F')
    pdf_output.seek(0)  # Move to the beginning of the BytesIO buffer
    return pdf_output

# Button to trigger PDF download
if st.button("Download PDF") and 'results' in locals():
    results = seq_chain({raw_text})
    pdf_bytes = create_downloadable_pdf(results["english_text"], results["Arabic_text"], results["final_plan"])
    st.download_button(label="Download PDF", data=pdf_bytes, file_name="translated_summary.pdf", mime="application/pdf")