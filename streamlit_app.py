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

# Function to handle file upload and text input
def get_input_text():
    input_text = st.text_area("Enter text to process or upload a file:", height=150)
    uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])
    if uploaded_file is not None:
        input_text = str(uploaded_file.read(), 'utf-8')
    return input_text

# Function to create a PDF from text data
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

# Run the sequential chain function
def run_sequential_chains(review):
    # Setup for sequential chains
    template1 = "Provide me with the following English text :\n{review}"
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain_1 = LLMChain(llm=model, prompt=prompt1, output_key="english_text")

    template2 = "Translate the following text into Arabic text :\n{english_text}"
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain_2 = LLMChain(llm=model, prompt=prompt2, output_key="Arabic_text")

    template3 = "Summarize the following text in Arabic language :\n{Arabic_text}"
    prompt3 = ChatPromptTemplate.from_template(template3)
    chain_3 = LLMChain(llm=model, prompt=prompt3, output_key="final_plan")

    seq_chain = SequentialChain(chains=[chain_1, chain_2, chain_3],
                                input_variables=['review'],
                                output_variables=['english_text', 'Arabic_text', 'final_plan'],
                                verbose=True)
    results = seq_chain(review)
    return results

def main():
    st.title("Text Processing App")
    user_input = get_input_text()
    review = user_input
    results = run_sequential_chains(review=review)
    English= print(results['english_text'])
    Arabic= print(results['Arabic_text'])
    Summarize= print(results['final_plan'])

    if user_input:
        if st.button("Translate to Arabic"):
            arabic_text = Arabic
            st.text_area("Arabic Translation:", arabic_text, height=150)

        if st.button("Summarize in Arabic"):
            summarized_text = Summarize
            st.text_area("Arabic Summary:", summarized_text, height=150)

        if st.button("Download PDF"):
            
            pdf_filename = create_pdf(English, Arabic, Summarize)
            with open(pdf_filename, "rb") as file:
                st.download_button("Download Text Summary", file, file_name=pdf_filename, mime="application/octet-stream")

        else:
            print("Please enter text or upload a file to proceed.")

if __name__ == "__main__":
    main()