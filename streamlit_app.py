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
def run_sequential_chains(user_input):
    # Setup for sequential chains
    template1 = "Provide me with the following English text :\n{user_input}"
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain_1 = LLMChain(llm=model, prompt=prompt1, output_key="english_text")

    template2 = "Translate the following text into Arabic text :\n{english_text}"
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain_2 = LLMChain(llm=model, prompt=prompt2, output_key="Arabic_text")

    template3 = "Summarize the following text in Arabic language :\n{Arabic_text}"
    prompt3 = ChatPromptTemplate.from_template(template3)
    chain_3 = LLMChain(llm=model, prompt=prompt3, output_key="final_plan")

    seq_chain = SequentialChain(chains=[chain_1, chain_2, chain_3],
                                input_variables=['user_input'],  # Use the variable directly in your template
                                output_variables=['english_text', 'Arabic_text', 'final_plan'],
                                verbose=True)
    return seq_chain.run(user_input=user_input)

def main():
    st.title("Text Processing App")
    user_input = get_input_text()

    if user_input:
        results = st.session_state.get('results')
        if st.button("Process Text"):
            results = run_sequential_chains(user_input)
            st.session_state['results'] = results  # Store results in session state
            st.success("Processing complete!")

        if st.button("Show Arabic Translation") and results:
            arabic_text = results.get('Arabic_text', "No Arabic text found.")
            st.text_area("Arabic Translation:", arabic_text, height=150)

        if st.button("Show Arabic Summary") and results:
            summarized_text = results.get('final_plan', "No summary available.")
            st.text_area("Arabic Summary:", summarized_text, height=150)

        if st.button("Download PDF") and results:
            english_text = results.get('english_text', "")
            arabic_text = results.get('Arabic_text', "")
            summarized_text = results.get('final_plan', "")
            pdf_filename = create_pdf(english_text, arabic_text, summarized_text)
            with open(pdf_filename, "rb") as file:
                st.download_button("Download Text Summary", file, file_name=pdf_filename, mime="application/octet-stream")
    else:
        st.warning("Please enter text or upload a file to proceed.")

if __name__ == "__main__":
    main()
