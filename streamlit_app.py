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

def main():
    st.title("Text Processing App")
    user_input = get_input_text()

    if user_input:
        if st.button("Translate to Arabic"):
            try:
                # Run only the first chain and check the output
                english_result = chain_1.run(review=user_input)
                if isinstance(english_result, dict) and 'english_text' in english_result:
                    english_text = english_result['english_text']
                    st.text_area("Translated Text:", english_text, height=150)

                    # Now run the second chain using the output of the first and check the output
                    arabic_result = chain_2.run(english_text=english_text)
                    if isinstance(arabic_result, dict) and 'Arabic_text' in arabic_result:
                        arabic_text = arabic_result['Arabic_text']
                        st.session_state['arabic_text'] = arabic_text  # Save Arabic text for other operations
                    else:
                        st.error("Failed to get Arabic translation. Check model output.")
                else:
                    st.error("Failed to translate text to English properly. Check model output.")

            except Exception as e:
                st.error(f"Translation failed: {str(e)}")

        if st.button("Summarize in Arabic"):
            try:
                # Check if Arabic text is already processed or needs to be processed
                if 'arabic_text' in st.session_state:
                    arabic_text = st.session_state['arabic_text']
                else:
                    # If not processed, run the first and second chains again
                    english_result = chain_1.run(review=user_input)
                    english_text = english_result['english_text']
                    arabic_result = chain_2.run(english_text=english_text)
                    arabic_text = arabic_result['Arabic_text']

                # Run the summarization chain and check the output
                summarized_result = chain_3.run(Arabic_text=arabic_text)
                if isinstance(summarized_result, dict) and 'final_plan' in summarized_result:
                    summarized_text = summarized_result['final_plan']
                    st.text_area("Arabic Summary:", summarized_text, height=150)
                else:
                    st.error("Failed to summarize the text. Check model output.")

            except Exception as e:
                st.error(f"Summarization failed: {str(e)}")

        if st.button("Download PDF"):
            try:
                # Ensure all chains have been processed
                english_result = chain_1.run(review=user_input)
                english_text = english_result['english_text']
                arabic_result = chain_2.run(english_text=english_text)
                arabic_text = arabic_result['Arabic_text']
                summarized_result = chain_3.run(Arabic_text=arabic_text)
                summarized_text = summarized_result['final_plan']

                # Create and download PDF
                pdf_filename = create_pdf(english_text, arabic_text, summarized_text)
                with open(pdf_filename, "rb") as file:
                    st.download_button("Download Text Summary", file, file_name=pdf_filename, mime="application/octet-stream")

            except Exception as e:
                st.error(f"PDF creation failed: {str(e)}")
    else:
        st.warning("Please enter text or upload a file to proceed.")

if __name__ == "__main__":
    main()
