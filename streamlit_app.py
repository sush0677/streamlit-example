import streamlit as st
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

template2 = "Translate the following text from English to Arabic text  :\n{english_text}"
prompt2 = ChatPromptTemplate.from_template(template2)
chain_2 = LLMChain(llm=model, prompt=prompt2, output_key="Arabic_text")

template3 = "Summarize the following text in Arabic :\n{Arabic_text}"
prompt3 = ChatPromptTemplate.from_template(template3)
chain_3 = LLMChain(llm=model, prompt=prompt3, output_key="final_plan")

seq_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3],
    input_variables=['review'],
    output_variables=['english_text', 'Arabic_text', 'final_plan'],
    verbose=True
)

# Streamlit User Interface for uploading and processing text
st.title("English to Arabic Translation")
uploaded_file = st.file_uploader("Upload your file", type=['txt'])

if uploaded_file is not None:
    raw_text = str(uploaded_file.read(), 'utf-8')  # Convert to string
else:
    raw_text = st.text_area("Or Enter the text", height=150)

if raw_text:
    results = seq_chain({"review": raw_text})  # Assuming seq_chain is defined elsewhere and works correctly
    english_text = results["english_text"]
    arabic_text = results["Arabic_text"]
    summary_text = results["final_plan"]

    if st.button("Show Original Text"):
        st.write("Original Text:")
        st.write(raw_text)

    if st.button("Translate to Arabic"):
        st.write("Translated Arabic Text:")
        st.write(arabic_text)

    if st.button("Summarization in Arabic"):
        st.write("Summary in Arabic:")
        st.write(summary_text)

