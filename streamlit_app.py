import streamlit as st
import requests

# Define the base URL for the Flask API
BASE_URL = "http://localhost:5000"

# Function to upload file and process it
def upload_file(file):
    url = f"{BASE_URL}/upload"
    files = {'file': file}
    response = requests.post(url, files=files)
    return response.json()

# Function to display plots
def display_plot(plot_name):
    url = f"{BASE_URL}/plot/{plot_name}"
    response = requests.get(url)
    return response.content

# Main function for the Streamlit app
def main():
    st.title("Machine Learning Model Deployment")
    st.markdown("This web app allows you to upload a CSV file, train a machine learning model, and view evaluation plots.")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "About", "Contact"))

    if page == "Home":
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            st.write("File Uploaded Successfully!")
            if st.button("Process File"):
                response = upload_file(uploaded_file)
                st.json(response)

    elif page == "About":
        st.subheader("About Us")
        st.write("This is a web application developed for training and deploying machine learning models.")

    elif page == "Contact":
        st.subheader("Contact Us")
        st.write("For any inquiries, please email us at example@example.com")

    if st.sidebar.checkbox("Show Evaluation Plots"):
        st.subheader("Evaluation Plots")
        plot_names = ["confusion_matrix.png", "classification_report.png"]
        for plot_name in plot_names:
            plot_content = display_plot(plot_name)
            st.image(plot_content, use_column_width=True)

if __name__ == "__main__":
    main()
