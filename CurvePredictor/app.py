import streamlit as st

st.set_page_config(page_title="Curve Prediction", layout="wide")

st.title("Curve Prediction Dashboard")
st.sidebar.header("Configuration")

dataset = st.sidebar.selectbox("Choose Data", ["tech_stocks_01", "Upload CSV..."])

if dataset == "Upload CSV...":
    uploaded_file = st.file_uploader("Upload your CSV file")
else:
    st.write(f"Querying SQLite for: **{dataset}**")

if st.button("Run Prediction"):
    st.success("Layered Architecture is ready to be built!")