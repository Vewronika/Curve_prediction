import streamlit as st

st.set_page_config(
    page_title="CurvePredictor",
    page_icon="📈",
    layout="wide",
)

st.title("CurvePredictor")
st.markdown("### Explore systemic risk through a stochastic market model")
st.caption("A diploma research prototype for studying how portfolio price dynamics relate to systemic-event probabilities.")

st.markdown("")

st.write(
    "Choose a market universe or upload your own price history, set the observation windows, "
    "and run one analysis. The output is a model-based scenario estimate, not a forecast or investment recommendation."
)

st.markdown("## How it works")
overview_cols = st.columns(3)

with overview_cols[0].container(border=True):
    st.markdown("#### 01 · Choose data")
    st.write("Use the current constituents stored in the local database, or upload a CSV price matrix.")

with overview_cols[1].container(border=True):
    st.markdown("#### 02 · Set the windows")
    st.write("Choose the sampling interval, anchor date, and prior and active observation windows.")

with overview_cols[2].container(border=True):
    st.markdown("#### 03 · Review the result")
    st.write("Inspect the model output alongside the portfolio and data-quality summary.")

st.info("Open **Predictor Engine** in the sidebar to begin.", icon="ℹ️")

