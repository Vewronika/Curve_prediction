import streamlit as st

st.set_page_config(
    page_title="Curve predictor",
    layout="centered"
)

st.title("Systemic Risk & Market Combinatorics")
st.subheader("A Heterogeneous Ensemble Approach to Crash Prediction")

st.markdown("---")

st.markdown("""
### Project Overview
Traditional Machine Learning struggles to predict "Black Swan" market crashes because it relies on historical pattern repetition. Statistical Mechanics, however, treats the market as a physical system, measuring realized energy and pairwise interactions to calculate the probability of extreme deviations.

This diploma project fuses both disciplines:
1. **The Physics Engine:** Uses Large Deviations Theory to calculate the thermodynamic probability of a systemic crash.
2. **The Meta-Learner:** A Layer 2 AI that optimally weights the physics-based predictions against standard ML baselines.

### How to use this tool:
**Select a module from the sidebar to begin.**
* **Predictor Engine:** Run live Yahoo Finance data through the step-by-step stochastic physics model.
* **ML Meta-Learner:** Compare the physics model against the Machine Learning ensemble.
""")

