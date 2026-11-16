import streamlit as st
import datetime
from data_acess import provider
from data_acess import math_adapter as ma
from method import algorithm
from data_acess.db_utils import fetch_all_dataset_names, fetch_companies_by_dataset


st.set_page_config(page_title="Predictor Engine", layout="centered")

if "step" not in st.session_state:
    st.session_state.step = 1


def next_step():
    st.session_state.step += 1


def prev_step():
    st.session_state.step -= 1


def reset_wizard():
    st.session_state.step = 1


#track progress
st.progress(st.session_state.step / 3.0, text=f"Step {st.session_state.step} of 3")
st.markdown("---")

# ==========================================
# STEP 1: DATASET & ASSET SELECTION
# ==========================================
if st.session_state.step == 1:
    st.header("1. Dataset Selection")
    st.write("First, choose where your data is coming from and which assets to analyze.")

    st.radio("Data Source", ["Live Market Data (yfinance)", "Upload Custom CSV"], key="data_source")

    if st.session_state.data_source == "Live Market Data (yfinance)":

        available_datasets = fetch_all_dataset_names()
        selected_dataset = st.selectbox("Select Market Portfolio", available_datasets)

        if selected_dataset:
            companies_df = fetch_companies_by_dataset(selected_dataset)
            ui_labels = [f"{row['ticker']} - {row['company_name']}" for _, row in companies_df.iterrows()]

            if selected_dataset:
                companies_df = fetch_companies_by_dataset(selected_dataset)
                ui_labels = [f"{row['ticker']} - {row['company_name']}" for _, row in companies_df.iterrows()]


                with st.popover("See and Customize Companies", use_container_width=True):
                    st.write("Tick or untick companies to include in the simulation:")

                    selected_labels = st.multiselect(
                        "Review and Filter Companies",
                        options=ui_labels,
                        default=ui_labels,
                        key="ui_company_selection",
                        label_visibility="collapsed"
                    )

                st.session_state.tickers = [label.split(" - ")[0].strip() for label in selected_labels]

                st.success(f"{len(st.session_state.tickers)} companies actively selected for analysis.")


            st.session_state.tickers = [label.split(" - ")[0].strip() for label in selected_labels]

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Next Step", on_click=next_step, type="primary")

# ==========================================
# STEP 2: TIMEFRAME & MATH CONFIGURATION
# ==========================================
elif st.session_state.step == 2:
    st.header("2. Timeframe & Model Parameters")

    col1, col2 = st.columns(2)
    with col1:
        st.date_input(
            "Target Date (Anchor)",
            value=datetime.date.today(),
            key="target_date",
            help="The engine counts backward from this date to fetch historical data."
        )
    with col2:
        st.selectbox(
            "Data Frequency",
            options=["1h", "1d", "1wk"],
            format_func=lambda x: {"1h": "Hourly", "1d": "Daily", "1wk": "Weekly"}[x],
            key="frequency"
        )

    st.markdown("### Window Calibration")
    st.number_input("Prediction Horizon (Steps)", min_value=1, value=21, key="future_k")

    advanced_mode = st.toggle("Advanced / Custom Mode")

    if not advanced_mode:
        st.success("**Automatic Mode:** Prior and Active windows matched to Prediction Horizon.")
        st.session_state.K = st.session_state.future_k
        st.session_state.K_prior = st.session_state.future_k
    else:
        st.warning("**Warning:** K and K_prior should mathematically align for optimal stochastic stability.")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Active Window (K)", min_value=1, value=st.session_state.future_k, key="K")
        with c2:
            st.number_input("Prior Window (K_prior)", min_value=1, value=st.session_state.future_k, key="K_prior")

    st.markdown("### Systemic Crash Definition")


    num_companies = len(st.session_state.get("tickers", []))

    st.slider(
        f"How many companies out of {num_companies} must default to trigger a 'Systemic Crash' (k)?",
        min_value=3,
        max_value=max(3, num_companies),
        value=max(3, num_companies // 2),
        key="k_target"
    )

    total_needed = st.session_state.K_prior + st.session_state.K + 1
    st.caption(
        f"**Data Requirement:** The pipeline will pull {total_needed} '{st.session_state.frequency}' intervals ending on {st.session_state.target_date}.")

    nav1, nav2 = st.columns([1, 4])
    with nav1:
        st.button("Back", on_click=prev_step)
    with nav2:
        st.button("Run Physics Engine", on_click=next_step, type="primary")


## ==========================================
# STEP 3: LIVE EXECUTION & RESULTS
# ==========================================
elif st.session_state.step == 3:
    st.header("3. Analysis Results")

    with st.spinner("Fetching live data and running statistical mechanics engine..."):
        try:
            # 1. Fetch data based on user choices
            raw_df = provider.fetch_stock_data(
                tickers=st.session_state.tickers,
                end_date=st.session_state.target_date,
                interval=st.session_state.frequency
            )



            # 2. Slice the DataFrame
            P_prior, P, N = ma.df_to_math_inputs(raw_df, st.session_state.K_prior, st.session_state.K)
            k_target = st.session_state.k_target

            # 3. Run the math
            results = algorithm.run_algorithm(P_prior, P, N, st.session_state.K_prior, st.session_state.K, k_target)

            st.success("Engine Execution Complete.")

            # --- ROW 1: Primary Metrics ---
            m1, m2, m3 = st.columns(3)
            m1.metric("Systemic Crash Probability", f"{results['crash_probability'] * 100:.4f}%")
            m2.metric("Equilibrium Default Prob (p*)", f"{results['p_star']:.4f}")
            m3.metric("Target Crash Fraction (q)", f"{results['q']:.2f}")

            # --- ROW 2: Advanced Physics Metrics ---
            st.markdown("<br>", unsafe_allow_html=True)
            m4, m5, m6 = st.columns(3)
            m4.metric("Large Deviation Exponent", f"{results['exponent']:.4f}")
            m5.metric("Rate Function (I_q)", f"{results['I_q']:.4f}")
            m6.metric("Total Assets (N) / Target (k)", f"{results['N']} / {results['k']}")

            # Warning if underflow occurred during math calculation
            if results['underflow_flag']:
                st.warning("Precision Warning: An underflow occurred in the probability calculation. Results may be extreme approximations.")

            st.markdown("---")
            st.subheader("Trajectory Visualization")
            st.info("(Next Step: Plotly charts here)")

        except Exception as e:
            st.error(f"Error running pipeline: {e}")
            st.info("Check if your target date has enough historical data for the selected interval.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Run New Simulation", on_click=reset_wizard)