import streamlit as st
import datetime
from data_acess import provider
from data_acess import math_adapter as ma
from method import algorithm
import plotly.express as px
import pandas as pd
from data_acess.provider import parse_uploaded_csv
from data_acess.db_utils import fetch_all_dataset_names, fetch_companies_by_dataset

st.set_page_config(page_title="Predictor Engine", layout="centered")

if "data_source" not in st.session_state:
    st.session_state.data_source = "Live Market Data (yfinance)"

if "_data_source_widget" not in st.session_state:
    st.session_state._data_source_widget = st.session_state.data_source

def save_data_source():
    st.session_state.data_source = st.session_state._data_source_widget

if "step" not in st.session_state:
    st.session_state.step = 1


def next_step():
    st.session_state.step += 1


def prev_step():
    st.session_state.step -= 1


def reset_wizard():
    st.session_state.step = 1

st.progress(st.session_state.step / 3.0, text=f"Step {st.session_state.step} of 3")
st.markdown("---")

# ==========================================
# STEP 1: DATASET & ASSET SELECTION
# ==========================================
if st.session_state.step == 1:
    st.header("1. Dataset Selection")
    st.write("Choose where the data is coming from and which assets to analyze.")

    st.radio(
        "Data Source",
        ["Live Market Data (yfinance)", "Upload Custom CSV"],
        key="_data_source_widget",
        on_change=save_data_source,
    )

    if st.session_state.data_source == "Live Market Data (yfinance)":
        available_datasets = fetch_all_dataset_names()
        selected_dataset = st.selectbox("Select Market Portfolio", available_datasets)

        if selected_dataset:
            companies_df = fetch_companies_by_dataset(selected_dataset)
            total_companies = len(companies_df)

            st.session_state.all_dataset_tickers = companies_df['ticker'].tolist()

            st.markdown("### Select Market Cap Slice")
            rank_start, rank_end = st.slider(
                "Filter by Company Rank (1 = Largest)",
                min_value=1,
                max_value=total_companies,
                value=(1, min(50, total_companies)),
                help="Select a range to analyze (e.g., Top 50, or Mid-Cap 100-200)."
            )

            sliced_df = companies_df.iloc[rank_start - 1: rank_end]

            ui_labels = [f"{row['ticker']} - {row['company_name']}" for _, row in sliced_df.iterrows()]

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



    else:

        st.info(
            "**Format requirement:** The first column must contain dates. "
            "All other columns must contain positive numeric asset prices, "
            "with ticker symbols as headers. Dates must be unique and "
            "ordered from oldest to newest. Missing prices are not allowed."
        )

        uploaded_file = st.file_uploader(
            "Upload your portfolio dataset",
            type=["csv"],
        )

        if uploaded_file is not None:
            try:
                custom_df = parse_uploaded_csv(uploaded_file)
                st.session_state.custom_df = custom_df

                st.session_state.tickers = list(custom_df.columns)

                st.success(
                    f"Successfully loaded {len(custom_df.columns)} assets "
                    f"and {len(custom_df)} trading periods."
                )

                with st.expander("Preview Uploaded Data"):
                    st.dataframe(custom_df.head())

            except ValueError as exc:
                st.session_state.pop("custom_df", None)
                st.session_state.pop("tickers", None)
                st.error(f"Invalid CSV: {exc}")

            except Exception:
                st.session_state.pop("custom_df", None)
                st.session_state.pop("tickers", None)
                st.error("Could not process the uploaded file. Check its format.")


    st.markdown("<br>", unsafe_allow_html=True)

    block_next_step = False
    if st.session_state.data_source == "Upload Custom CSV" and "custom_df" not in st.session_state:
        block_next_step = True
        st.warning("Please upload a CSV file to continue.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Next Step", on_click=next_step, type="primary", disabled=block_next_step)

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
            options=["1d", "1h", "1wk"],
            format_func=lambda x: {"1d": "Daily", "1h": "Hourly", "1wk": "Weekly"}[x],
            key="frequency"
        )

    st.markdown("### Window Calibration")
    st.number_input("Prediction Horizon (Steps)", min_value=1, value=30, key="future_k")

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

    if num_companies >= 3:
        # Reset a stale selection if the portfolio has changed.
        previous_k = st.session_state.get("k_target")
        if previous_k is not None and not (3 <= previous_k <= num_companies):
            st.session_state.pop("k_target", None)

        default_k = min(max(3, num_companies // 2), num_companies)

    else:
        # No valid threshold exists for a portfolio with fewer than 3 assets.
        st.session_state.pop("k_target", None)
        st.warning(
            "Select at least 3 companies before running the physics engine."
        )

    st.slider(
        f"How many companies out of {num_companies} must default to trigger a 'Systemic Crash' (k)?",
        min_value=3,
        max_value=max(3, num_companies),
        value=max(3, num_companies // 2),
        key="k_target",
        help="Defines how many simultaneous failures constitute a 'market crash'."
    )

    st.info(
        f"**What this means:** You are asking the engine to calculate the probability that at least {st.session_state.k_target} out of the {num_companies} selected companies will crash at the exact same time."
    )

    total_needed = st.session_state.K_prior + st.session_state.K + 1
    st.caption(
        f"**Data Requirement:** The pipeline will pull {total_needed} '{st.session_state.frequency}' intervals ending on {st.session_state.target_date}.")

    nav1, nav2 = st.columns([1, 4])

    with nav1:
        st.button("Back", on_click=prev_step)

    with nav2:
        st.button(
            "Run Physics Engine",
            on_click=next_step,
            type="primary",
            disabled=num_companies < 3,
        )

# ==========================================
# STEP 3: LIVE EXECUTION & RESULTS
# ==========================================
elif st.session_state.step == 3:
    st.header("3. Analysis Results")

    simulation_success = False

    with st.status("Executing Stochastic Physics Engine...", expanded=True) as status:
        try:
            if st.session_state.data_source == "Live Market Data (yfinance)":
                st.write("Fetching historical market data...")

                raw_df = provider.get_sliced_portfolio(
                    all_tickers=st.session_state.all_dataset_tickers,
                    selected_tickers=st.session_state.tickers,
                    end_date=st.session_state.target_date,
                    interval=st.session_state.frequency
                )

            else:
                st.write("Processing custom uploaded portfolio data...")
                raw_df = st.session_state.custom_df.copy()

            st.write(
                f"Structuring Matrix: {raw_df.shape[0]} periods × {raw_df.shape[1]} assets..."
            )

            if raw_df.shape[1] < 3:
                raise ValueError("The physics engine requires at least 3 assets.")

            K = st.session_state.K
            K_prior = st.session_state.K_prior
            k_target = st.session_state.get("k_target")

            if K < 1 or K_prior < 1:
                raise ValueError("Both time windows must be at least 1 step.")

            if k_target is None or not (3 <= k_target <= raw_df.shape[1]):
                raise ValueError(
                    "The crash threshold must be between 3 and the number of selected assets."
                )

            P_prior, P, N = ma.df_to_math_inputs(raw_df, K_prior, K)

            if N != raw_df.shape[1]:
                raise ValueError("The asset count does not match the input data.")

            st.write("Calibrating Prior Drift and running Monte Carlo simulations...")
            results = algorithm.run_algorithm(P_prior, P, N, st.session_state.K_prior, st.session_state.K, k_target)

            status.update(label="Engine Execution Complete!", state="complete", expanded=False)
            simulation_success = True


        except Exception as e:
            st.error(f"Error running pipeline: {e}")

            if isinstance(e, ValueError):
                st.info(
                    "Check the input data, selected assets, parameters, "
                    "and available historical data."
                )

            status.update(
                label="Execution Failed",
                state="error",
                expanded=False,
            )

    # ==========================================
    # OUTPUT LAYOUT
    # ==========================================
    if simulation_success:
        st.markdown("---")
        st.header("Simulation Results")

        # --- SECTION 1: DATA INPUT SUMMARY ---
        st.subheader("1. Configuration Summary")
        st.caption("The parameters used for this mathematical simulation.")

        sum_col1, sum_col2, sum_col3 = st.columns(3)
        sum_col1.metric("Anchor Date", str(st.session_state.target_date))
        sum_col2.metric("Portfolio Size (N)", f"{N} Assets")
        sum_col3.metric("Crash Threshold (k)", f"{k_target} Assets")

        sum_col4, sum_col5, sum_col6 = st.columns(3)
        sum_col4.metric("Data Frequency", st.session_state.frequency)
        sum_col5.metric("Prior Drift Window", f"{st.session_state.K_prior} steps")
        sum_col6.metric("Active Sim Window", f"{st.session_state.K} steps")

        st.markdown("---")

        # --- SECTION 2: MATH OUTPUT ---
        st.subheader("2. Macro State Variables")
        st.caption("Internal metrics calculated by the stochastic physics engine.")

        math_col1, math_col2, math_col3 = st.columns(3)
        math_col1.metric("Target Fraction (q)", f"{results['q']:.4f}", help="k / N")
        math_col2.metric("Equilibrium Default (p*)", f"{results['p_star']:.4f}",
                         help="Single-stock default probability")
        math_col3.metric("Rate Function (I*)", f"{results['I_q']:.4f}", help="Large deviation rate")

        st.markdown("---")

        # --- SECTION 3: THE BOTTOM LINE ---
        st.subheader("3. Systemic Crash Risk")

        # Convert probability to a percentage for readability
        crash_prob_pct = results['crash_probability'] * 100


        st.info(
            f"### Probability of $\ge {k_target}$ simultaneous crashes: \n"
            f"# **{crash_prob_pct:.4f} %**"
        )

        st.markdown("---")
        st.subheader("Trajectory Visualization")

        total_days = st.session_state.K_prior + st.session_state.K + 1
        plot_df = raw_df.iloc[-total_days:]
        plot_df_normalized = plot_df.div(plot_df.iloc[0], axis=1)

        fig = px.line(
            plot_df_normalized,
            labels={"value": "Normalized Price (1.0 = Start)", "index": "Date", "variable": "Ticker"},
            title="Asset Trajectories (Prior Drift vs. Active Window)"
        )
        fig.update_traces(line=dict(color="rgba(150, 150, 150, 0.15)", width=1))

        portfolio_avg = plot_df_normalized.mean(axis=1)
        fig.add_scatter(
            x=portfolio_avg.index,
            y=portfolio_avg,
            mode='lines',
            name='Portfolio Average',
            line=dict(color='#00d4ff', width=4)
        )

        split_date = plot_df_normalized.index[st.session_state.K_prior]
        fig.add_vline(
            x=split_date,
            line_dash="dash",
            line_color="red"
        )
        fig.add_annotation(
            x=split_date,
            y=1.05,
            yref="paper",
            text="Active Window Start ➔",
            showarrow=False,
            font=dict(color="red")
        )
        fig.update_layout(showlegend=False, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Run New Simulation", on_click=reset_wizard)