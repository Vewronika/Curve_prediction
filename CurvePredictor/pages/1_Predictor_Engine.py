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

if "step" not in st.session_state:
    st.session_state.step = 1
if "data_source" not in st.session_state:
    st.session_state.data_source = "Live Market Data (yfinance)"


def sync_data_source():
    st.session_state.data_source = st.session_state._data_source_widget


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

    st.radio("Data Source", ["Live Market Data (yfinance)", "Upload Custom CSV"], key="_data_source_widget", on_change=sync_data_source)

    if st.session_state.data_source == "Live Market Data (yfinance)":
        available_datasets = fetch_all_dataset_names()
        if not available_datasets:
            st.error("No company universes are available in the local database. Run `python setup_db.py` from the project directory to initialize/refresh them.")
        else:
            default_dataset = "S&P 500" if "S&P 500" in available_datasets else available_datasets[0]
            selected_dataset = st.selectbox("Select Market Universe", available_datasets,
                                             index=available_datasets.index(default_dataset))

            companies_df = fetch_companies_by_dataset(selected_dataset)
            total_companies = len(companies_df)
            st.session_state.all_dataset_tickers = companies_df["ticker"].tolist()

            st.caption(
                "These are current index constituents from the last successful database refresh. "
                "Historical membership is not reconstructed yet, so backtests can have survivorship bias."
            )
            if "market_cap_usd" in companies_df and companies_df["market_cap_usd"].notna().any():
                st.info("This universe has dated market-cap metadata; companies are ordered by available market cap.")
            else:
                st.warning("The current rank is source-list order, **not market-cap rank**. Use explicit company selection; similar company sizes are not guaranteed.")

            if total_companies:
                rank_start, rank_end = st.slider(
                    "Choose a range from the universe list",
                    min_value=1,
                    max_value=total_companies,
                    value=(1, min(50, total_companies)),
                    key="company_range",
                    help="This range follows the displayed ordering. It is not a market-cap slice unless dated market-cap data is available."
                )
                sliced_df = companies_df.iloc[rank_start - 1: rank_end]
                ui_labels = [f"{row['ticker']} - {row['company_name']}" for _, row in sliced_df.iterrows()]

                # The range slider changes the available multiselect options.
                # Reset selection when either the universe or range changes so
                # Streamlit cannot retain values that are no longer valid options.
                selection_context = (selected_dataset, rank_start, rank_end)
                if st.session_state.get("_company_selection_context") != selection_context:
                    st.session_state["ui_company_selection"] = ui_labels.copy()
                    st.session_state["_company_selection_context"] = selection_context
                elif "ui_company_selection" not in st.session_state:
                    st.session_state["ui_company_selection"] = ui_labels.copy()

                with st.popover("See and Customize Companies", use_container_width=True):
                    st.write("Select the companies to include. Changing the range above resets the selection to all companies in that range. Incomplete assets may be excluded and replaced from this same universe.")
                    selected_labels = st.multiselect(
                        "Review and Filter Companies",
                        options=ui_labels,
                        key="ui_company_selection",
                        label_visibility="collapsed"
                    )
                st.session_state.tickers = [label.split(" - ")[0].strip() for label in selected_labels]
                st.success(f"{len(st.session_state.tickers)} companies selected for analysis.")
                st.checkbox("Replace assets with incomplete/unavailable prices using other companies from this universe",
                            value=True, key="replace_incomplete_assets",
                            help="No missing prices are filled. Replacements and exclusions will be listed in the result summary.")
            else:
                st.session_state.tickers = []
                st.warning("The selected universe has no companies. Refresh the database.")


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

    if num_companies < 3:
        # Do not construct a slider with an invalid range for N < 3.
        st.session_state.pop("k_target", None)
        st.warning(
            f"Currently selected: {num_companies} companies. Return to Step 1 and select at least 3 companies before running the physics engine."
        )
    elif num_companies == 3:
        # Streamlit sliders require min_value < max_value.
        st.session_state["k_target"] = 3
        st.info("With N = 3 selected companies, the threshold is fixed at k = 3.")
    else:
        previous_k = st.session_state.get("k_target")
        if previous_k is None or not (3 <= previous_k <= num_companies):
            st.session_state["k_target"] = min(max(3, num_companies // 2), num_companies)

        st.slider(
            f"How many companies out of {num_companies} must default to trigger a 'Systemic Crash' (k)?",
            min_value=3,
            max_value=num_companies,
            key="k_target",
            help="Defines the threshold k used by the mathematical model; this is a modeling choice, not a validated real-world crash label."
        )

    if num_companies >= 3:
        st.info(
            f"**Model definition:** The threshold is k={st.session_state.k_target} out of the currently selected N={num_companies} assets. This is the model's systemic-event definition, not evidence that these companies will literally default at the same time."
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

                required_rows = int(st.session_state.K_prior) + int(st.session_state.K) + 1
                raw_df, data_report = provider.get_sliced_portfolio(
                    all_tickers=st.session_state.all_dataset_tickers,
                    selected_tickers=st.session_state.tickers,
                    end_date=st.session_state.target_date,
                    interval=st.session_state.frequency,
                    required_rows=required_rows,
                    replace_incomplete=st.session_state.get("replace_incomplete_assets", True),
                )
                st.session_state.data_report = data_report

            else:
                st.write("Processing custom uploaded portfolio data...")
                raw_df = st.session_state.custom_df.copy()
                data_report = {
                    "requested_count": raw_df.shape[1], "final_count": raw_df.shape[1],
                    "requested_tickers": list(raw_df.columns), "final_tickers": list(raw_df.columns),
                    "removed": [], "replacements": [], "shortfall": 0,
                    "observations": len(raw_df), "provider": "User-uploaded CSV",
                    "note": "CSV data supplied by the user; no market-universe membership metadata."
                }
                st.session_state.data_report = data_report

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

            if raw_df.shape[1] < 3:
                raise ValueError(
                    f"Only {raw_df.shape[1]} usable assets remain after data validation/replacement. At least 3 are required."
                )
            if k_target is None or not (3 <= k_target <= raw_df.shape[1]):
                raise ValueError(
                    f"Crash threshold k={k_target} is invalid for the final portfolio size N={raw_df.shape[1]}. "
                    "Go back and select a valid threshold for the usable portfolio."
                )

            P_prior, P, N = ma.df_to_math_inputs(raw_df, K_prior, K)

            if N != raw_df.shape[1]:
                raise ValueError("The asset count does not match the input data.")

            st.write("Calibrating Prior Drift and running Monte Carlo simulations...")
            results = algorithm.run_algorithm(P_prior, P, N, st.session_state.K_prior, st.session_state.K, k_target)

            status.update(label="Engine Execution Complete!", state="complete", expanded=False)
            simulation_success = True

        except provider.InsufficientHistoricalDataError as e:
            st.warning(str(e))
            status.update(label="Insufficient historical data", state="error", expanded=False)
        except Exception as e:
            st.error(f"Error running pipeline: {e}")
            if isinstance(e, provider.NoMarketDataError):
                st.info("No rows were returned. Check the selected date range and whether the provider supports this interval for that period.")
            status.update(label="Execution Failed", state="error", expanded=False)

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

        report = st.session_state.get("data_report", {})
        with st.expander("Data quality and portfolio changes", expanded=bool(report.get("removed") or report.get("replacements"))):
            st.write(f"Requested assets: **{report.get('requested_count', N)}** · Final usable assets: **{report.get('final_count', N)}** · Complete observations: **{report.get('observations', raw_df.shape[0])}**")
            st.caption(report.get("note", ""))
            if report.get("replacements"):
                st.write("**Replacement assets added:** " + ", ".join(report["replacements"]))
            if report.get("removed"):
                st.write("**Assets/candidates excluded:**")
                st.dataframe(pd.DataFrame(report["removed"]), use_container_width=True, hide_index=True)
            if report.get("shortfall", 0):
                st.warning(f"Could not fill the requested portfolio: shortfall of {report['shortfall']} asset(s). The displayed N is the actual usable count.")

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
            f"### Probability of $\\ge {k_target}$ simultaneous crashes: \n"
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
