import streamlit as st
import datetime
from data_acess import provider
import plotly.express as px
import pandas as pd
from data_acess.provider import parse_uploaded_csv
from data_acess.db_utils import fetch_all_dataset_names, fetch_companies_by_dataset
from application.pipeline import InsufficientObservationsError, run_prediction

st.set_page_config(page_title="Predictor Engine", page_icon="📊", layout="wide")

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

step_labels = ["Choose data", "Configure model", "Review results"]
current_step = st.session_state.step
st.title("Predictor engine")
st.caption("Run a single systemic-risk analysis using historical asset prices.")

with st.container(border=True):
    step_cols = st.columns(3)
    for index, label in enumerate(step_labels, start=1):
        marker = "✓" if index < current_step else f"{index:02d}"
        label_text = f"**{marker} · {label}**" if index == current_step else f"{marker} · {label}"
        step_cols[index - 1].markdown(label_text)
    st.progress(current_step / 3.0, text=f"Step {current_step} of 3 · {step_labels[current_step - 1]}")

# ==========================================
# STEP 1: DATASET & ASSET SELECTION
# ==========================================
if st.session_state.step == 1:
    st.header("Choose your data")
    st.write("Select a live market universe or bring your own price history.")

    st.radio(
        "Data source",
        ["Live Market Data (yfinance)", "Upload Custom CSV"],
        key="_data_source_widget",
        on_change=sync_data_source,
        horizontal=True,
    )

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

                with st.popover("Choose companies", use_container_width=True):
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
            "**Prepare your price history**\n\n"
            "CSV format: first column contains dates; each other column is one asset. "
            "Dates must be unique and ordered oldest to newest. Prices must be positive "
            "numbers with no missing values."
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

                with st.expander("Preview uploaded data", expanded=True):
                    st.dataframe(custom_df.head(), use_container_width=True)

            except ValueError as exc:
                st.session_state.pop("custom_df", None)
                st.session_state.pop("tickers", None)
                st.error(f"Invalid CSV: {exc}")

            except Exception:
                st.session_state.pop("custom_df", None)
                st.session_state.pop("tickers", None)
                st.error("Could not process the uploaded file. Check its format.")


    block_next_step = False
    if st.session_state.data_source == "Upload Custom CSV" and "custom_df" not in st.session_state:
        block_next_step = True
        st.warning("Please upload a CSV file to continue.")

    st.button("Continue to model settings →", on_click=next_step, type="primary", disabled=block_next_step, use_container_width=True)

# ==========================================
# STEP 2: TIMEFRAME & MATH CONFIGURATION
# ==========================================
elif st.session_state.step == 2:
    st.header("Configure the analysis")
    st.write("Choose the target period and observation windows used by the model.")

    with st.container(border=True):
        st.subheader("Market data")
        col1, col2 = st.columns(2)
        with col1:
            st.date_input(
                "Target date",
                value=datetime.date.today(),
                key="target_date",
                help="The engine counts backward from this date to fetch historical data."
            )
        with col2:
            st.selectbox(
                "Sampling interval",
                options=["1d", "1h", "1wk"],
                format_func=lambda x: {"1d": "Daily", "1h": "Hourly", "1wk": "Weekly"}[x],
                key="frequency"
            )

    with st.container(border=True):
        st.subheader("Observation windows")
        st.number_input("Prediction horizon (steps)", min_value=1, value=30, key="future_k")

        advanced_mode = st.toggle("Set prior and active windows separately")

        if not advanced_mode:
            st.caption("Prior and active windows both use the prediction horizon.")
            st.session_state.K = st.session_state.future_k
            st.session_state.K_prior = st.session_state.future_k
        else:
            st.warning("For a balanced setup, use similar lengths for the prior and active windows.")
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Active window (K)", min_value=1, value=st.session_state.future_k, key="K")
            with c2:
                st.number_input("Prior window (K_prior)", min_value=1, value=st.session_state.future_k, key="K_prior")

    with st.container(border=True):
        st.subheader("Systemic event threshold")

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
            st.caption(
                f"**Model definition:** The threshold is k={st.session_state.k_target} out of the currently selected N={num_companies} assets. This is the model's systemic-event definition, not evidence that these companies will literally default at the same time."
            )

    total_needed = st.session_state.K_prior + st.session_state.K + 1
    st.info(f"**Data requirement:** This run needs **{total_needed:,} complete observations** at the selected interval.")

    nav1, nav2 = st.columns([1, 3], vertical_alignment="bottom")

    with nav1:
        st.button("Back", on_click=prev_step)

    with nav2:
        st.button(
            "Run analysis →",
            on_click=next_step,
            type="primary",
            disabled=num_companies < 3,
            use_container_width=True,
        )

# ==========================================
# STEP 3: LIVE EXECUTION & RESULTS
# ==========================================
elif st.session_state.step == 3:
    st.header("Analysis results")
    st.write("Review the model output and the data used for this run.")

    simulation_success = False

    with st.status("Preparing data and running the model...", expanded=True) as status:
        try:
            is_market_data = st.session_state.data_source == "Live Market Data (yfinance)"
            if is_market_data:
                st.write("Fetching historical market data...")
            else:
                st.write("Processing custom uploaded portfolio data...")
            st.write("Validating inputs and preparing the model run...")
            run = run_prediction(
                data_mode="market" if is_market_data else "csv",
                k_prior=st.session_state.K_prior,
                k=st.session_state.K,
                k_target=st.session_state.get("k_target"),
                custom_prices=st.session_state.get("custom_df"),
                all_tickers=st.session_state.get("all_dataset_tickers"),
                selected_tickers=st.session_state.get("tickers"),
                target_date=st.session_state.get("target_date"),
                interval=st.session_state.get("frequency"),
                replace_incomplete=st.session_state.get("replace_incomplete_assets", True),
            )
            raw_df = run.prices
            data_report = run.data_report
            results = run.model_output
            N = run.asset_count
            K = run.k
            K_prior = run.k_prior
            k_target = run.k_target
            st.session_state.data_report = data_report

            st.write(f"Validated matrix: {raw_df.shape[0]} periods × {raw_df.shape[1]} assets.")

            status.update(label="Engine Execution Complete!", state="complete", expanded=False)
            simulation_success = True

        except (provider.InsufficientHistoricalDataError, InsufficientObservationsError) as e:
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


        with st.container(border=True):
            st.metric(
                f"Model probability of at least {k_target} simultaneous crashes",
                f"{crash_prob_pct:.4f}%",
            )
            st.caption("This is a model-based estimate for the selected inputs, not a real-world guarantee.")

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
        fig.update_layout(
            template="plotly_white",
            height=540,
            margin=dict(l=24, r=24, t=64, b=28),
            showlegend=False,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.button("Start a new analysis", on_click=reset_wizard, use_container_width=True)
