import datetime
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from application.experiment_runner import (
    ExperimentConfig,
    run_k_experiment,
    run_k_target_experiment,
    run_portfolio_size_experiment,
)
from data_acess.db_utils import fetch_all_dataset_names, fetch_companies_by_dataset
from data_acess.provider import parse_uploaded_csv


st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")

RESULT_STATE_KEY = "experiment_result"
DEFAULT_UNIVERSE_RANGE = 50


def _requested_size(config: ExperimentConfig) -> int:
    if config.data_mode == "csv":
        return config.custom_prices.shape[1]
    return len(config.selected_tickers)


def _range_values(label: str, minimum: int, maximum: int, key: str) -> list[int]:
    if minimum == maximum:
        st.info(f"Only {minimum} is available for {label}.")
        return [minimum]

    selected_range = st.slider(
        f"{label} range",
        min_value=minimum,
        max_value=maximum,
        value=(minimum, maximum),
        key=f"{key}_range_{minimum}_{maximum}",
    )
    start, end = selected_range
    step = st.number_input(
        f"Step between {label} values",
        min_value=1,
        max_value=max(1, maximum - minimum),
        value=1,
        step=1,
        key=f"{key}_step_{minimum}_{maximum}",
        help="Step 1 runs every value in the selected range. Larger steps skip values.",
    )
    values = list(range(start, end + 1, int(step)))
    if values[-1] != end:
        values.append(end)
    st.caption(f"Values to run: {', '.join(map(str, values))} · {len(values)} runs")
    return values


def _render_results(experiment):
    st.header("Experiment results")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Runs", experiment.total_runs)
    summary_col2.metric("Successful", experiment.successful_runs)
    summary_col3.metric("Failed", experiment.failed_runs)

    table_rows = []
    for run in experiment.runs:
        requested_n = _requested_size(run.config)
        if run.succeeded:
            actual_n = run.prediction.asset_count
            probability = run.prediction.model_output.get("crash_probability")
            probability_text = (
                f"{float(probability) * 100:.4f}%" if probability is not None else "Unavailable"
            )
            status = "Succeeded"
            error = ""
        else:
            actual_n = "—"
            probability_text = "—"
            status = "Failed"
            error = run.error_message or run.error_type or "Unknown error"
        table_rows.append(
            {
                "Requested N": requested_n,
                "Usable N": actual_n,
                "k_target": run.config.k_target,
                "K": run.config.k,
                "K_prior": run.config.k_prior,
                "Crash probability": probability_text,
                "Status": status,
                "Error": error,
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    experiment_type = st.session_state.get("experiment_result_type", "Vary portfolio size N")
    axis_config = {
        "Vary portfolio size N": ("Requested N", "Portfolio size N"),
        "Vary systemic threshold k_target": ("k_target", "Systemic threshold k_target"),
        "Vary active window K": ("K", "Active window K"),
    }
    x_column, x_label = axis_config.get(
        experiment_type, ("Requested N", "Portfolio size N")
    )
    plot_rows = []
    for run in experiment.runs:
        if not run.succeeded:
            continue
        probability = run.prediction.model_output.get("crash_probability")
        if probability is None:
            continue
        plot_rows.append(
            {
                x_column: getattr(run.config, x_column.lower())
                if x_column == "k_target"
                else (
                    run.config.k
                    if x_column == "K"
                    else _requested_size(run.config)
                ),
                "Crash probability (%)": float(probability) * 100,
                "Usable N": run.prediction.asset_count,
                "k_target": run.config.k_target,
                "K": run.config.k,
                "K_prior": run.config.k_prior,
            }
        )

    if plot_rows:
        plot_data = pd.DataFrame(plot_rows).sort_values(x_column)
        figure = px.line(
            plot_data,
            x=x_column,
            y="Crash probability (%)",
            markers=True,
            hover_data=["Usable N", "k_target", "K", "K_prior"],
            title=f"Model result by {x_label}",
        )
        figure.update_layout(
            template="plotly_white",
            height=430,
            xaxis_title=x_label,
            yaxis_title="Model crash probability (%)",
            margin=dict(l=24, r=24, t=60, b=32),
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("There are no successful runs with a probability result to plot.")

    for run in experiment.runs:
        requested_n = _requested_size(run.config)
        if experiment_type == "Vary systemic threshold k_target":
            title = f"Details for k_target = {run.config.k_target} (N = {requested_n})"
        elif experiment_type == "Vary active window K":
            title = f"Details for K = {run.config.k} (N = {requested_n})"
        else:
            title = f"Details for requested N = {requested_n}"
        with st.expander(title):
            if not run.succeeded:
                st.error(f"{run.error_type}: {run.error_message}")
                continue

            prediction = run.prediction
            probability = prediction.model_output.get("crash_probability")
            if probability is not None:
                st.metric(
                    f"Model probability of at least {prediction.k_target} simultaneous crashes",
                    f"{float(probability) * 100:.4f}%",
                )

            report = prediction.data_report
            st.write(
                f"Requested assets: **{report.get('requested_count', requested_n)}** · "
                f"Usable assets: **{prediction.asset_count}** · "
                f"Complete observations: **{report.get('observations', len(prediction.prices))}**"
            )
            if report.get("replacements"):
                st.write("**Replacement assets:** " + ", ".join(report["replacements"]))
            if report.get("removed"):
                st.write("**Excluded assets:**")
                st.dataframe(
                    pd.DataFrame(report["removed"]),
                    use_container_width=True,
                    hide_index=True,
                )
            if report.get("shortfall", 0):
                st.warning(f"The requested portfolio has a shortfall of {report['shortfall']} asset(s).")

            st.caption("Assets used in this run")
            st.write(", ".join(map(str, prediction.prices.columns)))


st.title("Experiments")
st.caption(
    "Change one model variable at a time and compare the results across configurations."
)

experiment_type = st.radio(
    "What do you want to vary?",
    [
        "Vary portfolio size N",
        "Vary systemic threshold k_target",
        "Vary active window K",
    ],
    horizontal=True,
    key="experiment_type_selector",
    captions=[
        "Hold k_target and the other settings fixed.",
        "Hold the portfolio size and other settings fixed.",
        "Hold K_prior, N, and k_target fixed.",
    ],
)

with st.container(border=True):
    st.subheader("1. Choose your data")
    data_source = st.radio(
        "Data source",
        ["Upload Custom CSV", "Live Market Data (yfinance)"],
        horizontal=True,
        key="n_experiment_data_source",
        index=1,
    )

    custom_prices = None
    all_tickers = None
    selected_tickers = None
    data_ready = False
    available_assets = 0
    data_mode = "csv" if data_source == "Upload Custom CSV" else "market"

    if data_mode == "csv":
        uploaded_file = st.file_uploader(
            "Upload a price-history CSV",
            type=["csv"],
            key="n_experiment_csv",
            help="Use a date column followed by one column per asset, as on the single-run page.",
        )
        if uploaded_file is not None:
            try:
                custom_prices = parse_uploaded_csv(BytesIO(uploaded_file.getvalue()))
                available_assets = custom_prices.shape[1]
                data_ready = True
                st.success(
                    f"Loaded {available_assets} assets and {len(custom_prices)} observations."
                )
                with st.expander("Preview uploaded prices"):
                    st.dataframe(custom_prices.head(), use_container_width=True)
            except ValueError as exc:
                st.error(f"Invalid CSV: {exc}")
    else:
        try:
            datasets = fetch_all_dataset_names()
        except Exception as exc:
            datasets = []
            st.error(f"Could not read the local market-universe database: {exc}")

        if datasets:
            default_dataset = "S&P 500" if "S&P 500" in datasets else datasets[0]
            dataset = st.selectbox(
                "Market universe",
                datasets,
                index=datasets.index(default_dataset),
                key="n_experiment_dataset",
            )
            try:
                companies = fetch_companies_by_dataset(dataset)
            except Exception as exc:
                companies = pd.DataFrame()
                st.error(f"Could not load companies for this universe: {exc}")

            if not companies.empty:
                all_tickers = companies["ticker"].astype(str).tolist()
                total_companies = len(companies)
                if total_companies == 1:
                    range_start = range_end = 1
                    st.info("This universe contains one company.")
                else:
                    range_start, range_end = st.slider(
                        "Choose a range from the market universe",
                        min_value=1,
                        max_value=total_companies,
                        value=(1, min(DEFAULT_UNIVERSE_RANGE, total_companies)),
                        key=f"n_experiment_universe_range_{dataset}_{total_companies}",
                        help="The selected range is ordered by the universe list. Every company in the range is included.",
                    )
                selected_companies = companies.iloc[range_start - 1 : range_end]
                selected_tickers = selected_companies["ticker"].astype(str).tolist()
                available_assets = len(selected_tickers)
                data_ready = available_assets > 0
                st.caption(
                    f"Selected {available_assets} companies, positions {range_start}–{range_end}. "
                    "An N-sized run uses the first N companies in this range."
                )
                if "market_cap_usd" in companies and companies["market_cap_usd"].notna().any():
                    st.info("This universe is ordered using available market-cap metadata.")
                else:
                    st.warning("This universe uses source-list order, not market-cap ranking.")
                st.checkbox(
                    "Replace assets with incomplete prices from this universe",
                    value=True,
                    key="n_experiment_replace_incomplete",
                    help="The provider reports replacements and exclusions for each portfolio size.",
                )
            else:
                st.warning("The selected universe has no companies. Refresh the database.")
        else:
            st.warning("No market universes are available. Upload a CSV or initialize the local database.")

with st.container(border=True):
    st.subheader("2. Set shared model parameters")

    if data_mode == "market":
        date_col, interval_col = st.columns(2)
        target_date = date_col.date_input(
            "Target date",
            value=datetime.date.today(),
            key="n_experiment_target_date",
            help="The provider loads historical prices ending at this date.",
        )
        interval = interval_col.selectbox(
            "Sampling interval",
            options=["1d", "1h", "1wk"],
            format_func=lambda value: {"1d": "Daily", "1h": "Hourly", "1wk": "Weekly"}[value],
            key="n_experiment_interval",
        )
    else:
        target_date = None
        interval = None

    if experiment_type == "Vary active window K":
        k_prior = st.number_input(
            "Fixed prior window K_prior (steps)",
            min_value=1,
            value=30,
            step=1,
            key="k_sweep_k_prior",
            help="K_prior stays the same while the active window K changes.",
        )
        k = 1
    else:
        separate_windows = st.toggle(
            "Set prior and active windows separately",
            key="n_experiment_separate_windows",
        )
        if separate_windows:
            window_col1, window_col2 = st.columns(2)
            k = window_col1.number_input(
                "Active window K (steps)", min_value=1, value=30, step=1, key="n_experiment_k"
            )
            k_prior = window_col2.number_input(
                "Prior window K_prior (steps)", min_value=1, value=30, step=1,
                key="n_experiment_k_prior",
            )
        else:
            horizon = st.number_input(
                "Window length for K and K_prior (steps)",
                min_value=1,
                value=30,
                step=1,
                key="n_experiment_window_length",
            )
            k = int(horizon)
            k_prior = int(horizon)

    fixed_n = None
    fixed_k_target = None
    experiment_values = []

    if available_assets < 3:
        st.warning("At least 3 assets are needed to set k_target and run an experiment.")
    elif experiment_type == "Vary portfolio size N":
        if available_assets == 3:
            fixed_k_target = 3
            st.info("With 3 available assets, k_target is fixed at 3.")
        else:
            fixed_k_target = st.slider(
                "Fixed systemic-event threshold k_target",
                min_value=3,
                max_value=available_assets,
                value=min(max(3, available_assets // 2), available_assets),
                key=f"n_experiment_k_target_{available_assets}",
                help="This threshold stays fixed while the portfolio size N changes.",
            )

        experiment_values = _range_values(
            "Portfolio size N",
            minimum=fixed_k_target,
            maximum=available_assets,
            key="n_experiment_sizes",
        )
    elif experiment_type == "Vary systemic threshold k_target":
        fixed_n = st.slider(
            "Fixed portfolio size N",
            min_value=3,
            max_value=available_assets,
            value=available_assets,
            key=f"k_experiment_portfolio_size_{available_assets}",
            help="This number of assets stays fixed while k_target changes.",
        )
        experiment_values = _range_values(
            "Systemic threshold k_target",
            minimum=3,
            maximum=fixed_n,
            key="k_experiment_thresholds",
        )
        fixed_k_target = experiment_values[0] if experiment_values else None
    elif experiment_type == "Vary active window K":
        fixed_n = st.slider(
            "Fixed portfolio size N",
            min_value=3,
            max_value=available_assets,
            value=available_assets,
            key=f"k_sweep_portfolio_size_{available_assets}",
            help="This number of assets stays fixed while K changes.",
        )
        if fixed_n == 3:
            fixed_k_target = 3
            st.info("With N = 3, k_target is fixed at 3.")
        else:
            fixed_k_target = st.slider(
                "Fixed systemic-event threshold k_target",
                min_value=3,
                max_value=fixed_n,
                value=min(max(3, fixed_n // 2), fixed_n),
                key=f"k_sweep_k_target_{fixed_n}",
                help="This threshold stays fixed while K changes.",
            )

        csv_max_k = (
            len(custom_prices) - int(k_prior) - 1
            if custom_prices is not None
            else None
        )
        if csv_max_k is not None and csv_max_k < 1:
            st.warning(
                f"This CSV supports no active window K with K_prior={k_prior}; upload more observations or reduce K_prior."
            )
            experiment_values = []
        elif csv_max_k == 1:
            st.info("Only K = 1 is possible with this CSV and K_prior.")
            experiment_values = [1]
        else:
            k_from_col, k_to_col, k_step_col = st.columns(3)
            k_max_value = csv_max_k if csv_max_k is not None else None
            default_k_end = csv_max_k if csv_max_k is not None else 30
            k_start = k_from_col.number_input(
                "Start K",
                min_value=1,
                max_value=k_max_value,
                value=1,
                step=1,
                key=f"k_sweep_start_{available_assets}_{k_prior}_{csv_max_k}",
            )
            k_end = k_to_col.number_input(
                "End K",
                min_value=1,
                max_value=k_max_value,
                value=default_k_end,
                step=1,
                key=f"k_sweep_end_{available_assets}_{k_prior}_{csv_max_k}",
            )
            k_step = k_step_col.number_input(
                "Step between K values",
                min_value=1,
                value=1,
                step=1,
                key=f"k_sweep_step_{available_assets}_{k_prior}",
                help="Step 1 tests each K in the selected range.",
            )
            if k_end < k_start:
                st.warning("End K must be greater than or equal to Start K.")
                experiment_values = []
            else:
                experiment_values = list(range(int(k_start), int(k_end) + 1, int(k_step)))
                if experiment_values[-1] != int(k_end):
                    experiment_values.append(int(k_end))
                st.caption(
                    f"K values to run: {', '.join(map(str, experiment_values))} · "
                    f"{len(experiment_values)} runs"
                )
                if len(experiment_values) > 100:
                    st.warning(
                        f"This will run {len(experiment_values)} predictions and may take a while. "
                        "Increase the step to reduce the number of runs."
                    )
    else:
        experiment_values = []

    max_selected_k = (
        max(experiment_values, default=1)
        if experiment_type == "Vary active window K"
        else int(k)
    )
    required_observations = int(k_prior) + max_selected_k + 1
    st.info(f"Each run needs at least **{required_observations:,} complete observations**.")
    if custom_prices is not None and len(custom_prices) < required_observations:
        st.warning(
            f"The uploaded CSV has {len(custom_prices)} observations, but the chosen windows need "
            f"{required_observations}. Reduce the window lengths or upload more data."
        )

if data_ready and available_assets >= 3:
    if experiment_type == "Vary portfolio size N":
        st.caption(f"Fixed for every run: k_target = {fixed_k_target}.")
        if data_mode == "csv":
            st.caption(
                f"For CSV data, each N uses the first N columns. {available_assets} assets are available."
            )
    elif experiment_type == "Vary systemic threshold k_target":
        st.caption(f"Fixed for every run: N = {fixed_n}.")
        if data_mode == "csv":
            st.caption(
                f"The fixed portfolio uses the first {fixed_n} CSV columns."
            )
        st.caption(f"Also fixed for every run: K = {k}, K_prior = {k_prior}.")
    else:
        st.caption(
            f"Fixed for every run: N = {fixed_n}, k_target = {fixed_k_target}, "
            f"K_prior = {k_prior}."
        )
        st.caption("K changes to the values entered above.")

can_run = (
    data_ready
    and available_assets >= 3
    and bool(experiment_values)
    and (custom_prices is None or len(custom_prices) >= required_observations)
)

if custom_prices is not None:
    data_signature = (
        tuple(custom_prices.columns),
        len(custom_prices),
        int(pd.util.hash_pandas_object(custom_prices, index=True).sum()),
    )
else:
    data_signature = (tuple(all_tickers or []), tuple(selected_tickers or []))

configuration_signature = (
    experiment_type,
    data_mode,
    data_signature,
    target_date,
    interval,
    int(k_prior),
    int(k),
    fixed_n,
    fixed_k_target,
    tuple(experiment_values),
    st.session_state.get("n_experiment_replace_incomplete", True),
)
signature_state_key = f"{RESULT_STATE_KEY}_configuration"
if st.session_state.get(signature_state_key) != configuration_signature:
    st.session_state.pop(RESULT_STATE_KEY, None)

if st.button(
    "Run experiment",
    type="primary",
    use_container_width=True,
    disabled=not can_run,
):
    config_n = (
        fixed_n
        if experiment_type in [
            "Vary systemic threshold k_target",
            "Vary active window K",
        ]
        else None
    )
    config = ExperimentConfig(
        data_mode=data_mode,
        k_prior=int(k_prior),
        k=int(experiment_values[0] if experiment_type == "Vary active window K" else k),
        k_target=int(fixed_k_target or 3),
        custom_prices=(
            custom_prices.iloc[:, :config_n].copy()
            if custom_prices is not None and config_n is not None
            else custom_prices
        ),
        all_tickers=all_tickers,
        selected_tickers=(
            selected_tickers[:config_n]
            if selected_tickers is not None and config_n is not None
            else selected_tickers
        ),
        target_date=target_date,
        interval=interval,
        replace_incomplete=(
            st.session_state.get("n_experiment_replace_incomplete", True)
            if data_mode == "market"
            else True
        ),
    )
    try:
        with st.spinner("Preparing data and running each configuration..."):
            if experiment_type == "Vary portfolio size N":
                experiment = run_portfolio_size_experiment(config, experiment_values)
            elif experiment_type == "Vary systemic threshold k_target":
                experiment = run_k_target_experiment(config, experiment_values)
            else:
                experiment = run_k_experiment(config, experiment_values)
            st.session_state[RESULT_STATE_KEY] = experiment
            st.session_state["experiment_result_type"] = experiment_type
            st.session_state[signature_state_key] = configuration_signature
    except Exception as exc:
        st.error(f"Could not start the experiment: {exc}")

if RESULT_STATE_KEY in st.session_state:
    st.divider()
    _render_results(st.session_state[RESULT_STATE_KEY])
