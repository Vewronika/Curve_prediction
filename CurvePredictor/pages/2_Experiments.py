import datetime
import math
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from application.experiment_runner import (
    ExperimentConfig,
    run_b_experiment,
    run_k_experiment,
    run_k_target_experiment,
    run_portfolio_size_experiment,
)
from data_acess.db_utils import fetch_all_dataset_names, fetch_companies_by_dataset
from data_acess.provider import parse_uploaded_csv


st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")

RESULT_STATE_KEY = "experiment_result"
DEFAULT_UNIVERSE_RANGE = 50


def _is_positive_whole_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 1
        and float(value).is_integer()
    )


def _requested_size(config: ExperimentConfig) -> int:
    if config.data_mode == "csv":
        return config.custom_prices.shape[1]
    return len(config.selected_tickers)


def _range_values(
    label: str,
    minimum: int,
    maximum: int,
    key: str,
    *,
    default_range: tuple[int, int] | None = None,
    default_step: int = 1,
) -> list[int]:
    if minimum == maximum:
        st.info(f"Only {minimum} is available for {label}.")
        return [minimum]

    selected_range = st.slider(
        f"{label} range",
        min_value=minimum,
        max_value=maximum,
        value=default_range or (minimum, maximum),
        key=f"{key}_range_{minimum}_{maximum}",
    )
    start, end = selected_range
    step = st.number_input(
        f"Increment between tested {label} values",
        value=default_step,
        step=1,
        key=f"{key}_step_{minimum}_{maximum}",
        help="Step 1 tests every value. Larger steps skip intermediate values; the selected end value is still included.",
    )
    if not _is_positive_whole_number(step):
        st.error("Step must be a whole number of at least 1.")
        return []
    max_step = max(1, maximum - minimum)
    if step > max_step:
        st.error(f"Step must be no greater than {max_step} for this selected range.")
        return []
    values = list(range(start, end + 1, int(step)))
    if values[-1] != end:
        values.append(end)
    st.caption(f"Values tested: {', '.join(map(str, values))} · {len(values)} runs")
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
                "B (%)": run.config.b_percent,
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
        "Vary barrier B": ("B (%)", "Barrier B (%)"),
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
                else run.config.b_percent
                if x_column == "B (%)"
                else (
                    run.config.k
                    if x_column == "K"
                    else _requested_size(run.config)
                ),
                "Crash probability (%)": float(probability) * 100,
                "Usable N": run.prediction.asset_count,
                "k_target": run.config.k_target,
                "B (%)": run.config.b_percent,
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
        elif experiment_type == "Vary barrier B":
            title = f"Details for B = {run.config.b_percent:g}% (N = {requested_n})"
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
    "These are one-factor-at-a-time sensitivity experiments: the model is rerun with selected parameter values, and its estimated crash probabilities are compared. They do not train a machine-learning model."
)

experiment_type = st.radio(
    "What do you want to vary?",
    [
        "Vary portfolio size N",
        "Vary systemic threshold k_target",
        "Vary active window K",
        "Vary barrier B",
    ],
    horizontal=True,
    key="experiment_type_selector",
    captions=[
        "Does the estimate change with portfolio size?",
        "Does the estimate change with the event threshold?",
        "Does the estimate change with active-window length?",
        "Does the estimate change with barrier B?",
    ],
)

experiment_guides = {
    "Vary portfolio size N": (
        "How does the estimated systemic-crash probability change as the portfolio gets larger?",
        "N changes. The selected k_target, B, K, and K_prior stay fixed. Each N uses a nested subset of the selected assets: the first N CSV columns or the first N assets in the chosen market-universe range.",
    ),
    "Vary systemic threshold k_target": (
        "How does the estimate change when a systemic event requires more assets to default at the same time?",
        "k_target changes while N, B, K, and K_prior stay fixed. k_target is a count of assets, not a time-window length.",
    ),
    "Vary active window K": (
        "How sensitive is the estimate to the length of the active time period?",
        "K changes while K_prior, N, k_target, and B stay fixed. First K and Last K set the smallest and largest active-window sizes tested; Increment sets the spacing. K counts time steps, so K=10 uses 11 price observations for the active window. The last value is included even when the increment does not land on it.",
    ),
    "Vary barrier B": (
        "How sensitive is the estimate to the algorithm's barrier threshold?",
        "B changes while N, k_target, K, and K_prior stay fixed. Values are selected as percentages and converted to decimal form before they reach the algorithm; for example, -10% becomes -0.1.",
    ),
}
guide_question, guide_description = experiment_guides[experiment_type]
with st.container(border=True):
    st.subheader("What this experiment tests")
    st.write(guide_question)
    st.caption(guide_description)

with st.expander("Parameter guide: N, k_target, K, K_prior, and B"):
    st.markdown(
        "- **N**: number of assets in one run.\n"
        "- **k_target**: number of assets that must default together for the model's systemic event.\n"
        "- **K**: number of time steps in the active window; K steps use K+1 price observations.\n"
        "- **K_prior**: number of time steps in the prior reference window; the prior and active windows share one boundary price observation.\n"
        "- **B**: algorithm barrier threshold, entered here as a percentage."
    )
    st.caption(
        "Uppercase K describes time windows; lowercase k_target counts assets. "
        "These experiments compare model outputs and do not train a machine-learning model."
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
    target_date_valid = True

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
        if target_date > datetime.date.today():
            target_date_valid = False
            st.error("Target date cannot be in the future for historical market data.")
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
        k_prior_input = st.number_input(
            "Fixed prior reference window K_prior (steps)",
            value=30,
            step=1,
            key="k_sweep_k_prior",
            help="This prior reference period stays fixed while the active window K is varied.",
        )
        prior_window_valid = _is_positive_whole_number(k_prior_input)
        window_inputs_valid = prior_window_valid
        if not prior_window_valid:
            st.error("Prior window K_prior must be a whole number of at least 1 step.")
        k_prior = int(k_prior_input) if prior_window_valid else 1
        k = 1
    else:
        separate_windows = st.toggle(
            "Set prior and active windows separately",
            key="n_experiment_separate_windows",
            help="When off, K and K_prior use the same observation count. When on, enter them independently.",
        )
        if separate_windows:
            window_col1, window_col2 = st.columns(2)
            k_input = window_col1.number_input(
                "Active window K (steps)",
                value=30,
                step=1,
                key="n_experiment_k",
                help="Number of time steps in the active period. K steps require K+1 price observations.",
            )
            k_prior_input = window_col2.number_input(
                "Prior reference window K_prior (steps)", value=30, step=1,
                key="n_experiment_k_prior",
                help="Number of time steps in the prior reference period. It shares one boundary price observation with the active window.",
            )
            k_valid = _is_positive_whole_number(k_input)
            prior_window_valid = _is_positive_whole_number(k_prior_input)
            window_inputs_valid = k_valid and prior_window_valid
            if not k_valid:
                st.error("Active window K must be a whole number of at least 1 step.")
            if not prior_window_valid:
                st.error("Prior window K_prior must be a whole number of at least 1 step.")
            k = int(k_input) if k_valid else 1
            k_prior = int(k_prior_input) if prior_window_valid else 1
        else:
            horizon = st.number_input(
                "Window length for K and K_prior (steps)",
                value=30,
                step=1,
                key="n_experiment_window_length",
                help="Uses the same number of time steps for the prior reference and active periods.",
            )
            window_inputs_valid = _is_positive_whole_number(horizon)
            if not window_inputs_valid:
                st.error("Window length must be a whole number of at least 1 step.")
            k = int(horizon) if window_inputs_valid else 1
            k_prior = int(horizon) if window_inputs_valid else 1

    if experiment_type == "Vary barrier B":
        fixed_b_percent = None
    else:
        fixed_b_percent = st.slider(
            "Fixed barrier B (%)",
            min_value=-100,
            max_value=0,
            value=-10,
            step=1,
            key="experiment_fixed_b_percent",
            help="The application converts this percent value to the decimal passed to the model. -10% is passed as -0.1.",
        )

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
                "First active window K to test",
                value=1,
                step=1,
                key=f"k_sweep_start_{available_assets}_{k_prior}_{csv_max_k}",
                help="The smallest active-window length included in this sensitivity experiment.",
            )
            k_end = k_to_col.number_input(
                "Last active window K to test",
                value=default_k_end,
                step=1,
                key=f"k_sweep_end_{available_assets}_{k_prior}_{csv_max_k}",
                help="The largest active-window length included. This endpoint is always tested.",
            )
            k_step = k_step_col.number_input(
                "Increment between tested K values",
                value=1,
                step=1,
                key=f"k_sweep_step_{available_assets}_{k_prior}",
                help="Example: first 1, last 10, increment 3 tests K = 1, 4, 7, 10.",
            )
            k_sweep_fields_valid = (
                _is_positive_whole_number(k_start)
                and _is_positive_whole_number(k_end)
                and _is_positive_whole_number(k_step)
            )
            if not _is_positive_whole_number(k_start):
                st.error("First active window K must be a whole number of at least 1.")
            if not _is_positive_whole_number(k_end):
                st.error("Last active window K must be a whole number of at least 1.")
            if not _is_positive_whole_number(k_step):
                st.error("Increment between K values must be a whole number of at least 1.")
            if (
                k_sweep_fields_valid
                and k_max_value is not None
                and (k_start > k_max_value or k_end > k_max_value)
            ):
                st.error(
                    f"First and last active window K cannot exceed {k_max_value} for this CSV and prior window."
                )
                experiment_values = []
            elif k_sweep_fields_valid and k_end < k_start:
                st.warning("Last active window K must be greater than or equal to the first K.")
                experiment_values = []
            elif k_sweep_fields_valid and window_inputs_valid:
                experiment_values = list(range(int(k_start), int(k_end) + 1, int(k_step)))
                if experiment_values[-1] != int(k_end):
                    experiment_values.append(int(k_end))
                st.caption(
                    f"Active-window K values tested: {', '.join(map(str, experiment_values))} · "
                    f"{len(experiment_values)} runs"
                )
                if len(experiment_values) > 100:
                    st.warning(
                        f"This will run {len(experiment_values)} predictions and may take a while. "
                        "Increase the step to reduce the number of runs."
                    )
            else:
                experiment_values = []
    elif experiment_type == "Vary barrier B":
        fixed_n = st.slider(
            "Fixed portfolio size N",
            min_value=3,
            max_value=available_assets,
            value=available_assets,
            key=f"b_sweep_portfolio_size_{available_assets}",
            help="This number of assets stays fixed while B changes.",
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
                key=f"b_sweep_k_target_{fixed_n}",
                help="This threshold stays fixed while B changes.",
            )
        experiment_values = _range_values(
            "Barrier B (%)",
            minimum=-100,
            maximum=0,
            key="b_experiment_values",
            default_range=(-50, 0),
            default_step=2,
        )
    else:
        experiment_values = []

    max_selected_k = (
        max(experiment_values, default=1)
        if experiment_type == "Vary active window K"
        else int(k)
    )
    required_observations = int(k_prior) + max_selected_k + 1
    if window_inputs_valid and experiment_values:
        st.info(f"Each run needs at least **{required_observations:,} complete observations**.")
    else:
        st.info("Choose valid window and experiment values to see the data requirement.")
    if (
        window_inputs_valid
        and experiment_values
        and custom_prices is not None
        and len(custom_prices) < required_observations
    ):
        st.warning(
            f"The uploaded CSV has {len(custom_prices)} observations, but the chosen windows need "
            f"{required_observations}. Reduce the window lengths or upload more data."
        )

if (
    data_ready
    and available_assets >= 3
    and window_inputs_valid
    and target_date_valid
    and experiment_values
):
    if experiment_type == "Vary portfolio size N":
        st.caption(
            f"Fixed for every run: k_target = {fixed_k_target}, B = {fixed_b_percent}%."
        )
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
        st.caption(
            f"Also fixed for every run: K = {k}, K_prior = {k_prior}, "
            f"B = {fixed_b_percent}%."
        )
    elif experiment_type == "Vary active window K":
        st.caption(
            f"Fixed for every run: N = {fixed_n}, k_target = {fixed_k_target}, "
            f"K_prior = {k_prior}, B = {fixed_b_percent}%."
        )
        st.caption("K changes to the values entered above.")
    else:
        st.caption(
            f"Fixed for every run: N = {fixed_n}, k_target = {fixed_k_target}, "
            f"K = {k}, K_prior = {k_prior}."
        )
        st.caption("B changes to the values selected above.")

can_run = (
    data_ready
    and available_assets >= 3
    and bool(experiment_values)
    and window_inputs_valid
    and target_date_valid
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
    fixed_b_percent,
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
            "Vary barrier B",
        ]
        else None
    )
    config = ExperimentConfig(
        data_mode=data_mode,
        k_prior=int(k_prior),
        k=int(experiment_values[0] if experiment_type == "Vary active window K" else k),
        k_target=int(fixed_k_target or 3),
        b_percent=float(
            experiment_values[0]
            if experiment_type == "Vary barrier B"
            else fixed_b_percent
        ),
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
            elif experiment_type == "Vary active window K":
                experiment = run_k_experiment(config, experiment_values)
            else:
                experiment = run_b_experiment(config, experiment_values)
            st.session_state[RESULT_STATE_KEY] = experiment
            st.session_state["experiment_result_type"] = experiment_type
            st.session_state[signature_state_key] = configuration_signature
    except Exception as exc:
        st.error(f"Could not start the experiment: {exc}")

if RESULT_STATE_KEY in st.session_state:
    st.divider()
    _render_results(st.session_state[RESULT_STATE_KEY])
