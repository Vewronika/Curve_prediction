import pandas as pd


def df_to_math_inputs(df: pd.DataFrame, k_prior: int, k: int):
    # 1. The data from provider.py is ALREADY just the close prices!
    close_df = df

    # 2. Check if we have enough data
    total_needed_days = k_prior + k + 1

    if len(close_df) < total_needed_days:
        raise ValueError(
            f"Not enough data, {total_needed_days} days required for math model, DataFrame has {len(close_df)}.")

    # 3. Slice the exact timeframe needed
    recent_df = close_df.iloc[-total_needed_days:]

    prior_df = recent_df.iloc[0: k_prior + 1]
    active_df = recent_df.iloc[k_prior: k_prior + k + 1]

    # 4. Convert to the nested lists that Vladyslav's algorithm expects
    p_prior = prior_df.T.values.tolist()
    p = active_df.T.values.tolist()

    n = len(close_df.columns)

    return p_prior, p, n