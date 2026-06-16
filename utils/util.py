from scipy.sparse import load_npz
import joblib
import numpy as np
import pandas as pd

import questionary

def load_model_artifacts(model_path, vectorizer):
    vectorizer = joblib.load(vectorizer)
    model = joblib.load(model_path)

    return model, vectorizer

def load_data(vectorizer_path, x_train_path, x_test_path, y_train_path, y_test_path):
    vectorizer = joblib.load(vectorizer_path) # needed when model predicts new sentences

    X_train_tfidf = load_npz(x_train_path)
    X_test_tfidf = load_npz(x_test_path)

    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    return vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test

def load_all_model_configs(config_path):
    import json

    with open(config_path, "r") as f:
        return json.load(f)

def get_cv_results_df(search_result):
    cv_results = search_result["cv_results"]
    rows = []

    for i, params in enumerate(cv_results["params"]):
        row = {
            "candidate": i + 1,
            "rank": cv_results["rank_test_score"][i],
            "mean_train_score": cv_results.get("mean_train_score", [""])[i],
            "mean_val_score": cv_results["mean_test_score"][i],
            "std_val_score": cv_results["std_test_score"][i],
            "mean_fit_time": cv_results["mean_fit_time"][i],
            "mean_score_time": cv_results["mean_score_time"][i],
        }

        for key, value in params.items():
            row[key] = value

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("rank")
    return df

def choose_chart_columns(df):
    score_cols = [
        "mean_train_score",
        "mean_val_score",
        "std_val_score",
        "mean_fit_time",
        "mean_score_time",
        "rank"
    ]

    common_x_cols = [
        "candidate",
        "rank"
    ]

    fixed_cols = [
        "candidate",
        "rank",
        "mean_train_score",
        "mean_val_score",
        "std_val_score",
        "mean_fit_time",
        "mean_score_time"
    ]

    # model-specific hyperparameter columns
    param_cols = [
        col for col in df.columns
        if col not in fixed_cols
    ]

    x_choices = common_x_cols + param_cols
    y_choices = score_cols

    while True:
        x_col = questionary.autocomplete(
            "Choose x-axis:",
            choices=x_choices,
            match_middle=True
        ).ask()

        if x_col in df.columns:
            break

        print(f"Invalid x-axis column: {x_col}")
        print(f"Available x-axis columns: {x_choices}")

    while True:
        y_col = questionary.autocomplete(
            "Choose y-axis:",
            choices=y_choices,
            match_middle=True
        ).ask()

        if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            break

        print(f"Invalid y-axis column: {y_col}")
        print(f"Available y-axis columns: {y_choices}")

    return x_col, y_col

def show_console_bar_chart(console, search_result):
    df = get_cv_results_df(search_result)

    x_col, y_col = choose_chart_columns(df)

    if x_col not in df.columns or y_col not in df.columns:
        console.print("[bold red]Invalid column selection.[/bold red]")
        return

    if not pd.api.types.is_numeric_dtype(df[y_col]):
        console.print("[bold red]Y-axis must be numeric.[/bold red]")
        return

    max_value = df[y_col].max()

    console.print(f"\n[bold cyan]{x_col} vs {y_col}[/bold cyan]\n")

    for _, row in df.iterrows():
        label = str(row[x_col])
        value = row[y_col]

        bar_length = int((value / max_value) * 40) if max_value != 0 else 0
        bar = "█" * bar_length

        console.print(
            f"[bold green]{label:>12}[/bold green] | "
            f"[cyan]{bar}[/cyan] "
            f"{value:.4f}"
        )