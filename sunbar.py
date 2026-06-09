import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tuning.tuning_registry import get_tuner
from utils.configuration import Configuration
from utils.tracker import Tracker
from utils.util import *
from evaluation.evaluation import *

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.align import Align
from rich import box
from pyfiglet import Figlet

console = Console()

def print_logo(text="NEWSVALI", clear=True):
    console.clear()

    figlet = Figlet(font="slant")
    logo = figlet.renderText(text)

    console.print(
        Align.center(
            f"[bold cyan]{logo}[/bold cyan]"
        )
    )

def print_subtitle():
    console.print(
        Align.center(
            "[bold white]ML EXPERIMENT MANAGER[/bold white]\n"
            "[dim]train • tune • test • track[/dim]"
        )
    )

def select_model(model_configs):
    show_available_models(model_configs)

    model_names = list(model_configs.keys())

    model_name = questionary.autocomplete(
        "What model do you want to play:",
        choices=model_names,
        match_middle=True
    ).ask()

    if model_name not in model_configs:
        console.print(f"[bold red]Invalid model name:[/bold red] {model_name}")
        return None

    return model_name

def show_available_models(model_configs):
    print_logo()
    print_subtitle()

    console.print("\n[bold cyan]Available models[/bold cyan]\n")

    for model_name, config in model_configs.items():
        console.print(
            f"[bold green]{model_name}[/bold green] "
            f"[yellow]({config.get('tuning_method', 'unknown')})[/yellow]"
        )

def run_training(model_name, model_configs, X_train_tfidf, y_train, X_test_tfidf, y_test, tracker, scoring, random_seed,run_test=True):
    model_config = model_configs[model_name]
    tuning_method = model_config["tuning_method"]

    tuner = get_tuner(
        model_name,
        model_config=model_config,
        tracker=tracker,
        scoring=scoring,
        cv=5,
        random_state=random_seed
    )

    console.print(
        f"\n[bold cyan]Training is ready...[/bold cyan] "
        f"Model: [bold green]{model_name}[/bold green] | "
        f"Tuning method: [yellow]{tuning_method}[/yellow]\n"
    )

    console.print("[bold cyan]##### Hyperparameter Search #####[/bold cyan]")
    search_result = tuner.fit(X_train_tfidf, y_train)

    console.print("\n[bold green]Hyperparameter experiment is finished.[/bold green]")
    console.print(f"Best params: {search_result['best_params']}")
    console.print(f"Best training score: {search_result['best_train_score']:.4f}")
    console.print(f"Best validation score: {search_result['best_val_score']:.4f}")

    if run_test:
        console.print("\n[bold cyan]##### Final Test Evaluation #####[/bold cyan]")

        test_result = evaluation_final(
            model=search_result["best_model"],
            X_test=X_test_tfidf,
            y_test=y_test,
            tracker=tracker,
            best_params=search_result["best_params"]
        )

        console.print("[bold green]Evaluation is finished.[/bold green]")

    return search_result

def train_flow(model_name, model_configs, X_train_tfidf, y_train, X_test_tfidf, y_test, tracker, scoring, random_seed):
    print_logo()
    print_subtitle()

    console.print(
        Panel.fit(
            "[bold cyan]TRAIN / TUNE MODE[/bold cyan]",
            border_style="cyan"
        )
    )

    if model_name is None:
        return

    save_best = Confirm.ask(
        "[bold yellow]Save best model?[/bold yellow]",
        default=True
    )

    show_result = Confirm.ask(
        "[bold yellow]Show the results after training?[/bold yellow]",
        default=True
    )

    run_test = Confirm.ask(
        "[bold yellow]Run final test after tuning?[/bold yellow]",
        default=True
    )

    console.print(
        Panel(
            f"[bold]Model:[/bold] {model_name}\n"
            f"[bold]Save best model:[/bold] {save_best}\n"
            f"[bold]Show the training result:[/bold] {show_result}\n"
            f"[bold]Run final test:[/bold] {run_test}",
            title="[bold green]Experiment Summary[/bold green]",
            border_style="green"
        )
    )

    start = questionary.confirm("[bold cyan]Start experiment?[/bold cyan]", default=True).ask()

    if not start:
        console.print("[yellow]Cancelled.[/yellow]")
        return

    search_result = run_training(
        model_name=model_name,
        model_configs=model_configs,
        X_train_tfidf=X_train_tfidf,
        y_train=y_train,
        X_test_tfidf=X_test_tfidf,
        y_test=y_test,
        tracker=tracker,
        scoring=scoring,
        random_seed=random_seed,
        run_test=run_test
    )

    console.print("\n[bold green]Experiment finished.[/bold green]")

    if show_result:
        show_cv_result_chart(
            search_result=search_result,
            output_dir=tracker.save_dir,
            model_name=model_name
        )

def test_saved_model(model_name, model_configs):
    # TODO: add evaluation code(evaluation.py)
    console.print(f"[yellow]TODO: test saved model for {model_name}[/yellow]")

def test_flow(model_name, model_configs):
    console.clear()
    print_logo()

    console.print(
        Panel.fit(
            "[bold cyan]TEST MODE[/bold cyan]",
            border_style="cyan"
        )
    )

    test_saved_model(model_name=model_name, model_configs=model_configs)

    console.print("\n[bold green]Test finished.[/bold green]")

def show_cv_result_chart(search_result, output_dir, model_name):
    df = get_cv_results_df(search_result)

    x_col, y_col = choose_chart_columns(df)

    if x_col is None or y_col is None:
        console.print("[yellow]Chart selection cancelled.[/yellow]")
        return

    # y-axis should be numeric
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        console.print("[bold red]Y-axis must be numeric.[/bold red]")
        return

    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(9, 5))

    # numeric x-axis -> scatter plot
    if pd.api.types.is_numeric_dtype(df[x_col]):
        ax = sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue="rank",
            size="mean_fit_time",
            sizes=(60, 180),
            palette="viridis",
            legend="brief"
        )

        # add error bars only for validation score
        if y_col == "mean_val_score" and "std_val_score" in df.columns:
            plt.errorbar(
                df[x_col],
                df[y_col],
                yerr=df["std_val_score"],
                fmt="none",
                ecolor="gray",
                alpha=0.5,
                capsize=3
            )

    # categorical x-axis -> bar plot
    else:
        plot_df = df.copy()
        plot_df[x_col] = plot_df[x_col].astype(str)

        ax = sns.barplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue="rank",
            palette="viridis"
        )

        if y_col == "mean_val_score" and "std_val_score" in df.columns:
            for i, row in plot_df.reset_index(drop=True).iterrows():
                ax.errorbar(
                    x=i,
                    y=row[y_col],
                    yerr=row["std_val_score"],
                    fmt="none",
                    ecolor="gray",
                    capsize=3
                )

    ax.set_title(f"{model_name}: {x_col} vs {y_col}", fontsize=14, weight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = os.path.join(
        output_dir,
        f"{model_name}_{x_col}_vs_{y_col}.png"
    )

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    console.print(f"[bold green]Chart saved to:[/bold green] {plot_path}")


def main():
    # ------------------------------------------
    # Prepare Dataset
    # ------------------------------------------
    vectorizer_path = "dataset/tfidf_vectorizer.pkl"
    x_train_path = "dataset/X_train_tfidf.npz"
    x_test_path = "dataset/X_test_tfidf.npz"
    y_train_path = "dataset/y_train.npy"
    y_test_path = "dataset/y_test.npy"

    vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test = (
        load_data(vectorizer_path, x_train_path, x_test_path, y_train_path, y_test_path))

    # ------------------------------------------
    # Configuration Setup
    # ------------------------------------------
    print("\n##### Start Environment Setting #####")
    config = Configuration.from_file("config.json")

    output_dir = config.get_str("general.output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    random_seed = config.get_int("general.random_seed", 42)
    scoring = config.get_str("general.scoring", "f1")

    #tracker_columns = config.get("general.tracker.columns", None)
    """
    tracker = Tracker(
        save_dir=output_dir,
        experiment_name="",
        filename="metrics.csv"
        #columns=tracker_columns
    )
    """

    # load all model configs
    model_configs = load_all_model_configs("model_param_grids.json")

    while True:
        print_logo()
        print_subtitle()

        action = questionary.select(
            "What do you want to do?",
            choices=[
                "Train / tune a model",
                "Test saved best model",
                "Show available models",
                "Show experiment results",
                "Exit"
            ]
        ).ask()

        if action == "Train / tune a model":
            model_name = select_model(model_configs)

            tracker = Tracker(
                save_dir=output_dir,
                experiment_name=model_name,
                filename="metrics.csv"
            )

            train_flow(
                model_name=model_name,
                model_configs=model_configs,
                X_train_tfidf=X_train_tfidf,
                y_train=y_train,
                X_test_tfidf=X_test_tfidf,
                y_test=y_test,
                tracker=tracker,
                scoring=scoring,
                random_seed=random_seed
            )

            input("\nPress Enter to return to menu...")

        elif action == "Test saved best model":
            model_name = select_model(model_configs)

            tracker = Tracker(
                save_dir=output_dir,
                experiment_name=model_name,
                filename="metrics.csv"
            )

            test_flow(model_name, model_configs)
            input("\nPress Enter to return to menu...")

        elif action == "Show available models":
            print_logo()
            show_available_models(model_configs)
            input("\nPress Enter to return to menu...")

        elif action == "Show experiment results":
            console.print("[yellow]Result viewer will be added later.[/yellow]")
            input("\nPress Enter to return to menu...")

        elif action == "Exit":
            # console.print("[bold cyan]!ByeBye![/bold cyan]")
            print_logo("!ByeBye!")
            break


if __name__ == "__main__":
    main()
