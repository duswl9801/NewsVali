import os

from tuning.tuning_registry import get_tuner
from utils.configuration import Configuration
from utils.tracker import Tracker
from utils.util import *
from evaluation.evaluation import *

"""

"""
def main():

    # ------------------------------------------
    # Prepare Dataset
    # ------------------------------------------
    vectorizer_path = "../dataset/tfidf_vectorizer.pkl"
    x_train_path = "../dataset/X_train_tfidf.npz"
    x_test_path = "../dataset/X_test_tfidf.npz"
    y_train_path = "../dataset/y_train.npy"
    y_test_path = "../dataset/y_test.npy"

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

    tracker_columns = config.get("general.tracker.columns", None)
    tracker = Tracker(
        save_dir=output_dir,
        experiment_name="",
        filename="metrics.csv",
        columns=tracker_columns
    )

    #############################################
    # Options:
    # "lr" - logistic regression
    # "svm" - support vector machine
    # "rf" - random forest
    # "gb" - gradient boosting
    # "mlp" - multiple layer perceptron
    #############################################
    model_name = "rf"
    # call configs matched with each models
    model_config = Configuration.load_model_config("model_param_grids.json", model_name)

    # scoring
    scoring = config.get_str("general.scoring", "f1")
    # tuner
    tuning_method = model_config["tuning_method"]
    # choose tuner from model config
    tuner = get_tuner(model_name, model_config=model_config, tracker=tracker, scoring=scoring, cv=5, random_state=random_seed)

    print(f"Training is ready... Model: {model_name} | Tuning_method: {tuning_method} | Output directory: {output_dir}\n")

    print("\n##### Hyperparameter Search #####")
    search_result = tuner.fit(X_train_tfidf, y_train)

    print("Hyperparameter experiment is finished...")
    print(f"Best params: {search_result['best_params']} | Best validation score: {search_result['best_score']:.4f}\n")

    # TODO: load best model

    #print("\n##### Final Test Evaluation #####")
    """
    test_result = evaluation_final(
        model=search_result["best_model"],
        X_test=X_test_tfidf,
        y_test=y_test,
        tracker=tracker,
        best_params=search_result["best_params"]
    )
    """

    #print("Evaluation is finished...")
    #print(f"Best params: {search_result["best_params"]} | Best validation score: {search_result["best_score"]:.4f}\n")
    #print(f"Result saved in: {output_dir}")

if __name__ == "__main__":
    main()