import os

from utils.configuration import Configuration
from utils.tracker import Tracker
from models.MyLogisticRegression import MyLogisticRegression
from utils.metrics_clf import *
from utils.util import *

"""

"""
def main():

    # ------------------------------------------
    # prepare dataset
    # ------------------------------------------
    vectorizer_path = "../dataset/tfidf_vectorizer.pkl"
    x_train_path = "../dataset/X_train_tfidf.npz"
    x_test_path = "../dataset/X_test_tfidf.npz"
    y_train_path = "../dataset/y_train.npy"
    y_test_path = "../dataset/y_test.npy"

    vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test = (
        load_data(vectorizer_path, x_train_path, x_test_path, y_train_path, y_test_path))

    print("##### Start Environment Setting #####")
    #############################################
    # Options:
    # "lr" - logistic regression
    # "svm" - support vector machine
    # "rf" - random forest
    # "gb" - gradient boosting
    # "mlp" - multiple layer perceptron
    #############################################
    model_name = "rf"


    config = Configuration.from_file("config.json")

    output_dir = config.get_str("general.output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    tracker = Tracker(
        save_dir=output_dir,
        experiment_name=model_name,
        filename="metrics.csv"
    )


    model_params = Configuration.load_model_config("model_param_grids.json", model_name)

    # tuner
    tuning_

    print(f"Training is ready... Model: {model_name} | Tuning_method: {tuning_method} | Output directory: {output_dir}\n")

    print("\n================ Training with Validation Evaluation ================")
    #model = train(X_train_tfidf, y_train, tracker=tracker, epochs=epochs, lr=lr, threshold=threshold)

    learning_rates = [6.0, 7.0, 8.0, 8.5, 9.0, 9.5]

    best_lr = None
    best_f1 = -1

    for lr in learning_rates:
        print(f"\n===== lr = {lr} =====")
        model, f1 = train(X_train_tfidf, y_train, tracker=tracker, epochs=epochs, lr=lr, threshold=threshold)

        if f1 > best_f1:
            best_f1 = f1
            best_lr = lr

    print(f"\nBest learning rate: {best_lr} | Best validation F1: {best_f1:.4f}")

    print("\n================ Final Training ================")
    final_model = MyLogisticRegression(epochs, best_lr, threshold)
    final_model.fit(X_train_tfidf, y_train, tracker=tracker)

    print("\n================ Testing ================")
    evaluate(final_model, X_test_tfidf, y_test, tracker=tracker)

if __name__ == "__main__":
    main()