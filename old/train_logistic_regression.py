import os

from old.utils import Configuration
from old.utils import Tracker
from models.MyLogisticRegression import MyLogisticRegression
from old.utils import *

from sklearn.model_selection import train_test_split

# training and validation
def train(X_train, y_train, tracker, epochs, lr, threshold):
    print("\n================ Split data ================")
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    model = MyLogisticRegression(epochs, lr, threshold)

    model.fit(X_train_sub, y_train_sub, tracker=tracker)

    y_prob = model.predict_proba(X_val)

    thresholds = [0.35, 0.4, 0.45, 0.5]

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        acc = accuracy(y_val, y_pred)
        p = precision(y_val, y_pred, positive_label=1)
        r = recall(y_val, y_pred, positive_label=1)
        f1 = f1_score(y_val, y_pred, positive_label=1)

        print(f"\nthreshold = {threshold}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall: {r:.4f}")
        print(f"F1: {f1:.4f}")

    '''
    y_prob, y_pred = model.predict(X_val)

    acc = accuracy(y_val, y_pred)
    p = precision(y_val, y_pred, positive_label=1)
    r = recall(y_val, y_pred, positive_label=1)
    f1 = f1_score(y_val, y_pred, positive_label=1)

    print("first loss:", model.loss_history[0])
    print("last loss:", model.loss_history[-1])
    print(f"Val Accuracy: {acc:.4f}")
    print(f"Val Suspicious Precision: {p:.4f}")
    print(f"Val Suspicious Recall: {r:.4f}")
    print(f"Val Suspicious F1: {f1:.4f}")

    tracker.write_row({
        "phase": "validation",
        "lr": lr,
        "first_loss": model.loss_history[0],
        "last_loss": model.loss_history[-1],
        "accuracy": acc,
        "suspicious_precision": p,
        "suspicious_recall": r,
        "suspicious_f1": f1
    })
    '''

    return model, f1

def evaluate(model, X_test, y_test, tracker):
    y_prob, y_pred = model.predict(X_test)

    acc = accuracy(y_test, y_pred)
    p = precision(y_test, y_pred, positive_label=1)
    r = recall(y_test, y_pred, positive_label=1)
    f1 = f1_score(y_test, y_pred, positive_label=1)

    tp, fp, fn, tn = confusion_counts(y_test, y_pred, positive_label=1)

    print("\n================ Evaluation Results ================")
    print(f"Prob: {y_prob} | Accuracy: {acc:.4f} | Suspicious Precision: {p:.4f} | Suspicious Recall: {r:.4f} | Suspicious F1-score: {f1:.4f}")

    print("\nConfusion Matrix Counts:")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    tracker.write_row({
        "phase": "test",
        "accuracy": acc,
        "suspicious_precision": p,
        "suspicious_recall": r,
        "suspicious_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    })

    return y_pred

def main():

    config = Configuration.from_file("../config.json")

    output_dir = config.get_str("general.output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    experiment_name = config.get_str("general.experiment_name", "lr_tfidf_baseline")
    save_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    epochs = config.get_int("model.epochs", 50)
    lr = config.get_float("model.learning_rate", 0.001)
    threshold = config.get_float("model.threshold", 0.5)

    tracker = Tracker(
        save_dir=save_dir,
        experiment_name=experiment_name,
        filename="metrics.csv"
    )

    vectorizer_path = "../dataset/tfidf_vectorizer.pkl"
    x_train_path = "../dataset/X_train_tfidf.npz"
    x_test_path = "../dataset/X_test_tfidf.npz"
    y_train_path = "../dataset/y_train.npy"
    y_test_path = "../dataset/y_test.npy"

    vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test = load_data(vectorizer_path, x_train_path, x_test_path, y_train_path, y_test_path)

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