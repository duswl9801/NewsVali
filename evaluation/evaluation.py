from evaluation.metrics_clf import *
from utils.util import *

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

    return

def evaluation_final(best_model, vectorizor, X_test, y_test, tracker, threshold=0.5):
    # load best model
    model, vectorizor = load_model_artifacts(best_model, vectorizor)

    evaluate(model, X_test, y_test, tracker)

    # TODO: use threshold