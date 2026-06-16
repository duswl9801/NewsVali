import numpy as np
import joblib

from evaluation.metrics_clf import *
from utils.util import *

def get_predictions(model, X, threshold=0.5):
    raw_pred = model.predict(X)

    # custom model returns (prob, pred)
    if isinstance(raw_pred, tuple) and len(raw_pred) == 2:
        y_score, y_pred = raw_pred
        return np.asarray(y_score), np.asarray(y_pred)

    # sklearn model with probability
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
        return y_score, y_pred

    # LinearSVC / SVM without probability
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
        y_pred = (y_score >= 0).astype(int)
        return y_score, y_pred

    # fallback
    y_pred = np.asarray(raw_pred)
    y_score = y_pred
    return y_score, y_pred

def evaluate(model, X_test, y_test, tracker, threshold=0.5, phase="test", best_params=None):
    y_prob, y_pred = get_predictions(model, X_test, threshold=threshold)

    acc = accuracy(y_test, y_pred)
    p = precision(y_test, y_pred, positive_label=1)
    r = recall(y_test, y_pred, positive_label=1)
    f1 = f1_score(y_test, y_pred, positive_label=1)

    tp, fp, fn, tn = confusion_counts(y_test, y_pred, positive_label=1)

    print("\n================ Evaluation Results ================")
    print(f"Prob: {y_prob} | Accuracy: {acc:.4f} | Suspicious Precision: {p:.4f} | Suspicious Recall: {r:.4f} | Suspicious F1-score: {f1:.4f}")

    print("\nConfusion Matrix Counts:")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    result = {
        "phase": phase,
        "accuracy": acc,
        "suspicious_precision": p,
        "suspicious_recall": r,
        "suspicious_f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
        "best_params": best_params,
    }

    if tracker is not None:
        tracker.write_row(result)

    return result

def evaluate_saved_model(model_path, X_test, y_test, tracker=None, threshold=0.5):
    model = joblib.load(model_path)

    return evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        tracker=tracker,
        threshold=threshold,
        phase="saved_model_test",
    )