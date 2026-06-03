import numpy as np

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(y_true == y_pred)

def confusion_counts(y_true, y_pred, positive_label=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    tn = np.sum((y_true != positive_label) & (y_pred != positive_label))

    return tp, fp, fn, tn

def precision(y_true, y_pred, positive_label=1):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred, positive_label)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true, y_pred, positive_label=1):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred, positive_label)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred, positive_label=1):
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)