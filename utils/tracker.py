import os
import csv
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

"""
Experiment tracker for training/validation/test metrics.

Main features:
- log metrics during training
- save metrics to CSV
- load metrics from CSV
- visualize metrics with seaborn
"""
class Tracker:
    def __init__(self, save_dir, experiment_name="temp_experiment", filename="temp_metrics.csv", columns=None):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.file_path = os.path.join(save_dir, filename)
        self.columns = columns

        os.makedirs(save_dir, exist_ok=True)

        self.buffer = defaultdict(list)

    # -----------------------------
    # Logging
    # -----------------------------
    """
    Temporarily store one metric value.
    Example:
        tracker.log("train_loss", 0.52)
        tracker.log("val_acc", 87.2)
    """
    def log(self, key, value):
        """
        Temporarily store one metric value.
        Example:
            tracker.log("train_loss", 0.52)
            tracker.log("val_acc", 87.2)
        """
        self.buffer[key].append(float(value))

    """
    Store multiple metric values at once.
    Example:
        tracker.log_dict({
        train_loss": 0.52,
        "train_acc": 88.1
        })
    """
    def log_dict(self, metric_dict):
        for key, value in metric_dict.items():
            self.log(key, value)

    """
    Average all buffered values.
    Useful when logging batch-level losses and writing epoch-level average.
    """
    def summarize(self):
        summary = {}

        for key, values in self.buffer.items():
            if len(values) > 0:
                summary[key] = sum(values) / len(values)

        return summary

    """
    Write summarized metrics to CSV.

    Example output row:
        epoch, phase, train_loss, train_acc, val_loss, val_acc
    """
    def write_experiment(self, epoch, phase=None, extra=None, clear_buffer=True):
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "experiment": self.experiment_name,
            "epoch": epoch
        }

        if phase is not None:
            row["phase"] = phase

        row.update(self.summarize())

        if extra is not None:
            row.update(extra)

        self._append_csv(row)

        if clear_buffer:
            self.buffer.clear()

        return row

    """
    Directly write a metric row without using buffer.
    Useful when you already calculated metrics.
    """
    def write_row(self, row):
        base_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment": self.experiment_name,
        }

        base_row.update(row)
        self._append_csv(base_row)

    def _format_row(self, row):
        if self.columns is None:
            return row

        formatted = {}

        # make columns in config first
        for col in self.columns:
            formatted[col] = row.get(col, "")

        # keep extra columns if some future code adds them
        for key, value in row.items():
            if key not in formatted:
                formatted[key] = value

        return formatted

    def _append_csv(self, row):
        row = self._format_row(row)

        file_exists = os.path.exists(self.file_path)
        file_empty = (not file_exists) or os.path.getsize(self.file_path) == 0

        # if CSV already exists and new columns appear, rewrite safely
        if file_exists and not file_empty:
            old_df = pd.read_csv(self.file_path)
            new_df = pd.DataFrame([row])
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined.to_csv(self.file_path, index=False)
        else:
            with open(self.file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)

    # -----------------------------
    # Loading
    # -----------------------------
    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Metric file not found: {self.file_path}")

        df = pd.read_csv(self.file_path)

        # remove black separator rows
        df = df.dropna(how="all")

        return df

    # -----------------------------
    # Visualization
    # -----------------------------
    """
    Plot one metric.
    Example:
        tracker.plot_metric("train_loss")
        tracker.plot_metric("val_acc")
    """
    def plot_metric(self, metric, x="epoch", title=None, hue=None, save=True, show=True):
        df = self.load()

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available columns: {list(df.columns)}")

        plt.figure(figsize=(15, 10))
        sns.lineplot(data=df, x=x, y=metric, hue=hue, marker="o")

        if title is None:
            title = f"{self.experiment_name}: {metric}"
        else:
            title = title

        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(metric)
        plt.tight_layout()

        if save:
            out_path = os.path.join(self.save_dir, f"{metric}.png")
            plt.savefig(out_path, dpi=200)

        if show:
            plt.show()

        plt.close()

    """
    Plot multiple metrics separately.
    If metrics is None, automatically plot numeric columns except epoch.
    """
    def plot_metrics(self, metrics=None, x="epoch", title=None, hue=None, save=True, show=True):
        df = self.load()

        if metrics is None:
            ignore_cols = {"timestamp", "experiment", "epoch", "phase"}
            metrics = [
                col for col in df.columns
                if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])
            ]

        for metric in metrics:
            self.plot_metric(metric=metric, x=x, hue=hue, save=save, show=show)

    """
    Convenient plot function for common train/val loss and accuracy.
    It automatically uses existing columns only.
    """
    def plot_loss_acc(self, save=True, show=True):
        df = self.load()

        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df = df.dropna(subset=["epoch"])

        metric_groups = {
            "loss": [col for col in df.columns if "loss" in col.lower()],
            "accuracy": [
                col for col in df.columns
                if "acc" in col.lower() or "accuracy" in col.lower()
            ]
        }

        for group_name, cols in metric_groups.items():
            if len(cols) == 0:
                continue

            plot_df = df.melt(id_vars=["epoch"], value_vars=cols, var_name="metric", value_name="value")

            plt.figure(figsize=(15, 10))
            sns.lineplot(data=plot_df, x="epoch", y="value", hue="metric", marker="o")

            plt.title("Loss & Accuracy")
            plt.xlabel("epoch")
            plt.ylabel(group_name)
            plt.tight_layout()

            if save:
                out_path = os.path.join(self.save_dir, f"{group_name}.png")
                plt.savefig(out_path, dpi=200)

            if show:
                plt.show()

            plt.close()

    """
    Plot confusion matrix using seaborn.

    Args:
        y_true: true labels, list or numpy array
        y_pred: predicted labels, list or numpy array
        class_names: class names, e.g. ["NORMAL", "PNEUMONIA"]
        normalize: if True, show ratio instead of raw counts
    """
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, normalize=False, save=True, show=True):
        cm = confusion_matrix(y_true, y_pred)

        if normalize: # when wanna see ratio instead of count
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

        plt.figure(figsize=(6, 5))

        fmt = ".2f" if normalize else "d"
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        if save:
            out_path = os.path.join(self.save_dir, "confusion_matrix.png")
            plt.savefig(out_path, dpi=200)

        if show:
            plt.show()

        plt.close()

        return cm


    """
    Print last n rows of metric CSV.
    """
    def print_last(self, n= 5):
        df = self.load()
        print(df.tail(n))

    """
    Add an empty row to CSV for human readability.
    Useful when separating random warm-up and real training sections.
    """
    def add_blank_row(self):

        if not os.path.exists(self.file_path):
            return

        df = pd.read_csv(self.file_path)

        blank_row = {col: "" for col in df.columns}
        df = pd.concat([df, pd.DataFrame([blank_row])], ignore_index=True)

        df.to_csv(self.file_path, index=False)


# ============================================================
# Appendix
# ============================================================
# defaultdict(list)는 self.buffer = {} 랑 비슷한데, 없는 key에 접근하면 자동으로 빈 list를 만들어주는 dictionary야.
# metric 이름마다 여러 값을 list로 모으려고 쓴 구조