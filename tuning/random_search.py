import os
import json
import joblib

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from models.model_registry import get_model
from evaluation.metrics_clf import *

"""
RandomCVTuner is for random search hyperparameter tuning.

Random search means that the experiment randomly samples a limited number of
hyperparameter combinations from a larger search space. Unlike grid search, it
does not test every possible combination.

This tuner:
1. runs RandomizedSearchCV
2. logs every searched parameter result to tracker
3. saves the best model
4. logs the best model result to tracker

Example:
    {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10]
    }
"""
class RandomCVTuner:
    def __init__(self, model_name, fixed_params, params, tracker, scoring, cv=5, n_iter=10, random_state=42):
        self.model_name = model_name
        self.fixed_params = fixed_params
        self.params = params
        self.tracker = tracker
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state

    def _log_cv_results(self, search):
        cv_results = search.cv_results_

        for i, params in enumerate(cv_results["params"]):
            self.tracker.write_row({
                "candidate": i + 1,
                "model_name": self.model_name,
                "tuning_method": "random_cv",
                "hyperparams": json.dumps(params),

                "mean_train_score": cv_results.get("mean_train_score", [""])[i],
                "mean_val_score": cv_results["mean_test_score"][i],
                "std_val_score": cv_results["std_test_score"][i],
                "rank": cv_results["rank_test_score"][i],

                "mean_fit_time": cv_results["mean_fit_time"][i],
                "std_fit_time": cv_results["std_fit_time"][i],
                "mean_score_time": cv_results["mean_score_time"][i],
                "std_score_time": cv_results["std_score_time"][i],

                "random_seed": self.random_state,
                "note": "candidate"
            })

    def _log_best_result(self, search, X_train, y_train, best_model_path):
        best_model = search.best_estimator_
        best_idx = search.best_index_

        train_pred = best_model.predict(X_train)
        train_acc = accuracy(y_train, train_pred)

        cv_results = search.cv_results_

        self.tracker.write_row({
            "candidate": "best",
            "model_name": self.model_name,
            "tuning_method": "random_cv",
            "hyperparams": json.dumps(search.best_params_),

            "mean_train_score": cv_results.get("mean_train_score", [""])[best_idx],
            "mean_val_score": search.best_score_,
            "std_val_score": cv_results["std_test_score"][best_idx],
            "rank": cv_results["rank_test_score"][best_idx],

            "mean_fit_time": cv_results["mean_fit_time"][best_idx],
            "std_fit_time": cv_results["std_fit_time"][best_idx],
            "mean_score_time": cv_results["mean_score_time"][best_idx],
            "std_score_time": cv_results["std_score_time"][best_idx],

            "final_train_acc": train_acc,
            "random_seed": self.random_state,
            "note": f"best_model_saved:{best_model_path}"
        })

    def _save_model(self, model, note="best_model"):
        model_dir = os.path.join(self.tracker.save_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(
            model_dir,
            f"{self.model_name}_{note}_random_search.pkl"
        )

        joblib.dump(model, model_path)

        return model_path

    def fit(self, X_train, y_train):
        base_model = get_model(self.model_name, fixed_params=self.fixed_params)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.params,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=self.cv,
            random_state=self.random_state,
            return_train_score=True,
            verbose=2,
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        # tracker - save every random search candidate result
        self._log_cv_results(search)

        best_model = search.best_estimator_
        best_params = search.best_params_

        best_idx = search.best_index_
        cv_results = search.cv_results_

        best_train_score = cv_results["mean_train_score"][best_idx]
        best_val_score = search.best_score_
        best_model_path = self._save_model(best_model, note="best_model")

        # calculate best model train accuracy on full training data
        # and save best model summary row
        self._log_best_result(
            search=search,
            X_train=X_train,
            y_train=y_train,
            best_model_path=best_model_path
        )

        return {
            "best_model": best_model,
            "best_params": best_params,
            "best_train_score": best_train_score,
            "best_val_score": best_val_score,
            "cv_results": cv_results,
            "best_model_path": best_model_path
        }