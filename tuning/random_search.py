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
            return_train_score=True
        )

        search.fit(X_train, y_train)

        # TODO: tracker
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        # save best model
        best_model_path = self._save_model(best_model, note="best_model")

        train_pred = best_model.predict(X_train)
        train_acc = accuracy(y_train, train_pred)

        # save best model row
        self.tracker.write_row({
            "model_name": self.model_name,
            "tuning_method": "random_cv",
            "hyperparams": json.dumps(best_params),

            "train_acc": train_acc,
            "train_loss": "",
            "val_acc": best_score,
            "val_loss": "",

            "random_seed": self.random_state,
            "note": ""
        })

        return {
            "best_model": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": search.cv_results_
        }