import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_predict

from model.model_registry import get_model
from evaluation.metrics_clf import *

"""
GridCVTuner is for grid search hyperparameter tuning.

Grid search means that the experiment tries every possible combination of the
given hyperparameter values. Each hyperparameter is given as a list of candidate
values, and the tuner expands them into all possible combinations.

Example:
    params = {
        "C": [0.1, 1.0],
        "class_weight": [None, "balanced"]
    }

This search space has 2 x 2 = 4 combinations, and grid search tests all of them.
"""
class GridCVTuner:
    def __init__(self, model_name, fixed_params, params, tracker, scoring, cv=5):
        self.model_name = model_name
        self.fixed_params = fixed_params
        self.params = params
        self.tracker = tracker
        self.scoring = scoring
        self.cv = cv

    def _log_cv_results(self, search):
        cv_results = search.cv_results_

        for i, params in enumerate(cv_results["params"]):
            self.tracker.write_row({
                "candidate": i + 1,
                "model_name": self.model_name,
                "tuning_method": "grid_cv",
                "hyperparams": json.dumps(params),

                "mean_train_score": cv_results.get("mean_train_score", [""])[i],
                "mean_val_score": cv_results["mean_test_score"][i],
                "std_val_score": cv_results["std_test_score"][i],
                "rank": cv_results["rank_test_score"][i],

                "mean_fit_time": cv_results["mean_fit_time"][i],
                "std_fit_time": cv_results["std_fit_time"][i],
                "mean_score_time": cv_results["mean_score_time"][i],
                "std_score_time": cv_results["std_score_time"][i],

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
            "tuning_method": "grid_cv",
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
            "note": f"best_model_saved:{best_model_path}"
        })

    def _save_model(self, model, note="best_model"):
        model_dir = os.path.join(self.tracker.save_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(
            model_dir,
            f"{self.model_name}_{note}_grid_search.pkl"
        )

        joblib.dump(model, model_path)

        return model_path

    # save out-of-fold validation scores using the best hyperparameters for threshold tuning
    def _save_oof_validation_scores(self, best_params, X_train, y_train):
        final_params = self.fixed_params.copy()
        final_params.update(best_params)

        model = get_model(self.model_name, fixed_params=final_params)

        output_dir = self.tracker.save_dir
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(model, "predict_proba"):
            oof_scores = cross_val_predict(
                model,
                X_train,
                y_train,
                cv=self.cv,
                method="predict_proba",
                n_jobs=8
            )[:, 1]

            score_type = "probability"
            oof_pred = (oof_scores >= 0.5).astype(int)

        elif hasattr(model, "decision_function"):
            oof_scores = cross_val_predict(
                model,
                X_train,
                y_train,
                cv=self.cv,
                method="decision_function",
                n_jobs=8
            )

            score_type = "decision_score"
            oof_pred = (oof_scores >= 0).astype(int)

        else:
            return None

        df = pd.DataFrame({
            "sample_index": range(len(y_train)),
            "y_true": y_train,
            "oof_score": oof_scores,
            "score_type": score_type,
            "oof_pred_default": oof_pred
        })

        score_path = os.path.join(
            output_dir,
            f"{self.model_name}_oof_validation_scores.csv"
        )

        df.to_csv(score_path, index=False)

        return score_path

    def fit(self, X_train, y_train, save_best=True):
        base_model = get_model(self.model_name, fixed_params=self.fixed_params)

        search = GridSearchCV(
            estimator=base_model,
            param_grid=self.params,
            scoring=self.scoring,
            cv=self.cv,
            return_train_score=True,
            n_jobs=8,
            verbose=2
        )

        search.fit(X_train, y_train)

        # tracker - save every grid search candidate result
        self._log_cv_results(search)

        best_model = search.best_estimator_
        best_params = search.best_params_

        best_idx = search.best_index_
        cv_results = search.cv_results_

        best_train_score = cv_results["mean_train_score"][best_idx]
        best_val_score = search.best_score_

        if save_best:
            best_model_path = self._save_model(best_model, note="best_model")
        else:
            best_model_path = None

        oof_score_path = self._save_oof_validation_scores(
            best_params=best_params,
            X_train=X_train,
            y_train=y_train
        )

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
