import itertools
from sklearn.model_selection import train_test_split

from models.model_registry import get_model

"""
ManualHoldoutTuner is for manual hyperparameter search.

Manual search means that the user directly defines specific hyperparameter
combinations to test. Instead of automatically trying every possible combination,
the experiment old only the configurations that are explicitly listed.

Example:
    [
        {"C": 0.1, "kernel": "linear"},
        {"C": 1.0, "kernel": "linear"},
        {"C": 10.0, "kernel": "rbf"}
    ]
"""
class ManualHoldoutTuner:
    def __init__(self, model_name, fixed_params, params, tracker, scoring, val_size=0.2, random_state=42):
        self.model_name = model_name
        self.fixed_params = fixed_params
        self.params = params
        self.tracker = tracker
        self.scoring = scoring
        self.val_size = val_size
        self.random_state = random_state

    def fit(self, X_train, y_train):
        base_model = get_model(self.model_name, fixed_params=self.fixed_params)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.params,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=self.cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        # TODO: tracker

        return {
            "best_model": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": search.cv_results_
        }