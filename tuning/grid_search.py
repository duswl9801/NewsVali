from sklearn.model_selection import GridSearchCV

from models.model_registry import get_model

"""
GridCVTuner is for grid search hyperparameter tuning.

Grid search means that the experiment tries every possible combination of the
given hyperparameter values. Each hyperparameter is given as a list of candidate
values, and the tuner expands them into all possible combinations.

Example:
    {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"]
    }
"""
class GridCVTuner:
    def __init__(self, model_name, fixed_params, params, tracker, scoring, cv=5):
        self.model_name = model_name
        self.fixed_params = fixed_params
        self.params = params
        self.tracker = tracker
        self.scoring = scoring
        self.cv = cv

    def fit(self, X_train, y_train):
        base_model = get_model(self.model_name, fixed_params=self.fixed_params)

        search = GridSearchCV(
            estimator=base_model,
            param_grid=self.params,
            scoring=self.scoring,
            cv=self.cv,
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