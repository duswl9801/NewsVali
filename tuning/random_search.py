from sklearn.model_selection import RandomizedSearchCV
from models.model_registry import get_model

"""
RandomCVTuner is for random search hyperparameter tuning.

Random search means that the experiment randomly samples a limited number of
hyperparameter combinations from a larger search space. Unlike grid search, it
does not test every possible combination.

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