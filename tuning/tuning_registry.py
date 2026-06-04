from random import random

from tuning.manual_search import ManualHoldoutTuner
from tuning.grid_search import GridCVTuner
from tuning.random_search import RandomCVTuner

def get_tuner(model_name, model_config, tracker, scoring, cv=5, random_state=42):
    tuning_method = model_config["tuning_method"]
    fixed_params = model_config.get("fixed_params", {})
    params = model_config.get("params", {})
    search_params = model_config.get("search_params", {})

    if tuning_method == "manual_holdout":
        return ManualHoldoutTuner(
            model_name=model_name,
            fixed_params=fixed_params,
            params=params,
            tracker=tracker,
            scoring=scoring,
            val_size=search_params.get("val_size", 0.2),
            random_state=random_state
        )

    elif tuning_method == "grid_cv":
        return GridCVTuner(
            model_name=model_name,
            fixed_params=fixed_params,
            params=params,
            tracker=tracker,
            scoring=scoring,
            cv=cv
        )

    elif tuning_method == "random_cv":
        return RandomCVTuner(
            model_name=model_name,
            fixed_params=fixed_params,
            params=params,
            tracker=tracker,
            scoring=scoring,
            cv=cv,
            n_iter=search_params.get("n_iter", 10),
            random_state=random_state
        )

    else:
        raise ValueError(f"Unknown tuning method: {tuning_method}")
