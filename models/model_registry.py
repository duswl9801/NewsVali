from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from models.MyLogisticRegression import MyLogisticRegression

def get_model(model_name, fixed_params=None):
    if fixed_params is None:
        fixed_params = {}

    if model_name == "my_logistic_regression":
        return MyLogisticRegression(**fixed_params)
    elif model_name == "lr": # logistic regression
        return LogisticRegression(**fixed_params)
    elif model_name == "rf": # random forest
        return RandomForestClassifier(**fixed_params)
    elif model_name == "gb": # gradient boosting
        return GradientBoostingClassifier(**fixed_params)
    elif model_name == "svm": # support vector machine
        return LinearSVC(**fixed_params)
    elif model_name == "mlp": # multiple layer perceptron
        return LinearSVC(**fixed_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



def old_get_model(model_name, params, random_state=42):
    if model_name == "my_logistic_regression":
        return MyLogisticRegression(epochs=params["epochs"], lr=params["learning_rate"], threshold=params["threshold"])
    elif model_name == "lr": # logistic regression
        return LogisticRegression(
            C=params["C"],
            max_iter=params["max_iter"]
        )
    elif model_name == "rf": # random forest
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            random_state=random_state
        )
    elif model_name == "gb": # gradient boosting
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            random_state=random_state
        )
    elif model_name == "svm": # support vector machine
        return LinearSVC(
            C=params["C"],
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")