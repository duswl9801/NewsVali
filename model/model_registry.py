from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

def get_model(model_name, fixed_params=None):
    if fixed_params is None:
        fixed_params = {}

    if model_name == "lr" or model_name.startswith("lr_"): # logistic regression
        return LogisticRegression(**fixed_params)
    elif model_name == "rf" or model_name.startswith("rf_"): # random forest
        return RandomForestClassifier(**fixed_params)
    elif model_name == "gb" or model_name.startswith("gb_"): # gradient boosting
        return GradientBoostingClassifier(**fixed_params)
    elif model_name == "svm" or model_name.startswith("svm_"): # support vector machine
        return LinearSVC(**fixed_params)
    elif model_name == "mlp" or model_name.startswith("mlp_"): # multiple layer perceptron
        return LinearSVC(**fixed_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")