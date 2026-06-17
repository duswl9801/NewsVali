# NewsVali

### ML-Based News Verification Support Program

NewsVali is a machine learning based news verification support application. It helps users identify sentences in news articles that may require further verification. The system does not determine whether an article is true or false. Instead, it highlights potentially check-worthy sentences so that users can review them more easily.

## Main Features

* Desktop application built with CustomTkinter
* Article input and sentence-level prediction
* Highlighted article view

  * Red: needs verification
  * Yellow: suspicious
  * No highlight: likely reliable
* Output panel showing selected sentence, predicted label, model score, and rule-based explanation
* Multiple trained model options
* Custom test articles for application-level demo
* Command-line experiment manager for training, tuning, and testing models

## Project Structure

```text
NewsVali/
│
├── dataset/                    # Preprocessed data, TF-IDF vectorizer, and demo articles
├── evaluation/                 # Evaluation utilities
├── model/                      # Model definitions and registry
├── outputs/models/             # Saved trained model files
├── tuning/                     # Hyperparameter tuning logic
├── utils/                      # Utility modules
│
├── config.json                 # General project configuration
├── model_param_grids.json      # Model hyperparameter settings
├── run_app.py                  # Desktop application entry point
├── run_experiment.py           # Command-line experiment manager
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/duswl9801/NewsVali.git
cd NewsVali
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages

To run the desktop application, install the main required packages:

```bash
pip install customtkinter pandas numpy scipy scikit-learn joblib questionary rich pyfiglet
```

### 4. Check required files

Before running the app, make sure these files or folders exist:

```text
dataset/tfidf_vectorizer.pkl
dataset/self_test_articles.json
outputs/models/
config.json
```

The `outputs/models/` folder should contain saved model files such as:

```text
gradient_boosting.pkl
logistic_regression.pkl
random_forest.pkl
support_vector_machine.pkl
```

### 5. Run the desktop application

```bash
python run_app.py
```

The application window will open. You can choose a custom test article from the article dropdown, choose a trained model from the model dropdown, and inspect highlighted sentence-level predictions.

To test your own article, click **Clear**, paste article text into the left panel, and click **Analyze Article**.


## Optional: Reproducing the Machine Learning Experiments

The NewsVali desktop application can be run directly with the saved TF-IDF vectorizer and trained model files included in the repository. Users do not need to retrain the models to run the application.

To use the command-line experiment manager, make sure the additional experiment packages are installed:
```bash
pip install matplotlib seaborn dotenv
```

Then run:
```bash
python run_experiment.py
```

From the menu, you can:

```text
Train / tune a model
Test saved best model
Exit
```

The experiment manager can train or tune models based on the settings in `model_param_grids.json`. It can also load saved model files and print evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix counts.

### Dataset and Preprocessing Notes

This project uses three public fact-checking and verification-related datasets:

* FEVER
* LIAR
* MultiFC

The original labels were mapped into a binary classification setting:

```text
likely reliable
needs verification
```

The application then maps model scores into three user-facing labels:

```text
likely reliable
suspicious
needs verification
```

The final processed dataset contains:

```text
136,650 unique claims
109,320 training samples
27,330 test samples
10,000 maximum TF-IDF features
```

A small custom article-level evaluation set is also included. It contains five real articles, 37 sentences, and 19 manually modified verification-needed claims. This custom set was used only for application-level evaluation and was not used for model training.

### Models Used

The following traditional machine learning models were trained and compared:

* Logistic Regression
* Linear SVM
* Random Forest
* Gradient Boosting

All models use the same TF-IDF feature representation with unigrams and bigrams. Hyperparameter tuning was performed using random search followed by grid search.

## Notes

NewsVali is a verification support tool, not an automatic fact-checker. Highlighted sentences are not necessarily false. They are sentences that the model predicts may be worth reviewing.

The current system uses claim text and TF-IDF features, so it cannot fully understand external evidence, factual grounding, or article context. Future improvements could include evidence-aware features, Sentence-BERT embeddings, threshold tuning, and larger article-level evaluation data.
