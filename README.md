# Fake News Detection Using Machine Learning and NLP

A beginner-friendly B.Tech mini project that classifies a news headline/article as **FAKE** or **REAL** using a labelled dataset, TF-IDF text features, and Logistic Regression. A Streamlit application provides a simple live demonstration.

> **Important limitation:** This is a text classifier, not an independent fact-checking system. A prediction does not prove that a claim is true or false. The model learns patterns from the dataset and may be wrong.

## 1. Features

- Supports the Fake and Real News Dataset (`Fake.csv` and `True.csv`) or a single labelled `news.csv`.
- Cleans empty records and duplicate text.
- Uses an 80:20 stratified train/test split.
- Uses a scikit-learn Pipeline to fit TF-IDF only on training data.
- Reports accuracy, macro precision, macro recall, macro F1-score, and a confusion matrix.
- Saves the model and evaluation outputs locally.
- Provides a Streamlit interface for trying news text.

## 2. Dataset download

Recommended educational dataset: **Fake and Real News Dataset** on Kaggle.

Dataset page: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

A Kaggle account may be required to download it. Download and extract the archive. Copy these two files into this project's `dataset` folder:

- `Fake.csv`
- `True.csv`

Expected layout:

```text
fake-news-detection/
└── dataset/
    ├── Fake.csv
    └── True.csv
```

The code assigns the label `FAKE` to rows from `Fake.csv` and `REAL` to rows from `True.csv`. It uses the `title` and `text` columns when available.

Alternatively, place one labelled CSV at `dataset/news.csv`. It must have a text column (for example `text`, `article`, `content`, or `title`) and a label column (for example `label` or `class`). Labels can be `Fake`/`Real`, `False`/`True`, or `0`/`1` where 0 means FAKE and 1 means REAL.

The dataset's Kaggle page lists a CC BY-NC-SA 4.0 license. Review its terms and cite the dataset in your report; do not redistribute the raw data without checking the license terms.

## 3. Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- Visual Studio Code
- VS Code Python extension
- Internet access for the initial dataset download and package installation

## 4. Run in VS Code (Windows PowerShell)

Open the `fake-news-detection` folder in VS Code. Select **Terminal → New Terminal**. Make sure the terminal is in the project root (the folder containing `app.py`).

Run these commands one at a time:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

The Streamlit command prints a local address, usually `http://localhost:8501`. Open that address in your browser. Keep the terminal running while demonstrating the app. Press `Ctrl+C` in the terminal to stop it.

If PowerShell blocks virtual-environment activation, you can avoid changing execution policy by running the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 5. Files created by training

After successful training, these files should appear:

```text
models/fake_news_model.joblib
reports/metrics.json
reports/classification_report.txt
reports/confusion_matrix.png
```

Metrics displayed by the app come from your actual run; they are not pre-filled.

## 6. Evaluation metrics

- **Accuracy:** fraction of test articles classified correctly.
- **Precision:** among articles predicted as a class, how many belong to that class.
- **Recall:** among actual articles in a class, how many the model found.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** counts of actual versus predicted classes. Rows are actual labels and columns are predicted labels.

The script reports macro-averaged precision, recall, and F1-score so both classes contribute equally to the average.

## 7. Example text for interface testing

These are **illustrative test inputs, not verified news stories**. The model may return either class for any example; do not hard-code an expected result.

1. `City officials announced that the public library will reopen on Monday after scheduled maintenance.`
2. `A viral post claims that drinking a special juice guarantees everyone will become rich overnight.`
3. `The transport department published an update about changes to bus routes starting next month.`

Use your own sample articles for the live demonstration. Explain that plausible wording alone does not establish whether a story is true.

## 8. Troubleshooting

### `python` is not recognized
Install Python from https://www.python.org/downloads/ and select the option to add Python to PATH during installation. Restart VS Code and run `python --version`.

### Activation is blocked
Use the direct `.venv\Scripts\python.exe` commands shown above instead of changing PowerShell security settings.

### `ModuleNotFoundError`
Activate the environment or use `.venv\Scripts\python.exe -m pip install -r requirements.txt`. Check that VS Code is using the `.venv` interpreter.

### Dataset not found
Make sure the CSV files are inside the `dataset` folder, not the project root. File names should be `Fake.csv` and `True.csv`, or use `news.csv`.

### Incorrect columns or labels
Inspect the CSV header row. For a single CSV, make sure there is a text column and a label column. Use labels such as `Fake` and `Real` or `0` and `1` (0 = FAKE, 1 = REAL).

### Model file not found in the app
Run `python train_model.py` successfully before starting Streamlit.

### Port already in use
Run `python -m streamlit run app.py --server.port 8502` and open the address printed in the terminal.

## 9. Five-minute classroom presentation plan

1. **Introduction (30 seconds):** Explain the challenge of misleading information online.
2. **Objective (30 seconds):** Classify news text into two dataset labels using supervised machine learning.
3. **Dataset (40 seconds):** Show the labelled Fake and Real CSV files and explain the source.
4. **Method (60 seconds):** Explain text input → TF-IDF → Logistic Regression → prediction.
5. **Live demo (60 seconds):** Paste a sample into the Streamlit app and click Predict.
6. **Evaluation (50 seconds):** Show the actual accuracy, precision, recall, F1-score, and confusion matrix.
7. **Limitations and conclusion (30 seconds):** Predictions depend on dataset quality and do not prove factual truth.

### Short speech

“Good morning respected faculty and my dear friends. My mini project is Fake News Detection Using Machine Learning and Natural Language Processing. The objective is to classify a news article as fake or real based on patterns learned from labelled examples. I use TF-IDF to convert text into numerical features and Logistic Regression to learn a classification model. I split the dataset into training and testing sets and evaluate it using accuracy, precision, recall, F1-score, and a confusion matrix. The Streamlit interface allows a user to enter news text and see the model's prediction. This system is a learning demonstration, not a definitive fact-checking tool, so its output should be checked against reliable sources. Thank you.”

## 10. Viva questions and answers

1. **What is fake-news detection?** A text-classification task that predicts a label for a news item based on examples.
2. **What is NLP?** Natural Language Processing helps computers work with human language.
3. **What is supervised learning?** Learning from examples that have known labels.
4. **What is a dataset?** A collection of examples used to train and evaluate a model.
5. **What is TF-IDF?** A method that represents words by considering how often they appear in a document and how common they are across documents.
6. **Why use TF-IDF?** It converts text into numerical features that a machine learning model can process.
7. **What is Logistic Regression?** A supervised classification algorithm that estimates class probabilities from input features.
8. **Why split data into training and testing sets?** To evaluate the model on examples it did not train on.
9. **What does an 80:20 split mean?** 80% of usable data is used for training and 20% for testing.
10. **What is accuracy?** Correct predictions divided by all predictions.
11. **What is precision?** Correct positive predictions divided by all predictions made as positive for that class.
12. **What is recall?** Correct positive predictions divided by all actual positive examples for that class.
13. **What is F1-score?** The harmonic mean of precision and recall.
14. **What is a confusion matrix?** A table comparing actual classes with predicted classes.
15. **Does a FAKE prediction prove the story is false?** No. It only shows that the model assigned that label based on learned text patterns; claims need independent verification.

## 11. Limitations and future work

- The dataset may be older, English-only, or biased toward particular topics and publishers.
- The model may learn dataset-specific wording instead of general truthfulness.
- New events and different writing styles may reduce performance.
- Confidence estimates are model probabilities, not guarantees.
- Future improvements could include testing on newer independent datasets, adding source credibility signals, and integrating trusted fact-checking sources with proper evaluation.
