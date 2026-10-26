"""Train and evaluate a fake-news text classifier."""

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"


def normalize_label(value):
    """Convert common label formats to FAKE or REAL."""
    value = str(value).strip().lower()
    fake_values = {"fake", "false", "0", "f", "unreliable"}
    real_values = {"real", "true", "1", "r", "reliable", "authentic"}
    if value in fake_values:
        return "FAKE"
    if value in real_values:
        return "REAL"
    return None


def find_column(df, names):
    """Find a column without depending on capitalization."""
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        if name in lookup:
            return lookup[name]
    return None


def load_dataset():
    """Load either Fake.csv + True.csv, or a single labelled CSV."""
    fake_path = DATA_DIR / "Fake.csv"
    true_path = DATA_DIR / "True.csv"
    fake_lower = DATA_DIR / "fake.csv"
    true_lower = DATA_DIR / "true.csv"

    if fake_path.exists() and true_path.exists():
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(true_path)
        fake_df["label"] = "FAKE"
        real_df["label"] = "REAL"
        df = pd.concat([fake_df, real_df], ignore_index=True)
        print("Loaded dataset/Fake.csv and dataset/True.csv")
    elif fake_lower.exists() and true_lower.exists():
        fake_df = pd.read_csv(fake_lower)
        real_df = pd.read_csv(true_lower)
        fake_df["label"] = "FAKE"
        real_df["label"] = "REAL"
        df = pd.concat([fake_df, real_df], ignore_index=True)
        print("Loaded dataset/fake.csv and dataset/true.csv")
    else:
        single_path = DATA_DIR / "news.csv"
        if not single_path.exists():
            raise FileNotFoundError(
                "\nDataset not found.\n"
                "Place either Fake.csv and True.csv inside the dataset folder,\n"
                "or place one labelled CSV named news.csv there.\n"
                "See README.md for download instructions."
            )
        df = pd.read_csv(single_path)
        print("Loaded dataset/news.csv")

    text_col = find_column(df, ["text", "article", "content", "news", "statement"])
    title_col = find_column(df, ["title", "headline"])
    label_col = find_column(df, ["label", "class", "target", "fake", "status"])

    if label_col is None:
        raise ValueError(
            "Could not find a label column in news.csv. Expected a column such as "
            "'label' or 'class'. If using Fake.csv and True.csv, keep both files."
        )
    if text_col is None and title_col is None:
        raise ValueError("No text column found. Expected 'text', 'article', 'content', or 'title'.")

    if text_col is not None:
        body = df[text_col].fillna("").astype(str).str.strip()
    else:
        body = pd.Series([""] * len(df), index=df.index)

    if title_col is not None and title_col != text_col:
        title = df[title_col].fillna("").astype(str).str.strip()
        combined_text = (title + " " + body).str.strip()
    else:
        combined_text = body.str.strip()

    labels = df[label_col].map(normalize_label)
    clean = pd.DataFrame({"text": combined_text, "label": labels})
    clean = clean.dropna(subset=["label"])
    clean = clean[clean["text"].str.len() > 0]
    clean["text_key"] = clean["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    clean = clean.drop_duplicates(subset=["text_key"]).drop(columns=["text_key"])

    if clean.empty:
        raise ValueError("No usable labelled text remains after cleaning.")
    if clean["label"].nunique() < 2:
        raise ValueError(
            "The dataset must contain both FAKE and REAL labels. "
            "For a single CSV, use labels such as Fake/Real or 0/1 (0=Fake, 1=Real)."
        )
    counts = clean["label"].value_counts()
    if counts.min() < 2:
        raise ValueError("Each class needs at least two rows to create a train/test split.")

    print(f"Usable articles after cleaning: {len(clean):,}")
    print("Class counts:")
    print(clean["label"].value_counts().to_string())
    return clean


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        data = load_dataset()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=100000,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    print("\nTraining TF-IDF + Logistic Regression model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    labels = ["FAKE", "REAL"]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, labels=labels, average="macro", zero_division=0)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "total_rows": int(len(data)),
        "class_counts": {str(k): int(v) for k, v in data["label"].value_counts().items()},
    }

    report = classification_report(y_test, predictions, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    joblib.dump(model, MODEL_PATH)
    (REPORT_DIR / "classification_report.txt").write_text(
        "Evaluation on the held-out test set\n\n" + report, encoding="utf-8"
    )
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.title("Fake News Detection - Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\nMODEL EVALUATION (held-out test set)")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro precision:{metrics['precision_macro']:.4f}")
    print(f"Macro recall:   {metrics['recall_macro']:.4f}")
    print(f"Macro F1-score: {metrics['f1_macro']:.4f}")
    print("\nClassification report:\n")
    print(report)
    print(f"Saved model: {MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved reports: {REPORT_DIR.relative_to(ROOT)}")
    print("\nTraining complete. Now run: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
