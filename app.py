"""Streamlit interface for the fake-news text classifier."""

from pathlib import Path
import json
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_news_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
MATRIX_PATH = ROOT / "reports" / "confusion_matrix.png"
REPORT_PATH = ROOT / "reports" / "classification_report.txt"

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection Using Machine Learning and NLP")
st.caption("B.Tech Mini Project | TF-IDF + Logistic Regression")
st.write(
    "Enter a news headline or article. The model predicts whether the text resembles "
    "FAKE or REAL examples in its training dataset."
)
st.warning(
    "Educational demonstration only: a prediction is not a fact-check. The model can be wrong, "
    "especially for recent events, unfamiliar topics, or text unlike its training data."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = None
if MODEL_PATH.exists():
    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Could not load the saved model: {exc}")
        st.info("Run `python train_model.py` again from the project folder.")
else:
    st.info(
        "The model has not been trained yet. Add the dataset, then run "
        "`python train_model.py` in the VS Code terminal."
    )

news_text = st.text_area(
    "Enter a news headline or article",
    height=180,
    placeholder="Paste the news headline or article text here..."
)

if st.button("Predict News Label", type="primary", use_container_width=True):
    if not news_text.strip():
        st.error("Please enter a headline or article before predicting.")
    elif model is None:
        st.error("The trained model is unavailable. Follow the training instructions above.")
    else:
        prediction = str(model.predict([news_text.strip()])[0]).upper()
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([news_text.strip()])[0]
            classes = [str(item).upper() for item in model.classes_]
            score = float(probabilities[classes.index(prediction)])
        else:
            score = None

        if prediction == "FAKE":
            st.error("Prediction: FAKE")
        else:
            st.success("Prediction: REAL")

        if score is not None:
            st.metric("Model-estimated confidence", f"{score * 100:.1f}%")
            st.caption("This is the model's estimated probability for its chosen class, not a guarantee of truth.")

        st.write("**What this means:** The text has been classified using patterns learned from the training examples.")
        st.write("**Please verify important claims** using multiple reliable sources and independent fact-checking.")

st.divider()
st.header("How the model works")
st.markdown(
    """
    1. **Dataset:** labelled examples of fake and real news are loaded.
    2. **TF-IDF:** converts words and word pairs into numerical features.
    3. **Logistic Regression:** learns patterns associated with each label.
    4. **Prediction:** the trained pipeline predicts a label for the text entered above.
    """
)

st.header("Model evaluation")
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        c1, c2 = st.columns(2)
        c1.metric("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        c2.metric("Macro F1-score", f"{metrics['f1_macro'] * 100:.2f}%")
        c3, c4 = st.columns(2)
        c3.metric("Macro precision", f"{metrics['precision_macro'] * 100:.2f}%")
        c4.metric("Macro recall", f"{metrics['recall_macro'] * 100:.2f}%")
        st.caption(
            f"Evaluation used {metrics['testing_rows']:,} held-out test rows "
            f"from {metrics['total_rows']:,} cleaned dataset rows. These are the results "
            "from your actual training run."
        )
    except Exception as exc:
        st.info(f"Metrics could not be read: {exc}")
else:
    st.info("Evaluation metrics will appear here after you run train_model.py.")

if MATRIX_PATH.exists():
    st.subheader("Confusion matrix")
    st.image(str(MATRIX_PATH), caption="Actual labels compared with predicted labels", use_container_width=True)

if REPORT_PATH.exists():
    with st.expander("View full classification report"):
        st.text(REPORT_PATH.read_text(encoding="utf-8"))

with st.expander("About this project"):
    st.write(
        "This project demonstrates supervised text classification. It does not connect to live news "
        "sources and does not independently check claims against evidence. Dataset labels, source "
        "bias, repeated wording, and changes over time can affect results."
    )
