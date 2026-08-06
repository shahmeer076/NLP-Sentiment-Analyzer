import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==================================================
# Configuration
# ==================================================

MODEL_NAME = "textattack/bert-base-uncased-imdb"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================
# Load Model
# ==================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# ==================================================
# Streamlit UI
# ==================================================

st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Sentiment Analysis")

st.write(
    """
Analyze a movie review using a **BERT** model and predict
whether the sentiment is **Positive** or **Negative**.
"""
)

st.divider()

review = st.text_area(
    "Enter Movie Review",
    placeholder="Example: This movie was absolutely amazing. I loved every minute!",
    height=180,
)

# ==================================================
# Prediction
# ==================================================

if st.button("Analyze Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("⚠ Please enter a movie review.")
        st.stop()

    with st.spinner("Analyzing Review..."):

        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

    confidence = confidence.item()
    prediction = prediction.item()

    st.divider()

    st.subheader("Prediction")

    if prediction == 1:
        st.success("😊 Positive Review")
    else:
        st.error("😞 Negative Review")

    st.write(f"**Confidence : {confidence * 100:.2f}%**")

    st.progress(float(confidence))

st.divider()

st.caption("Built with ❤️ using Streamlit + Hugging Face Transformers")