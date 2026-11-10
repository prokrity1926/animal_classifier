"""Streamlit web app:  streamlit run app.py"""
import streamlit as st
from PIL import Image, UnidentifiedImageError

import config
from predictor import AnimalPredictor, NONE_LABEL

st.set_page_config(page_title="Animal Image Classifier", page_icon="🐾")
st.title("🐾 Animal Image Classifier")
st.write("Upload an image to check whether it contains a **Cat**, **Dog** or **Cow**.")


@st.cache_resource
def get_predictor():
    return AnimalPredictor()


if not config.MODEL_PATH.exists():
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

threshold = st.sidebar.slider("Confidence threshold", 0.30, 0.95, config.CONFIDENCE_THRESHOLD, 0.05)
file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if file:
    try:
        img = Image.open(file)
        img.verify()
        file.seek(0)
        img = Image.open(file)
    except (UnidentifiedImageError, OSError):
        st.error("This file is not a valid image.")
        st.stop()

    st.image(img, caption="Uploaded image", use_container_width=True)
    predictor = get_predictor()
    predictor.threshold = threshold

    with st.spinner("Analyzing..."):
        label, conf, probs = predictor.predict(img)

    if label == NONE_LABEL:
        st.warning(f"Result: **{label}**")
    else:
        st.success(f"Result: **{label}**  ({conf:.1%} confidence)")

    st.subheader("Class probabilities")
    for name, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        st.write(name.capitalize())
        st.progress(min(max(p, 0.0), 1.0))
