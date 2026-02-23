import streamlit as st
from transformers import pipeline
import re

st.set_page_config(
    page_title="Sports Commentary Sentiment Analyzer",
    page_icon="🏆",
    layout="centered"
)

# ---- Custom Styling ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%);
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1 {
    background: linear-gradient(135deg, #f59e0b, #ef4444, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.5px;
    text-align: center;
}

.stSelectbox > div > div,
.stTextArea textarea,
.stRadio > div {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.3s ease;
}

.stSelectbox > div > div:hover,
.stTextArea textarea:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2.5rem !important;
    width: 100%;
    transition: all 0.3s ease !important;
    letter-spacing: 0.3px;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(245, 158, 11, 0.35) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

.result-card {
    background: linear-gradient(145deg, #1e293b, #162032);
    border: 1px solid #334155;
    padding: 24px 28px;
    border-radius: 16px;
    margin-top: 16px;
    margin-bottom: 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

.result-card .commentary-text {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin-bottom: 14px;
    line-height: 1.6;
    border-left: 3px solid #f59e0b;
    padding-left: 14px;
}

.result-card .sentiment-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.result-card .confidence-text {
    font-size: 0.95rem;
    color: #94a3b8;
    font-weight: 500;
}

.result-card .interpretation-text {
    font-size: 0.9rem;
    color: #64748b;
    font-style: italic;
    margin-top: 6px;
}

.stProgress > div > div {
    border-radius: 10px !important;
    height: 8px !important;
}

div[data-testid="stCaption"] {
    text-align: center;
    color: #475569 !important;
    font-size: 0.8rem !important;
}

hr {
    border-color: #1e293b !important;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-top: -10px;
    margin-bottom: 30px;
}

.section-label {
    color: #f59e0b;
    font-weight: 600;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}

.sport-badge {
    display: inline-block;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #f59e0b;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏆 Sports Commentary Sentiment Analyzer")
st.markdown('<p class="subtitle">Analyze live sports commentary using Transformer-based NLP models</p>', unsafe_allow_html=True)

# ---- Sport Selector ----
st.markdown('<p class="section-label">🎯 Sport Type</p>', unsafe_allow_html=True)
sport = st.selectbox(
    "Select Sport Type:",
    ["Cricket 🏏", "Football ⚽", "Tennis 🎾", "Basketball 🏀"],
    label_visibility="collapsed"
)

# ---- Load Model ----
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# ---- Input Mode ----
st.markdown('<p class="section-label">📝 Input Mode</p>', unsafe_allow_html=True)
mode = st.radio("Choose Input Mode:", ["Single Commentary", "Multiple Lines"], label_visibility="collapsed")

st.markdown('<p class="section-label">💬 Commentary</p>', unsafe_allow_html=True)
if mode == "Single Commentary":
    commentary = st.text_area("Enter Sports Commentary Text:", placeholder="e.g. What a brilliant goal by Messi!", label_visibility="collapsed")
    comments = [commentary]
else:
    commentary = st.text_area("Enter multiple commentaries (one per line):", placeholder="Enter one commentary per line...", label_visibility="collapsed")
    comments = commentary.split("\n")

st.markdown("<br>", unsafe_allow_html=True)

# ---- Analyze ----
if st.button("⚡ Analyze Sentiment"):

    if commentary.strip() == "":
        st.warning("⚠️ Please enter commentary text.")
    else:
        st.markdown(f'<span class="sport-badge">{sport}</span>', unsafe_allow_html=True)

        for text in comments:
            if text.strip() == "":
                continue

            with st.spinner("Analyzing..."):
                result = classifier(text)[0]

            label = result["label"]
            confidence = result["score"]

            # Sports-specific keyword boost
            positive_words = ["goal", "six", "win", "victory", "ace", "slam", "brilliant", "amazing"]
            negative_words = ["miss", "out", "foul", "injury", "loss", "mistake", "poor"]

            score_adjust = 0
            for word in positive_words:
                if re.search(rf"\b{word}\b", text.lower()):
                    score_adjust += 0.05
            for word in negative_words:
                if re.search(rf"\b{word}\b", text.lower()):
                    score_adjust -= 0.05

            confidence = min(max(confidence + score_adjust, 0), 1)

            # Final Sentiment
            if confidence > 0.75:
                if label == "POSITIVE":
                    sentiment = "🔥 Strong Positive"
                    color = "#22c55e"
                    border_color = "#22c55e"
                else:
                    sentiment = "💔 Strong Negative"
                    color = "#ef4444"
                    border_color = "#ef4444"
            else:
                sentiment = "😐 Neutral"
                color = "#f59e0b"
                border_color = "#f59e0b"

            # Sentiment Interpretation
            if sentiment == "🔥 Strong Positive":
                interpretation = "Excited / Dominating Moment"
            elif sentiment == "💔 Strong Negative":
                interpretation = "Frustrated / Disappointing Moment"
            else:
                interpretation = "Balanced / Informative Commentary"

            # Display
            st.markdown(
                f"""
                <div class="result-card" style="border-left: 4px solid {border_color};">
                    <div class="commentary-text">"{text}"</div>
                    <div class="sentiment-label" style="color: {color};">{sentiment}</div>
                    <div class="confidence-text">Confidence: <strong>{confidence:.2%}</strong></div>
                    <div class="interpretation-text">💡 {interpretation}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Confidence Meter
            st.progress(confidence)

st.markdown("---")
