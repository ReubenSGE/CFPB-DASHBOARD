import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ─── Page Configuration ──────────────────────────────
st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide", initial_sidebar_state="expanded")

# ─── Load Data ───────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("150clusterbetter.csv")

df = load_data()

# ─── Train TF-IDF Model ──────────────────────────────
@st.cache_resource
def train_model():
    df_clean = df.dropna(subset=["Consumer complaint narrative", "New Issue Tag"])
    X = df_clean["Consumer complaint narrative"]
    y = df_clean["New Issue Tag"]

    vectorizer = TfidfVectorizer(max_features=3000)
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = train_model()

# ─── Tag Frequency for Visuals ───────────────────────
tag_counts = df["New Issue Tag"].value_counts().reset_index()
tag_counts.columns = ["Tag", "Count"]

# ─── HEADER ──────────────────────────────────────────
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("cfpb_logo.png", width=100)
with col2:
    st.markdown("## Consumer Complaint Categorization")
with col3:
    st.empty()
st.markdown("<hr>", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────
mode = st.sidebar.radio("Select Mode", ["🔍 Predict Issue", "📊 Visualize Categories"])

# ─── PREDICT MODE ────────────────────────────────────
if mode == "🔍 Predict Issue":
    st.subheader("Enter a &#8203;:contentReference[oaicite:0]{index=0}&#8203;
