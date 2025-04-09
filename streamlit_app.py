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
    st.subheader("Enter a Complaint Narrative")
    user_input = st.text_area("Type or paste a consumer complaint:")

    if st.button("Predict Category"):
        if user_input.strip():
            matched = df[df["Consumer complaint narrative"].str.strip().str.lower() == user_input.strip().lower()]
            if not matched.empty:
                st.success("✅ Exact Match Found")
                st.markdown(f"**Tag**: `{matched.iloc[0]['New Issue Tag']}`")
            else:
                input_vec = vectorizer.transform([user_input])
                prediction = model.predict(input_vec)[0]
                st.success("✅ Model Prediction")
                st.markdown(f"**Tag**: `{prediction}`")
        else:
            st.info("Please enter a narrative.")

# ─── VISUALIZATION MODE ─────────────────────────────
elif mode == "📊 Visualize Categories":
    st.subheader("Complaint Category Visualization")
    viz_type = st.radio("Choose Visualization Style:", ["Treemap", "Bar (Horizontal)", "Bubble Chart"])

    if viz_type == "Treemap":
        fig = px.treemap(tag_counts, path=["Tag"], values="Count")

    elif viz_type == "Bar (Horizontal)":
        sorted_tags = tag_counts.sort_values("Count", ascending=True).reset_index(drop=True)
        total = len(sorted_tags)
        top_n = 5
        bottom_n = 5
        colors = ["red" if i < bottom_n else "green" if i >= total - top_n else "orange" for i in range(total)]
        sorted_tags["Color"] = colors

        fig = px.bar(
            sorted_tags, x="Count", y="Tag", orientation='h',
            color="Color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            height=800
        )
        fig.update_layout(showlegend=False)

    elif viz_type == "Bubble Chart":
        fig = px.scatter(tag_counts, x='Tag', y='Count', size='Count', color='Tag', size_max=60)
        fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

# ─── FOOTER ──────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.empty()
with col2:
    st.markdown("### Powered by CFPB Open Consumer Complaint Data")
with col3:
    st.empty()
