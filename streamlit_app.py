import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from PIL import Image

st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("150clusterbetter.csv")

df = load_data()

@st.cache_resource
def train_model():
    data = df.dropna(subset=["Consumer complaint narrative", "New Issue Tag"])
    X = data["Consumer complaint narrative"]
    y = data["New Issue Tag"]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=1500, class_weight="balanced")
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = train_model()

tag_counts = df["New Issue Tag"].value_counts().reset_index()
tag_counts.columns = ["Tag", "Count"]

# ───────────────────────────────
# 🔝 Header
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("cfpb_logo.png", width=100)
with col2:
    st.markdown("<h2 style='text-align:center;'>Consumer Complaint Categorization</h2>", unsafe_allow_html=True)
with col3:
    st.empty()
st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────
# 📍 Sidebar Input
st.sidebar.header("📥 Enter a Complaint")
user_input = st.sidebar.text_area("Type or paste a complaint:")

if st.sidebar.button("🔍 Predict Category"):
    if user_input.strip():
        matched = df[df["Consumer complaint narrative"].str.strip().str.lower() == user_input.strip().lower()]
        if not matched.empty:
            st.sidebar.success("✅ Exact Match Prediction")
            st.sidebar.markdown(f"**Tag:** `{matched.iloc[0]['New Issue Tag']}`")
        else:
            pred_vec = vectorizer.transform([user_input])
            predicted_tag = model.predict(pred_vec)[0]
            st.sidebar.success("✅ Model Prediction")
            st.sidebar.markdown(f"**Predicted Tag:** `{predicted_tag}`")
    else:
        st.sidebar.info("Enter a complaint narrative.")

# ───────────────────────────────
# 📊 Visualization
st.markdown("### Complaint Category Visualization")
viz_type = st.radio("Choose a Visualization", ["Treemap", "Bar (Horizontal)", "Bubble Chart"])

if viz_type == "Treemap":
    fig = px.treemap(tag_counts, path=['Tag'], values='Count', title="Treemap of Complaint Categories")
elif viz_type == "Bar (Horizontal)":
    sorted_tags = tag_counts.sort_values("Count", ascending=True).reset_index(drop=True)
    top_n = 5
    bottom_n = 5
    total = len(sorted_tags)

    colors = []
    for i in range(total):
        if i < bottom_n:
            colors.append("red")
        elif i >= total - top_n:
            colors.append("green")
        else:
            colors.append("orange")

    sorted_tags["Color"] = colors
    fig = px.bar(
        sorted_tags,
        x="Count", y="Tag",
        orientation='h',
        color="Color",
        color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
        height=800
    )
    fig.update_layout(showlegend=False)
elif viz_type == "Bubble Chart":
    fig = px.scatter(tag_counts, x='Tag', y='Count',
                     size='Count', color='Tag', size_max=60,
                     title='Bubble Chart of Complaint Categories')
    fig.update_layout(showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────
# 🔚 Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Powered by CFPB Open Consumer Complaint Data</div>", unsafe_allow_html=True)
