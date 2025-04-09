import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# ───────────────────────────────
# Page Config
st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide", initial_sidebar_state="expanded")

# ───────────────────────────────
# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("150clusterbetter.csv")

df = load_data()

# ───────────────────────────────
# Load Model & Encoder
@st.cache_resource
def train_model():
    data = df.dropna(subset=["Consumer complaint narrative", "New Issue Tag"])
    X = data["Consumer complaint narrative"]
    y = data["New Issue Tag"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X_embed = embedder.encode(X.tolist(), show_progress_bar=True)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_embed, y_encoded)

    return model, embedder, encoder

model, embedder, encoder = train_model()

# ───────────────────────────────
# Sidebar Navigation
mode = st.sidebar.radio("Select Mode", ["Predict Category", "Visualize Categories"])

# ───────────────────────────────
# Header with CFPB logo
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("cfpb_logo.png", width=100)
with col2:
    st.markdown("## Consumer Complaint Categorization", unsafe_allow_html=True)
with col3:
    st.empty()
st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────
# Predict Category
if mode == "Predict Category":
    st.subheader("Enter a Complaint Narrative")
    user_input = st.text_area("Type or paste a consumer complaint:")

    if st.button("🔍 Predict Category"):
        if user_input.strip():
            matched = df[df["Consumer complaint narrative"].str.strip().str.lower() == user_input.strip().lower()]
            if not matched.empty:
                st.success("✅ Category Predicted (Exact Match):")
                st.markdown(f"**Tag**: `{matched.iloc[0]['New Issue Tag']}`")
            else:
                vector = embedder.encode([user_input])
                prediction = model.predict(vector)[0]
                tag = encoder.inverse_transform([prediction])[0]
                st.success("✅ Category Predicted (Model):")
                st.markdown(f"**Tag**: `{tag}`")
        else:
            st.info("Please enter a narrative.")

# ───────────────────────────────
# Visualize Categories
elif mode == "Visualize Categories":
    st.subheader("Complaint Category Visualization")

    tag_counts = df["New Issue Tag"].value_counts().reset_index()
    tag_counts.columns = ["Tag", "Count"]

    viz_type = st.radio("Choose Visualization Style", ["Treemap", "Bar (Horizontal)", "Bubble Chart"])

    if viz_type == "Treemap":
        fig = px.treemap(tag_counts, path=["Tag"], values="Count", title="Treemap of Complaint Categories")
        st.plotly_chart(fig, use_container_width=True)

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
            title="Bar Chart of Complaint Categories",
            color="Color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            height=800
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Bubble Chart":
        fig = px.scatter(tag_counts, x='Tag', y='Count',
                         size='Count', color='Tag', size_max=60,
                         title='Bubble Chart of Complaint Categories')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────
# Footer
st.markdown("<hr>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.empty()
with col2:
    st.markdown("### Powered by CFPB Open Consumer Complaint Data", unsafe_allow_html=True)
with col3:
    st.image("cfpb_logo.png", width=100)
