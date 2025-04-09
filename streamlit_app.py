import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Page config
st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide", initial_sidebar_state="expanded")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("150clusterbetter.csv")

df = load_data()

# Sidebar Mode Switcher
mode = st.sidebar.radio("Select Mode", ["🔍 Predict Complaint Category", "📊 View Complaint Category Distribution"])

# Load model resources
@st.cache_resource
def load_model_and_encoder():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["New Issue Tag"])
    X = model.encode(df["Consumer complaint narrative"].astype(str).tolist())
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return model, clf, label_encoder

embedder, classifier, encoder = load_model_and_encoder()

# Header
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("cfpb_logo.png", width=100)
with col2:
    st.markdown("<h2 style='text-align:center;'>Consumer Complaint Categorization</h2>", unsafe_allow_html=True)
with col3:
    st.empty()
st.markdown("<hr>", unsafe_allow_html=True)

# Mode 1: Prediction
if mode == "🔍 Predict Complaint Category":
    st.subheader("Enter a Complaint Narrative")
    user_input = st.text_area("Type or paste a consumer complaint:")

    if st.button("🔍 Predict Category"):
        if user_input.strip():
            # Exact match first
            matched = df[df["Consumer complaint narrative"].str.strip().str.lower() == user_input.strip().lower()]
            if not matched.empty:
                st.success("✅ Category Predicted (Exact Match):")
                st.markdown(f"**Tag**: `{matched.iloc[0]['New Issue Tag']}`")
            else:
                # Semantic prediction
                X_pred = embedder.encode([user_input])
                pred = classifier.predict(X_pred)
                category = encoder.inverse_transform(pred)[0]
                st.success("✅ Category Predicted (Model):")
                st.markdown(f"**Tag**: `{category}`")
        else:
            st.info("Please enter a narrative.")

# Mode 2: Visualization
elif mode == "📊 View Complaint Category Distribution":
    st.subheader("Complaint Category Visualization")
    
    tag_counts = df["New Issue Tag"].value_counts().reset_index()
    tag_counts.columns = ["Tag", "Count"]

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
            title="Bar Chart of Complaint Categories",
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

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.empty()
with col2:
    st.markdown("### Powered by CFPB Open Consumer Complaint Data", unsafe_allow_html=True)
with col3:
    st.image("cfpb_logo.png", width=100)
