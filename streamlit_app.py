from pathlib import Path

# Let's prepare the content for the main Streamlit app with pretrained model
main_code = """
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from PIL import Image

# ───────────────────────────────
# 🔧 Page Config
st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide")

# ───────────────────────────────
# 📦 Load Data
@st.cache_data
def load_data():
    return pd.read_csv("150clusterbetter.csv")

df = load_data()

# 📊 Preprocess for visualization
tag_counts = df["New Issue Tag"].value_counts().reset_index()
tag_counts.columns = ["Tag", "Count"]

# ───────────────────────────────
# 🧠 Load Pretrained Model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ───────────────────────────────
# 🔝 Header
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("cfpb_logo.png", width=100)
with col2:
    st.markdown("<h2 style='text-align: center;'>Consumer Complaint Categorization</h2>", unsafe_allow_html=True)
with col3:
    st.empty()
st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────
# 🎛️ Sidebar Options
option = st.sidebar.radio("Choose View", ["Predict Category", "View Visualizations"])

# ───────────────────────────────
# ✍️ Input Section
if option == "Predict Category":
    st.subheader("Enter a Complaint Narrative")
    user_input = st.text_area("Type or paste a consumer complaint:")

    if st.button("🔍 Predict Category"):
        if user_input.strip():
            matched = df[df["Consumer complaint narrative"].str.strip().str.lower() == user_input.strip().lower()]
            if not matched.empty:
                st.success("✅ Category Predicted (Exact Match):")
                st.markdown(f"**Tag**: `{matched.iloc[0]['New Issue Tag']}`")
            else:
                X_input = vectorizer.transform([user_input])
                predicted_tag = model.predict(X_input)[0]
                st.success("✅ Category Predicted (Model):")
                st.markdown(f"**Tag**: `{predicted_tag}`")
        else:
            st.info("Please enter a narrative.")

# ───────────────────────────────
# 📊 Visualization Section
elif option == "View Visualizations":
    st.subheader("Complaint Category Visualization")
    viz_type = st.radio("Choose Visualization Style:", ["Treemap", "Bar (Horizontal)", "Bubble Chart"])

    if viz_type == "Treemap":
        fig = px.treemap(tag_counts, path=['Tag'], values='Count', title="Treemap of Complaint Categories")
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
        fig = px.bar(sorted_tags, x="Count", y="Tag", orientation='h',
                     title="Bar Chart of Complaint Categories", color="Color",
                     color_discrete_map={"green": "green", "orange": "orange", "red": "red"}, height=800)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Bubble Chart":
        fig = px.scatter(tag_counts, x='Tag', y='Count',
                         size='Count', color='Tag', size_max=60,
                         title='Bubble Chart of Complaint Categories')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────
# 🔚 Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Powered by CFPB Open Consumer Complaint Data</h4>", unsafe_allow_html=True)
"""

file_path = Path("/mnt/data/streamlit_app.py")
file_path.write_text(main_code)
file_path.name
