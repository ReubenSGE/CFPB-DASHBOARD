import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import accuracy_score, classification_report

# ───────────────────────────────
# 🔧 Page Config
st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide")

# ───────────────────────────────
# 📦 Load Data (for visualizations)
@st.cache_data
def load_data():
    return pd.read_csv("150clusterbetter.csv").sample(n=1000, random_state=42)

df = load_data()

# ───────────────────────────────
# 📦 Load Pretrained Components
@st.cache_resource
def load_pretrained_model():
    model = joblib.load("logreg_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, encoder = load_pretrained_model()

# 📊 Tag Distribution
tag_counts = df["New Issue Tag"].value_counts().reset_index()
tag_counts.columns = ["Tag", "Count"]

# ───────────────────────────────
# 🔝 Header
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Consumer_Financial_Protection_Bureau_logo.svg/320px-Consumer_Financial_Protection_Bureau_logo.svg.png' width='100'/>
        <h2 style='margin-top: 0;'>Consumer Complaint Categorization</h2>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────
# Sidebar
option = st.sidebar.radio("Choose View", ["Predict Category", "View Visualizations"])

# ───────────────────────────────
# 🔍 Predict Category
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
                predicted_label = model.predict(X_input)[0]
                predicted_tag = encoder.inverse_transform([predicted_label])[0]
                st.success("✅ Category Predicted (Pretrained Model):")
                st.markdown(f"**Tag**: `{predicted_tag}`")
        else:
            st.info("Please enter a narrative.")

# ───────────────────────────────
# 📊 Visualizations
elif option == "View Visualizations":
    st.subheader("Complaint Category Visualization")

    viz_type = st.radio("Choose Visualization Style:", ["Treemap", "Bar (Horizontal)", "Bubble Chart"])

    if viz_type == "Treemap":
        fig = px.treemap(tag_counts, path=["Tag"], values="Count", title="Treemap of Complaint Categories")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Bar (Horizontal)":
        sorted_tags = tag_counts.sort_values("Count", ascending=True).reset_index(drop=True)
        top_n = 5
        bottom_n = 5
        total = len(sorted_tags)
        colors = [
            "red" if i < bottom_n else "green" if i >= total - top_n else "orange"
            for i in range(total)
        ]
        sorted_tags["Color"] = colors

        fig = px.bar(
            sorted_tags, x="Count", y="Tag",
            orientation="h", title="Bar Chart of Complaint Categories",
            color="Color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            height=800
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Bubble Chart":
        fig = px.scatter(tag_counts, x="Tag", y="Count",
                         size="Count", color="Tag", size_max=60,
                         title="Bubble Chart of Complaint Categories")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────
# 🔚 Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center;'>
        <h4>Powered by CFPB Open Consumer Complaint Data</h4>
    </div>
    """,
    unsafe_allow_html=True
)
