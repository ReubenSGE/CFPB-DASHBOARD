from pathlib import Path

# Rewriting the file again after kernel reset
github_app_code = '''
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# 🔧 Page Config
st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide")

# ─────────────────────────────────────────
# 📦 Load Data and Models
@st.cache_data
def load_data():
    return pd.read_csv("150clusterbetter.csv")

@st.cache_resource
def load_models():
    with open("sentence_transformer_10000.pkl", "rb") as f:
        transformer = pickle.load(f)
    with open("umap_model_10000.pkl", "rb") as f:
        umap_model = pickle.load(f)
    with open("cluster_umap_10000.pkl", "rb") as f:
        cluster_data = pickle.load(f)
    return transformer, umap_model, cluster_data

df = load_data()
transformer, umap_model, cluster_data = load_models()

# ─────────────────────────────────────────
# 📊 Preprocess
tag_counts = df["New Issue Tag"].value_counts().reset_index()
tag_counts.columns = ["Tag", "Count"]

# ─────────────────────────────────────────
# 🔝 Header
st.markdown(
    "<h2 style='text-align: center;'>Consumer Complaint Categorization</h2><hr>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────
# 🎛️ Sidebar Navigation
option = st.sidebar.radio("Choose View", [
    "Predict Category",
    "UMAP Cluster View",
    "Complaint Visualizations",
    "Word Cloud"
])

# ─────────────────────────────────────────
# 🧠 Prediction
if option == "Predict Category":
    st.subheader("Enter a Complaint Narrative")
    text_input = st.text_area("Paste complaint here:")

    if st.button("🔍 Predict Cluster"):
        if text_input.strip():
            embed = transformer.encode([text_input])
            reduced = umap_model.transform(embed)
            kmeans = KMeans(n_clusters=len(set(cluster_data['Cluster'])))
            kmeans.fit(cluster_data[['X', 'Y']])
            pred_label = kmeans.predict(reduced)
            st.success(f"Predicted Cluster Label: {pred_label[0]}")
        else:
            st.warning("Please enter a narrative.")

# ─────────────────────────────────────────
# 📍 UMAP Visualization
elif option == "UMAP Cluster View":
    st.subheader("Complaint Clusters (UMAP Projection)")
    fig = px.scatter(cluster_data, x="X", y="Y", color="Cluster", hover_data=["Tag"],
                     title="UMAP 2D Visualization of Complaint Clusters")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# 📊 Complaint Visualizations
elif option == "Complaint Visualizations":
    st.subheader("Complaint Category Visualization")
    viz_type = st.radio("Choose Visualization Style:", ["Treemap", "Bar Chart", "Sunburst"])

    if viz_type == "Treemap":
        fig = px.treemap(tag_counts, path=["Tag"], values="Count", title="Treemap of Tags")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Bar Chart":
        sorted_tags = tag_counts.sort_values("Count", ascending=True).reset_index(drop=True)
        top_n, bottom_n, total = 5, 5, len(sorted_tags)
        colors = ["red" if i < bottom_n else "green" if i >= total - top_n else "orange" for i in range(total)]
        sorted_tags["Color"] = colors
        fig = px.bar(sorted_tags, x="Count", y="Tag", orientation="h",
                     title="Tag Distribution", color="Color",
                     color_discrete_map={"green": "green", "orange": "orange", "red": "red"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Sunburst":
        fig = px.sunburst(tag_counts, path=["Tag"], values="Count", title="Sunburst View")
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# ☁️ Word Cloud View
elif option == "Word Cloud":
    st.subheader("Word Cloud by Category")
    tag_choice = st.selectbox("Select a Tag", df["New Issue Tag"].unique())
    text_data = " ".join(df[df["New Issue Tag"] == tag_choice]["Consumer complaint narrative"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text_data)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# ─────────────────────────────────────────
# 🔚 Footer
st.markdown("<hr>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("cfpb_logo.png", width=60)
with col2:
    st.markdown("<h4 style='text-align: center;'>Powered by CFPB Open Consumer Complaint Data</h4>", unsafe_allow_html=True)
with col3:
    st.empty()
'''

file_path = Path("/mnt/data/streamlit_app_github_version.py")
file_path.write_text(github_app_code)
file_path.name
