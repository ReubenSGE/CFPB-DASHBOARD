import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ───────────────────────────────
# 🔧 Page Config
st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide")

# ───────────────────────────────
# 📦 Load Data
@st.cache_data
def load_data():
    return pd.read_csv("150clusterbetter.csv")

df = load_data()

# ───────────────────────────────
# 🧠 Train BERT-based Classifier
@st.cache_resource
def train_model():
    clean_df = df.dropna(subset=["Consumer complaint narrative", "New Issue Tag"])
    X_text = clean_df["Consumer complaint narrative"].tolist()
    y = clean_df["New Issue Tag"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    X_embed = bert_model.encode(X_text, show_progress_bar=True)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_embed, y_encoded)

    return classifier, bert_model, label_encoder

model, embedder, encoder = train_model()

# 📊 Preprocess for visualization
tag_counts = df["New Issue Tag"].value_counts().reset_index()
tag_counts.columns = ["Tag", "Count"]

# ───────────────────────────────
# 🔝 Centered Header
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
# 🎛️ Sidebar
option = st.sidebar.radio("Choose View", ["Predict Category", "View Visualizations"])

# ───────────────────────────────
# 🔍 Predict Mode
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
                input_vec = embedder.encode([user_input])
                prediction = model.predict(input_vec)[0]
                predicted_tag = encoder.inverse_transform([prediction])[0]
                st.success("✅ Category Predicted (BERT Model):")
                st.markdown(f"**Tag**: `{predicted_tag}`")
        else:
            st.info("Please enter a narrative.")

# ───────────────────────────────
# 📊 Visualization Mode
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
# 🔚 Centered Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center;'>
        <h4>Powered by CFPB Open Consumer Complaint Data</h4>
    </div>
    """,
    unsafe_allow_html=True
)
