import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
import re
import string

# ─────────────────────────────────────────
# 🔧 Page Config
st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide")

# ─────────────────────────────────────────
# 📦 Load Data and Model
@st.cache_data
def load_data():
    return pd.read_csv("150clusterbetter.csv")

df = load_data()

# ─────────────────────────────────────────
# 🔤 Text Cleaner
stop_words = set(TfidfVectorizer(stop_words='english').get_stop_words())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b[xX]+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join([word for word in text.split() if word not in stop_words])

@st.cache_resource
def train_model():
    vectorizer = TfidfVectorizer()
    cleaned_text = df["Consumer complaint narrative"].fillna("").apply(clean_text)
    X = vectorizer.fit_transform(cleaned_text)
    y = df["New Issue Tag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, vectorizer

model, vectorizer = train_model()

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
    "Complaint Visualizations",
    "Word Cloud"
])

# ─────────────────────────────────────────
# 🧠 Prediction
if option == "Predict Category":
    st.subheader("Enter a Complaint Narrative")
    text_input = st.text_area("Paste complaint here:")

    if st.button("🔍 Predict Category"):
        if text_input.strip():
            cleaned_input = clean_text(text_input)
            X_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(X_input)
            st.success(f"Predicted Category: {prediction[0]}")
        else:
            st.warning("Please enter a narrative.")

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
        fig = px.sunburst(tag_counts, path=["Tag"], values="Count", title="Sunburst View", height=800, width=800)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────
# ☁️ Word Cloud View
elif option == "Word Cloud":
    st.subheader("Word Cloud by Category")
    tag_choice = st.selectbox("Select a Tag", df["New Issue Tag"].unique())
    text_data = " ".join(df[df["New Issue Tag"] == tag_choice]["Consumer complaint narrative"].dropna().apply(clean_text))
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
