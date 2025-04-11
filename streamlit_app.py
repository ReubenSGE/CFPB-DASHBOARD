import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ───────────────────────────────
# 🔧 Page Config
st.set_page_config(page_title="CFPB NLP Dashboard", layout="wide")

# ───────────────────────────────
# 📦 Load Data (Optionally sample)
@st.cache_data
def load_data():
    df = pd.read_csv("150clusterbetter.csv")
    return df.sample(n=3000, random_state=42)  # Faster load

df = load_data()

# ───────────────────────────────
# 🧠 Train/Test + TF-IDF Classifier
@st.cache_resource
def train_model():
    df_clean = df.dropna(subset=["Consumer complaint narrative", "New Issue Tag"])
    X = df_clean["Consumer complaint narrative"]
    y = df_clean["New Issue Tag"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    tfidf = TfidfVectorizer(max_features=3000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

    return model, tfidf, label_encoder, acc, report

model, vectorizer, encoder, acc, report = train_model()

# 📊 Tag counts
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
# Sidebar
option = st.sidebar.radio("Choose View", ["Predict Category", "View Visualizations"])

# ───────────────────────────────
# 🔍 Prediction UI
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
                prediction = model.predict(X_input)[0]
                predicted_tag = encoder.inverse_transform([prediction])[0]
                st.success("✅ Category Predicted (TF-IDF Model):")
                st.markdown(f"**Tag**: `{predicted_tag}`")
        else:
            st.info("Please enter a narrative.")

    with st.expander("📈 Model Evaluation on Test Set"):
        st.markdown(f"**Accuracy:** `{acc:.2%}`")
        st.text(report)

# ───────────────────────────────
# 📊 Visualization UI
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
            sorted_tags, x="Count", y="Tag",
            orientation='h', title="Bar Chart of Complaint Categories",
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
