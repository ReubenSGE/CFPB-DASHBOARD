import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# ───────────────────────────────
# 🔧 Page Config
st.set_page_config(page_title="TIAA CFPB NLP Dashboard", layout="wide")

# ───────────────────────────────
# 📦 Load Data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\reube\Downloads\streamlit_dashboard_project\streamlit_dashboard_project\150clusterbetter.csv")

df = load_data()

# 📊 Preprocess for visualization
tag_counts = df["New Issue Tag"].value_counts().reset_index()
tag_counts.columns = ["Tag", "Count"]

# ───────────────────────────────
# 🔝 Header Layout: TIAA Logo + Title
# ─── CENTERED HEADER ───
# ─── TOP HEADER ───
# ─── TIAA LOGO ABOVE CENTERED TITLE ───
# ─── SINGLE-LINE HEADER ───
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.image("tiaa_logo.jpeg", width=100)

with col2:
    st.markdown("## Consumer Complaint Categorization")

with col3:
    st.empty()

st.markdown("<hr>", unsafe_allow_html=True)


# ───────────────────────────────
# ✍️ Input Section
st.subheader("Enter a Complaint Narrative")
user_input = st.text_area("Type or paste a consumer complaint:")

if st.button("🔍 Predict Category"):
    if user_input.strip():
        matched = df[df["Consumer complaint narrative"].str.strip().str.lower() == user_input.strip().lower()]
        if not matched.empty:
            st.success("✅ Category Predicted:")
            st.markdown(f"**Tag**: `{matched.iloc[0]['New Issue Tag']}`")
        else:
            st.warning("❗Exact match not found. Try a known sample.")
    else:
        st.info("Please enter a narrative.")

# ───────────────────────────────
# Visualization Section
st.markdown("---")
st.subheader("Complaint Category Visualization")

viz_type = st.radio("Choose Visualization Style:", ["Treemap", "Bar (Horizontal)", "Bubble Chart"])

if viz_type == "Treemap":
    fig = px.treemap(tag_counts, path=['Tag'], values='Count', title="Treemap of Complaint Categories")
elif viz_type == "Bar (Horizontal)":
    sorted_tags = tag_counts.sort_values("Count", ascending=True).reset_index(drop=True)

    # Customize top, mid, bottom thresholds
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

# ───────────────────────────────
# 🔚 CFPB Footer Logo
# ─── CENTERED FOOTER ───
# ─── BOTTOM FOOTER ───
# ─── CFPB LOGO BELOW CENTERED CAPTION ───
st.markdown("<hr>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

# ─── SINGLE-LINE FOOTER ───
st.markdown("<hr>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.empty()

with col2:
    st.markdown("### Powered by CFPB Open Consumer Complaint Data", unsafe_allow_html=True)

with col3:
    st.image("cfpb_logo.png", width=100)



