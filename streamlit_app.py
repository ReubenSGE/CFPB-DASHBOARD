pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import pandas as pd

# Load data
df = pd.read_csv("/mnt/data/150clusterbetter.csv")

# Drop missing entries
df = df.dropna(subset=["Consumer complaint narrative", "New Issue Tag"])

# Balance the dataset: take max 150 per class
balanced_df = df.groupby("New Issue Tag").apply(lambda x: x.sample(n=min(len(x), 150), random_state=42)).reset_index(drop=True)

# Sentence embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X = embedder.encode(balanced_df["Consumer complaint narrative"].tolist(), show_progress_bar=True)
y = balanced_df["New Issue Tag"].tolist()

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# Save model and embedder
import joblib
joblib.dump(model, "/mnt/data/logreg_model.pkl")
joblib.dump(embedder, "/mnt/data/embedder_model.pkl")

"✅ Model and embedder successfully trained and saved."
