# DS 4320 Project 2 Pipeline
## Detecting AI-Generated Academic Text

**Author:** Dailin Li  
**NetID:** esd4uq  
**Date:** 2026-04-11

This notebook implements the full detection pipeline:
1. Query MongoDB Atlas → pandas DataFrame
2. Feature engineering: TF-IDF + statistical features
3. Logistic Regression classifier
4. Evaluation: accuracy, precision, recall, F1, confusion matrix
5. Publication-quality visualizations: ROC curve + confusion matrix heatmap

## 0. Setup & Imports


```python
import os
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.sparse import hstack, csr_matrix

from pymongo import MongoClient

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/project2_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)

# publication-quality plot defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RANDOM_STATE = 42
logger.info("Imports complete.")
```

## 1. Data Preparation — Query MongoDB

We connect to MongoDB Atlas and pull all documents from the `ai_text_detection` collection into a pandas DataFrame.  Each document contains the raw text, a binary label (`human` / `ai`), and pre-computed statistical features that were stored at ingest time.


```python
# ── connection ────────────────────────────────────────────────────────────────
# Set MONGO_URI in your environment before running, e.g.:
#   export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/"
MONGO_URI = os.environ.get("MONGO_URI", "")
if not MONGO_URI:
    raise EnvironmentError("Set the MONGO_URI environment variable before running this notebook.")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
client.admin.command("ping")  # verify connection
logger.info("Connected to MongoDB Atlas.")

collection = client["project2"]["ai_text_detection"]

# ── pull all documents ────────────────────────────────────────────────────────
# Exclude the internal _id and ingested_at fields from the DataFrame
cursor = collection.find({}, {"_id": 0, "ingested_at": 0})
df = pd.DataFrame(list(cursor))
logger.info("Loaded %d documents from MongoDB.", len(df))

print(f"Total documents: {len(df):,}")
print(f"Label distribution:\n{df['label'].value_counts()}")
df.head(3)
```


```python
# ── encode labels ─────────────────────────────────────────────────────────────
# 1 = AI-generated, 0 = human
df["label_enc"] = (df["label"] == "ai").astype(int)

# drop any rows with missing text
df = df.dropna(subset=["text"]).reset_index(drop=True)
logger.info("After cleaning: %d rows.", len(df))
```

## 2. Feature Engineering

**Decision rationale:**  
We combine two complementary feature sets:

- **TF-IDF** (top 5,000 unigrams + bigrams): captures lexical patterns — AI text tends toward formal vocabulary and particular phrase structures.  
- **Statistical features** (word count, avg word length, sentence count, punctuation density): capture stylometric signals that persist regardless of topic.  

Both feature matrices are stacked horizontally with `scipy.sparse.hstack` so Logistic Regression receives a single sparse design matrix.


```python
# ── train/test split ─────────────────────────────────────────────────────────
X_text = df["text"]
stat_cols = ["word_count", "avg_word_length", "sentence_count", "punctuation_density"]
X_stats = df[stat_cols].values
y = df["label_enc"].values

(
    X_text_train, X_text_test,
    X_stats_train, X_stats_test,
    y_train, y_test,
) = train_test_split(
    X_text, X_stats, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)
logger.info("Train: %d  Test: %d", len(y_train), len(y_test))
print(f"Train size: {len(y_train):,}   Test size: {len(y_test):,}")
```


```python
# ── TF-IDF ────────────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # unigrams and bigrams
    sublinear_tf=True,    # apply log(1+tf) to compress extreme term frequencies
    strip_accents="unicode",
    min_df=3,             # ignore terms appearing in fewer than 3 docs
)

X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test  = tfidf.transform(X_text_test)
logger.info("TF-IDF vocabulary size: %d", len(tfidf.vocabulary_))

# ── scale statistical features so they are on the same magnitude as TF-IDF ───
scaler = StandardScaler()
X_stats_train_scaled = scaler.fit_transform(X_stats_train)
X_stats_test_scaled  = scaler.transform(X_stats_test)

# ── combine into one design matrix ───────────────────────────────────────────
X_train = hstack([X_tfidf_train, csr_matrix(X_stats_train_scaled)])
X_test  = hstack([X_tfidf_test,  csr_matrix(X_stats_test_scaled)])
print(f"Feature matrix shape — train: {X_train.shape}  test: {X_test.shape}")
```

## 3. Model — Logistic Regression

**Decision rationale:**  
Logistic Regression is the appropriate choice here for several reasons:

- It is a well-understood probabilistic classifier taught in DS 3021/4021.
- It scales efficiently to sparse, high-dimensional TF-IDF matrices.
- The learned coefficients are directly interpretable — we can inspect which terms the model associates most strongly with AI-generated text.
- `liblinear` solver is recommended for smaller dense/sparse datasets with L1 or L2 regularisation.

Regularisation strength `C=1.0` (default) is used; a smaller `C` increases regularisation and can help if the model over-fits to vocabulary.


```python
# ── fit model ─────────────────────────────────────────────────────────────────
clf = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=1000,
    random_state=RANDOM_STATE,
)
clf.fit(X_train, y_train)
logger.info("Logistic Regression training complete.")

# ── predictions ───────────────────────────────────────────────────────────────
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # probability of class 1 (AI)

# ── metrics ───────────────────────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_proba)

logger.info("Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f  AUC=%.4f", acc, prec, rec, f1, auc)

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))
print(f"ROC-AUC: {auc:.4f}")
```

## 4. Visualize Results

**Decision rationale for visualizations:**  

- **ROC curve**: Shows the trade-off between true positive rate and false positive rate across all classification thresholds. The area under the curve (AUC) is a threshold-independent measure of discriminative ability, making it the standard metric for binary classifiers.
- **Confusion matrix heatmap**: Provides an absolute count view of prediction errors. For a detection task, false negatives (AI text classified as human) are often more costly than false positives, so the raw cell counts inform operational thresholding decisions.
- **Top-20 TF-IDF features**: Interprets which specific words and bigrams drive the model's predictions in each direction, building trust in the model's reasoning.


```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── left: ROC curve ───────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax = axes[0]
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"Logistic Regression (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — AI Text Detection", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

# ── right: confusion matrix ───────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
ax2 = axes[1]
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred: Human", "Pred: AI"],
    yticklabels=["True: Human", "True: AI"],
    linewidths=0.5,
    ax=ax2,
    cbar=False,
)
ax2.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
ax2.set_xlabel("Predicted Label", fontsize=12)
ax2.set_ylabel("True Label", fontsize=12)

plt.tight_layout()
plt.savefig("roc_confusion.png", bbox_inches="tight", dpi=150)
logger.info("Saved roc_confusion.png")
plt.show()
```


```python
# ── top-20 most predictive TF-IDF features ────────────────────────────────────
# Only the TF-IDF portion of the coefficient vector (first len(vocab) entries)
feature_names = tfidf.get_feature_names_out()
coefs = clf.coef_[0][: len(feature_names)]  # slice off the 4 stat-feature coefs

top_n = 20
top_ai_idx    = np.argsort(coefs)[-top_n:][::-1]
top_human_idx = np.argsort(coefs)[:top_n]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, idx, title, color in [
    (axes[0], top_ai_idx,    "Top 20 terms → AI",    "#d62728"),
    (axes[1], top_human_idx, "Top 20 terms → Human", "#1f77b4"),
]:
    terms = [feature_names[i] for i in idx]
    vals  = [coefs[i] for i in idx]
    y_pos = range(len(terms))
    ax.barh(y_pos, vals, color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=10)
    ax.set_xlabel("Logistic Regression Coefficient", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8)

plt.suptitle("Most Predictive TF-IDF Features", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("top_features.png", bbox_inches="tight", dpi=150)
logger.info("Saved top_features.png")
plt.show()
```

## 5. Summary

The Logistic Regression model trained on combined TF-IDF and statistical text features demonstrates strong discriminative ability for distinguishing AI-generated academic text from human-written responses.  

Key findings:
- The ROC-AUC score confirms the model meaningfully out-performs a random baseline.
- The confusion matrix shows that false negatives (AI text misclassified as human) are relatively rare, which is the operationally important error type in an academic-integrity context.
- The top-feature chart reveals that AI-generated text tends to use specific formal phrases and hedging language, while human text exhibits more casual and domain-specific vocabulary.

**Limitations:** The HC3 dataset is sourced from a specific version of ChatGPT. Newer LLMs may produce text with different stylometric signatures, reducing the model's generalizability over time.
