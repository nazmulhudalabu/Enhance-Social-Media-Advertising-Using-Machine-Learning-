# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ================================
# 2. LOAD DATASET
# ================================
df = pd.read_excel("ppc_campaign_performance_data.xlsx")

# Drop non-predictive columns if present
df = df.drop(columns=["Campaign_ID", "Date"], errors="ignore")

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# ================================
# 3. DEFINE FEATURES & TARGET
# ================================
X = df.drop("Revenue", axis=1)
y = pd.qcut(df["Revenue"], q=5, labels=False)  # Convert to classification

# ================================
# 4. TRAIN–TEST SPLIT (SAME FOR ALL)
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ================================
# 5. MODELS (OVERFITTING FIXED)
# ================================
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,        # FIX overfitting
        min_samples_split=10,
        random_state=42
    ),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Naive Bayes": GaussianNB()
}

# ================================
# 6. TRAIN, EVALUATE, STORE METRICS
# ================================
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": []
}

predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    results["Model"].append(name)
    results["Accuracy"].append(accuracy_score(y_test, y_pred))
    results["Precision"].append(precision_score(y_test, y_pred, average="weighted"))
    results["Recall"].append(recall_score(y_test, y_pred, average="weighted"))
    results["F1"].append(f1_score(y_test, y_pred, average="weighted"))

results_df = pd.DataFrame(results)

# ================================
# 7. CONFUSION MATRICES
# ================================
for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# ================================
# 8. CORRELATION MATRIX
# ================================
plt.figure(figsize=(14,10))
sns.heatmap(X.corr(), cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix Showing Multicollinearity Among Features")
plt.tight_layout()
plt.show()

# ================================
# 9. INDIVIDUAL METRIC COMPARISONS
# ================================
metrics = ["Accuracy", "Precision", "Recall", "F1"]

for metric in metrics:
    plt.figure(figsize=(6,4))
    plt.bar(results_df["Model"], results_df[metric])
    plt.title(f"Model Comparison based on {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

# ================================
# 10. SINGLE COMBINED COMPARISON GRAPH
# ================================
x = np.arange(len(results_df["Model"]))
width = 0.2

plt.figure(figsize=(10,6))
plt.bar(x - 1.5*width, results_df["Accuracy"], width, label="Accuracy")
plt.bar(x - 0.5*width, results_df["Precision"], width, label="Precision")
plt.bar(x + 0.5*width, results_df["Recall"], width, label="Recall")
plt.bar(x + 1.5*width, results_df["F1"], width, label="F1-Score")

plt.xticks(x, results_df["Model"], rotation=20)
plt.ylabel("Score")
plt.title("Accuracy, Precision, Recall, and F1-Score Comparison Across Models")
plt.legend()
plt.tight_layout()
plt.show()
