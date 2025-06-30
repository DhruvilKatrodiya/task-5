# ❤️ Heart Disease Classification using Trees
# Task 5: Decision Tree & Random Forest (AI/ML Internship)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("heart.csv")
print("✅ Dataset Loaded: ", df.shape)
print(df.head())

# Info & Clean
print(df.info())
print(df.isnull().sum())

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Visualize Full Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=["No Disease", "Disease"])
plt.title("Decision Tree - Full Depth")
plt.savefig("decision_tree_full.png")
plt.show()

# Limited Depth Tree
dt_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_limited.fit(X_train, y_train)
y_pred_limited = dt_limited.predict(X_test)

print("Limited Tree Accuracy:", accuracy_score(y_test, y_pred_limited))
print(classification_report(y_test, y_pred_limited))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Feature Importance
importances = rf_model.feature_importances_
feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.show()

# Cross-Validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("CV Scores:", cv_scores)
print("Avg Accuracy:", cv_scores.mean())
