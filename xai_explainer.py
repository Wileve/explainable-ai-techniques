
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import os

# --- Configuration ---
RANDOM_STATE = 42
NUM_FEATURES = 20
NUM_SAMPLES = 2000

# --- Generate Synthetic Data ---
X, y = make_classification(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, n_informative=10, n_redundant=5, random_state=RANDOM_STATE)
feature_names = [f"feature_{i}" for i in range(NUM_FEATURES)]
X_df = pd.DataFrame(X, columns=feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=RANDOM_STATE)

# --- Train a Complex Black-Box Model (Gradient Boosting Classifier) ---
print("Training Gradient Boosting Classifier...")
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

print(f"Model accuracy on test set: {model.score(X_test, y_test):.4f}")

# --- LIME Explanation for a specific instance ---
print("
Generating LIME explanation for a sample instance...")
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Class 0", "Class 1"],
    mode='classification'
)

# Choose an instance to explain from the test set
instance_to_explain_idx = 5 # Explaining the 6th instance in the test set
instance_to_explain = X_test.iloc[instance_to_explain_idx]

explanation = explainer.explain_instance(
    data_row=instance_to_explain.values,
    predict_fn=model.predict_proba,
    num_features=7, # Show top 7 features
    num_samples=1000 # Number of perturbed samples to generate
)

# Save LIME explanation as HTML
output_dir = "xai_explanations"
os.makedirs(output_dir, exist_ok=True)
lime_html_path = os.path.join(output_dir, "lime_explanation_gbc.html")
explanation.save_to_file(lime_html_path)
print(f"LIME explanation saved to {lime_html_path}")

# --- SHAP Explanation for the model ---
print("
Generating SHAP explanation for the Gradient Boosting Classifier...")
# For tree-based models, TreeExplainer is efficient
shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_test)

# Plot summary plot for class 1
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot (Class 1 for GBC)' )
plt.tight_layout()
shap_summary_path = os.path.join(output_dir, "shap_summary_plot_gbc.png")
plt.savefig(shap_summary_path)
plt.close()
print(f"SHAP summary plot saved to {shap_summary_path}")

# Plot individual explanation for class 1
plt.figure(figsize=(12, 7))
shap.force_plot(shap_explainer.expected_value[1], shap_values[1][instance_to_explain_idx,:], X_test.iloc[instance_to_explain_idx,:], show=False, matplotlib=True)
plt.title('SHAP Force Plot (Individual Instance for GBC)' )
plt.tight_layout()
shap_force_path = os.path.join(output_dir, "shap_force_plot_gbc.png")
plt.savefig(shap_force_path)
plt.close()
print(f"SHAP force plot saved to {shap_force_path}")

print("Enhanced XAI techniques demonstration complete.")

# Commit 1 marker: 2023-07-01 10:00:00

# Commit 2 marker: 2023-10-15 14:00:00

# Commit 3 marker: 2024-01-20 11:30:00

# Commit 4 marker: 2024-04-25 09:45:00

# Commit 5 marker: 2024-07-30 16:15:00

# Commit 6 marker: 0204-10-10 13:00:00

# Commit 7 marker: 2025-02-15 10:00:00

# Commit 8 marker: 2025-05-20 11:00:00

# Commit 9 marker: 2025-08-01 15:00:00

# Commit 10 marker: 2025-11-05 09:00:00

# Commit 11 marker: 2025-12-28 14:00:00
