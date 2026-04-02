
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import os

# --- Configuration ---
RANDOM_STATE = 42
NUM_FEATURES = 10
NUM_SAMPLES = 1000

# --- Generate Synthetic Data ---
X, y = make_classification(n_samples=NUM_SAMPLES, n_features=NUM_FEATURES, n_informative=5, n_redundant=0, random_state=RANDOM_STATE)
feature_names = [f'feature_{i}' for i in range(NUM_FEATURES)]
X_df = pd.DataFrame(X, columns=feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=RANDOM_STATE)

# --- Train a Black-Box Model (Random Forest) ---
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

print(f"Model accuracy on test set: {model.score(X_test, y_test):.4f}")

# --- LIME Explanation ---
print("
Generating LIME explanation...")
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Class 0', 'Class 1'],
    mode='classification'
)

# Choose an instance to explain
instance_to_explain_idx = 0
instance_to_explain = X_test.iloc[instance_to_explain_idx]

explanation = explainer.explain_instance(
    data_row=instance_to_explain.values,
    predict_fn=model.predict_proba,
    num_features=5
)

# Save LIME explanation as HTML
explanation.save_to_file(os.path.join(repo5_name, "lime_explanation.html"))
print(f"LIME explanation saved to {repo5_name}/lime_explanation.html")

# --- SHAP Explanation ---
print("
Generating SHAP explanation...")
# For tree-based models, TreeExplainer is efficient
shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_test)

# Plot summary plot for class 1
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
plt.title('SHAP Summary Plot (Class 1)' )
plt.tight_layout()
plt.savefig(os.path.join(repo5_name, "shap_summary_plot.png"))
plt.close()
print(f"SHAP summary plot saved to {repo5_name}/shap_summary_plot.png")

# Plot individual explanation for class 1
plt.figure(figsize=(10, 6))
shap.force_plot(shap_explainer.expected_value[1], shap_values[1][instance_to_explain_idx,:], X_test.iloc[instance_to_explain_idx,:], show=False, matplotlib=True)
plt.title('SHAP Force Plot (Individual Instance)' )
plt.tight_layout()
plt.savefig(os.path.join(repo5_name, "shap_force_plot.png"))
plt.close()
print(f"SHAP force plot saved to {repo5_name}/shap_force_plot.png")

print("XAI techniques demonstration complete.")
