# =============================================================================
# CustomerGuard — Customer Churn Prediction System
# Run this on Google Colab (paste each section as a separate cell)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
# !pip install xgboost shap -q


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from xgboost import XGBClassifier
import shap
import time

print("✅ All libraries imported successfully")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load Dataset
# Option A: Upload manually → from google.colab import files; files.upload()
# Option B: Download directly from URL (used here)
# ─────────────────────────────────────────────────────────────────────────────
import os

LOCAL_CSV = "Telecom_customer_churn.csv"
URL_CSV   = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Try: local file → URL → Colab upload
if os.path.exists(LOCAL_CSV):
    df_raw = pd.read_csv(LOCAL_CSV)
    print(f"✅ Dataset loaded from local file: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
else:
    try:
        df_raw = pd.read_csv(URL_CSV)
        print(f"✅ Dataset loaded from URL: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    except Exception:
        print("⚠️  URL load failed. Please upload the Telco CSV manually:")
        from google.colab import files
        uploaded = files.upload()
        fname = list(uploaded.keys())[0]
        df_raw = pd.read_csv(fname)
        print(f"✅ Dataset loaded: {df_raw.shape}")

print("\nFirst 3 rows:")
print(df_raw.head(3))
print("\nColumn names:", df_raw.columns.tolist())


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Data Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

df = df_raw.copy()

# --- Step 1: Fix TotalCharges (spaces → NaN → numeric) ---
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# --- Step 2: Impute missing numerical values with median ---
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in num_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  Imputed {missing} missing values in '{col}' with median")

# --- Step 3: Drop customerID (not a feature) ---
if "customerID" in df.columns:
    df.drop(columns=["customerID"], inplace=True)

# --- Step 4: Encode target variable ---
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# --- Step 5: Label encode binary columns ---
binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
               "PaperlessBilling"]
for col in binary_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# --- Step 6: One-hot encode multi-class columns ---
multi_cols = ["MultipleLines", "InternetService", "OnlineSecurity",
              "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"]
multi_cols_present = [c for c in multi_cols if c in df.columns]
df = pd.get_dummies(df, columns=multi_cols_present, drop_first=True)

# --- Step 7: Separate features and target ---
X = df.drop(columns=["Churn"])
y = df["Churn"]

print(f"\n✅ Preprocessing complete")
print(f"   Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
print(f"   Churn rate: {y.mean()*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Exploratory Data Analysis (EDA)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 55)

churn_rate = df_raw["Churn"].map({"Yes": 1, "No": 0}).mean() * 100
print(f"\nOverall Churn Rate: {churn_rate:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("CustomerGuard — EDA Overview", fontsize=14, fontweight="bold")

# Plot 1: Churn Distribution
churn_counts = df_raw["Churn"].value_counts()
axes[0].bar(churn_counts.index, churn_counts.values,
            color=["steelblue", "tomato"], edgecolor="white", width=0.5)
axes[0].set_title("Churn Distribution")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Count")
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

# Plot 2: Churn by Contract Type
if "Contract" in df_raw.columns:
    contract_churn = df_raw.groupby("Contract")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100).reset_index()
    contract_churn.columns = ["Contract", "ChurnRate"]
    axes[1].bar(contract_churn["Contract"], contract_churn["ChurnRate"],
                color="steelblue", edgecolor="white")
    axes[1].set_title("Churn Rate by Contract Type")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].tick_params(axis="x", rotation=15)

# Plot 3: MonthlyCharges Distribution
axes[2].hist(df_raw[df_raw["Churn"] == "Yes"]["MonthlyCharges"].dropna(),
             bins=30, alpha=0.7, color="tomato", label="Churned")
axes[2].hist(df_raw[df_raw["Churn"] == "No"]["MonthlyCharges"].dropna(),
             bins=30, alpha=0.7, color="steelblue", label="Stayed")
axes[2].set_title("Monthly Charges Distribution")
axes[2].set_xlabel("Monthly Charges ($)")
axes[2].legend()

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ EDA plots saved as 'eda_overview.png'")


# Plot: Correlation Heatmap
plt.figure(figsize=(14, 10))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=False,
            linewidths=0.3, vmin=-1, vmax=1)
plt.title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Correlation matrix saved")

# Plot: Tenure histogram
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Feature Distributions: Churned vs Stayed", fontweight="bold")
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    churn_vals = df_raw[df_raw["Churn"] == "Yes"][col].dropna()
    stay_vals  = df_raw[df_raw["Churn"] == "No"][col].dropna()
    ax.hist(stay_vals,  bins=30, alpha=0.7, color="steelblue", label="Stayed")
    ax.hist(churn_vals, bins=30, alpha=0.7, color="tomato",    label="Churned")
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.legend()
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Train / Test Split + Feature Scaling
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

feature_names = X.columns.tolist()

print(f"✅ Split done")
print(f"   Train: {X_train_sc.shape[0]} rows  |  Test: {X_test_sc.shape[0]} rows")
print(f"   Train churn rate: {y_train.mean()*100:.2f}%")
print(f"   Test  churn rate: {y_test.mean()*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Train All Three Models
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  MODEL TRAINING")
print("=" * 55)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=200, random_state=42,
                                         use_label_encoder=False, eval_metric="logloss",
                                         verbosity=0),
}

trained_models = {}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train_sc, y_train)
    elapsed = time.time() - t0
    trained_models[name] = model
    print(f"  ✅ {name:25s} trained in {elapsed:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Evaluate All Models + Pick Best
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  MODEL EVALUATION")
print("=" * 55)

results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    results[name] = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "AUC-ROC":   roc_auc_score(y_test, y_prob),
    }

results_df = pd.DataFrame(results).T.round(4)
print("\n", results_df.to_string())

# Pick best model by AUC-ROC
best_name = results_df["AUC-ROC"].idxmax()
best_model = trained_models[best_name]
print(f"\n🏆  Best Model: {best_name}  (AUC-ROC = {results_df.loc[best_name,'AUC-ROC']:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Confusion Matrices + ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Confusion Matrices & ROC Curves — All Models", fontsize=14, fontweight="bold")

colors = ["steelblue", "tomato", "seagreen"]
roc_ax = None

for idx, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ax_cm = axes[0][idx]
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title(f"{name}\nConfusion Matrix", fontsize=10)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
    ax_cm.set_xticklabels(["No Churn","Churn"])
    ax_cm.set_yticklabels(["No Churn","Churn"])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i,j]), ha="center", va="center",
                       color="white" if cm[i,j] > thresh else "black",
                       fontsize=14, fontweight="bold")

    # ROC Curve (all on same axes)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    axes[1][idx].plot(fpr, tpr, color=colors[idx], lw=2,
                      label=f"AUC = {auc:.3f}")
    axes[1][idx].plot([0,1],[0,1],"k--", lw=1)
    axes[1][idx].set_title(f"{name} — ROC Curve", fontsize=10)
    axes[1][idx].set_xlabel("False Positive Rate")
    axes[1][idx].set_ylabel("True Positive Rate")
    axes[1][idx].legend(loc="lower right")

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Evaluation plots saved as 'model_evaluation.png'")

# Detailed classification report for best model
print(f"\n📋 Classification Report — {best_name}")
print(classification_report(y_test, best_model.predict(X_test_sc),
                             target_names=["No Churn","Churn"]))


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Churn Probability Prediction + Risk Classification
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  CHURN PROBABILITY + RISK CLASSIFICATION")
print("=" * 55)

def classify_risk(prob, low_max=0.40, medium_max=0.70):
    if prob < low_max:   return "Low"
    if prob < medium_max: return "Medium"
    return "High"

# Generate predictions on full test set
y_prob_best = best_model.predict_proba(X_test_sc)[:, 1]

# Build results dataframe
results_test = X_test.copy().reset_index(drop=True)
results_test["churn_probability"] = y_prob_best
results_test["risk_level"]        = [classify_risk(p) for p in y_prob_best]
results_test["actual_churn"]      = y_test.reset_index(drop=True)

# Risk distribution
risk_dist = results_test["risk_level"].value_counts()
print("\nRisk Level Distribution:")
for level in ["High","Medium","Low"]:
    count = risk_dist.get(level, 0)
    pct   = count / len(results_test) * 100
    print(f"  {level:8s}: {count:4d} customers ({pct:.1f}%)")

# Visualise risk distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Churn Risk Classification", fontsize=13, fontweight="bold")

risk_colors = {"Low": "seagreen", "Medium": "goldenrod", "High": "tomato"}
counts = [risk_dist.get(l,0) for l in ["Low","Medium","High"]]
axes[0].bar(["Low","Medium","High"], counts,
            color=[risk_colors[l] for l in ["Low","Medium","High"]],
            edgecolor="white", width=0.5)
axes[0].set_title("Customers by Risk Level")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

axes[1].pie(counts, labels=["Low","Medium","High"],
            colors=[risk_colors[l] for l in ["Low","Medium","High"]],
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor":"white","linewidth":1.5})
axes[1].set_title("Risk Level Distribution")

plt.tight_layout()
plt.savefig("risk_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# Churn probability vs Tenure scatter
fig, ax = plt.subplots(figsize=(10, 5))
for level, color in risk_colors.items():
    mask = results_test["risk_level"] == level
    ax.scatter(results_test.loc[mask, "tenure"],
               results_test.loc[mask, "churn_probability"],
               c=color, label=level, alpha=0.5, s=20)
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Churn Probability")
ax.set_title("Churn Probability vs Customer Tenure")
ax.legend(title="Risk Level")
ax.axhline(0.40, color="goldenrod", ls="--", lw=1.2, label="Low/Medium boundary")
ax.axhline(0.70, color="tomato",    ls="--", lw=1.2, label="Medium/High boundary")
plt.tight_layout()
plt.savefig("churn_vs_tenure.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Risk classification plots saved")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  SHAP EXPLAINABILITY")
print("=" * 55)

# Use TreeExplainer for RF / XGBoost, LinearExplainer for LR
if best_name in ("Random Forest", "XGBoost"):
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_sc)
    # For RF shap_values is a list [class0, class1]; take class 1
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
else:  # Logistic Regression
    explainer = shap.LinearExplainer(best_model, X_train_sc)
    sv = explainer.shap_values(X_test_sc)

print(f"✅ SHAP values computed for {sv.shape[0]} test customers")

# Global feature importance bar chart
mean_abs_shap = np.abs(sv).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(importance_df["feature"][::-1],
        importance_df["mean_abs_shap"][::-1],
        color="steelblue")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title(f"Top 15 Churn Drivers — {best_name} (SHAP)", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_global_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ SHAP global importance saved as 'shap_global_importance.png'")

print("\nTop 10 Features by SHAP Importance:")
print(importance_df[["feature","mean_abs_shap"]].head(10).to_string(index=False))


# SHAP Summary dot plot (beeswarm)
print("\nGenerating SHAP summary plot (beeswarm)...")
shap.summary_plot(sv, X_test_sc,
                  feature_names=feature_names,
                  show=False, max_display=15)
plt.title(f"SHAP Summary — {best_name}", fontweight="bold")
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Beeswarm plot saved as 'shap_summary_beeswarm.png'")

# SHAP force plot for a single HIGH-risk customer
high_risk_idx = results_test[results_test["risk_level"] == "High"].index
if len(high_risk_idx) > 0:
    sample_idx = high_risk_idx[0]
    base_val = (explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value)
    print(f"\nSHAP Force Plot — Customer at test index {sample_idx} "
          f"(Churn Probability: {results_test.loc[sample_idx,'churn_probability']:.3f})")
    force = shap.force_plot(base_val, sv[sample_idx],
                            feature_names=feature_names, matplotlib=True,
                            show=False, figsize=(16, 3))
    plt.tight_layout()
    plt.savefig("shap_force_sample.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Force plot saved as 'shap_force_sample.png'")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Retention Recommendation Engine
# ─────────────────────────────────────────────────────────────────────────────
RECOMMENDATION_RULES = {
    "tenure":            {"direction": "negative",
                          "text": "🎁 Enroll customer in a Loyalty Reward Program"},
    "Contract":          {"direction": "positive",
                          "text": "📋 Offer upgrade to a 1-year or 2-year contract"},
    "MonthlyCharges":    {"direction": "positive",
                          "text": "💰 Offer a discounted pricing plan or bundle"},
    "TechSupport":       {"direction": "positive",
                          "text": "🔧 Provide complimentary TechSupport upgrade"},
    "OnlineSecurity":    {"direction": "positive",
                          "text": "🔒 Provide complimentary OnlineSecurity upgrade"},
    "InternetService":   {"direction": "positive",
                          "text": "🌐 Review and improve Internet service package"},
    "PaperlessBilling":  {"direction": "positive",
                          "text": "📧 Offer billing assistance and paperless incentive"},
    "PaymentMethod":     {"direction": "positive",
                          "text": "💳 Offer convenient automatic payment options"},
}

def get_recommendations(risk_level, shap_row, feature_names, top_n=5):
    """Return retention recommendations based on risk level and top SHAP features."""
    if risk_level == "Low":
        return [{"status": "No intervention required — customer is low-risk."}]

    top_indices = np.argsort(np.abs(shap_row))[::-1][:top_n]
    recommendations = []
    seen_texts = set()

    for idx in top_indices:
        fname   = feature_names[idx]
        sval    = shap_row[idx]
        direction = "positive" if sval > 0 else "negative"

        # Match on base feature name (handles OHE columns like Contract_Two year)
        matched_key = None
        for key in RECOMMENDATION_RULES:
            if fname == key or fname.startswith(key + "_"):
                matched_key = key
                break

        if matched_key:
            rule = RECOMMENDATION_RULES[matched_key]
            text = rule["text"]
        else:
            text = f"📊 Review '{fname}' — significant churn driver"

        if text not in seen_texts:
            recommendations.append({
                "feature":    fname,
                "shap_value": round(float(sval), 4),
                "direction":  direction,
                "recommendation": text,
            })
            seen_texts.add(text)

    return recommendations

# Demo: show recommendations for 5 high-risk customers
print("=" * 60)
print("  RETENTION RECOMMENDATIONS — Sample High-Risk Customers")
print("=" * 60)

high_risk_samples = results_test[results_test["risk_level"] == "High"].head(5)
for i, row in high_risk_samples.iterrows():
    print(f"\n📌 Customer Index: {i}")
    print(f"   Churn Probability : {row['churn_probability']:.3f}")
    print(f"   Risk Level        : {row['risk_level']}")
    recs = get_recommendations(row["risk_level"], sv[i], feature_names)
    print("   Recommendations:")
    for r in recs[:3]:
        print(f"     • {r['recommendation']}  "
              f"[{r['feature']} | SHAP={r['shap_value']:+.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — Save Predictions CSV
# ─────────────────────────────────────────────────────────────────────────────
import json

rec_list = []
for i, prob in enumerate(y_prob_best):
    risk = classify_risk(prob)
    recs = get_recommendations(risk, sv[i], feature_names)
    rec_list.append(json.dumps(recs))

predictions_df = pd.DataFrame({
    "customer_index":    results_test.index,
    "churn_probability": y_prob_best.round(4),
    "risk_level":        [classify_risk(p) for p in y_prob_best],
    "actual_churn":      y_test.reset_index(drop=True),
    "recommendations":   rec_list,
})
predictions_df.to_csv("predictions.csv", index=False)
print(f"✅ Predictions saved to 'predictions.csv' ({len(predictions_df)} rows)")
print(predictions_df[["churn_probability","risk_level","actual_churn"]].head(10))


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Final Dashboard Summary (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("🛡️  CustomerGuard — Churn Analysis Dashboard",
             fontsize=16, fontweight="bold", y=0.98)

# ── KPI row ──────────────────────────────────────────────────────────────────
ax_kpi = fig.add_axes([0.0, 0.82, 1.0, 0.12])
ax_kpi.axis("off")
total     = len(predictions_df)
churn_pct = (predictions_df["actual_churn"].sum() / total * 100)
high_c    = (predictions_df["risk_level"] == "High").sum()
med_c     = (predictions_df["risk_level"] == "Medium").sum()
low_c     = (predictions_df["risk_level"] == "Low").sum()
kpis = [
    ("Total Customers",  f"{total}",     "steelblue"),
    ("Churn Rate",        f"{churn_pct:.1f}%", "tomato"),
    ("🔴 High Risk",      f"{high_c}",   "tomato"),
    ("🟡 Medium Risk",    f"{med_c}",    "goldenrod"),
    ("🟢 Low Risk",       f"{low_c}",    "seagreen"),
    ("Best Model",        best_name,     "slateblue"),
    ("AUC-ROC",           f"{results_df.loc[best_name,'AUC-ROC']:.4f}", "slateblue"),
]
for i, (label, value, color) in enumerate(kpis):
    ax_kpi.text(i / len(kpis) + 0.02, 0.75, label,
                transform=ax_kpi.transAxes, fontsize=8,
                color="gray", ha="left")
    ax_kpi.text(i / len(kpis) + 0.02, 0.25, value,
                transform=ax_kpi.transAxes, fontsize=13,
                color=color, fontweight="bold", ha="left")

# ── Chart 1: Risk Pie ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 4)
ax1.pie([high_c, med_c, low_c], labels=["High","Medium","Low"],
        colors=["tomato","goldenrod","seagreen"],
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor":"white","linewidth":1.5})
ax1.set_title("Risk Distribution", fontweight="bold")

# ── Chart 2: Top 10 SHAP Features ────────────────────────────────────────────
ax2 = fig.add_subplot(3, 3, 5)
top10 = importance_df.head(10)
ax2.barh(top10["feature"][::-1], top10["mean_abs_shap"][::-1], color="steelblue")
ax2.set_xlabel("Mean |SHAP|")
ax2.set_title("Top 10 Churn Drivers (SHAP)", fontweight="bold")
ax2.tick_params(axis="y", labelsize=8)

# ── Chart 3: Model Comparison Bar ────────────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 6)
metrics_to_plot = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
x_pos = np.arange(len(metrics_to_plot))
bar_colors = ["steelblue","tomato","seagreen"]
width = 0.25
for i, (mname, color) in enumerate(zip(results_df.index, bar_colors)):
    vals = [results_df.loc[mname, m] for m in metrics_to_plot]
    ax3.bar(x_pos + i*width, vals, width, label=mname, color=color, alpha=0.85)
ax3.set_xticks(x_pos + width)
ax3.set_xticklabels(metrics_to_plot, rotation=15, fontsize=8)
ax3.set_ylim(0.5, 1.0)
ax3.set_title("Model Comparison", fontweight="bold")
ax3.legend(fontsize=7)

# ── Chart 4: Churn Prob vs Tenure ────────────────────────────────────────────
ax4 = fig.add_subplot(3, 3, 7)
for level, color in {"High":"tomato","Medium":"goldenrod","Low":"seagreen"}.items():
    mask = predictions_df["risk_level"] == level
    ax4.scatter(results_test.loc[mask.values, "tenure"],
                predictions_df.loc[mask, "churn_probability"],
                c=color, alpha=0.4, s=10, label=level)
ax4.set_xlabel("Tenure (months)")
ax4.set_ylabel("Churn Probability")
ax4.set_title("Churn Probability vs Tenure", fontweight="bold")
ax4.legend(fontsize=7)

# ── Chart 5: Churn Rate by Contract ──────────────────────────────────────────
ax5 = fig.add_subplot(3, 3, 8)
if "Contract" in df_raw.columns:
    cr = df_raw.groupby("Contract")["Churn"].apply(
        lambda x: (x=="Yes").mean()*100).reset_index()
    cr.columns = ["Contract","ChurnRate"]
    ax5.bar(cr["Contract"], cr["ChurnRate"], color="steelblue", edgecolor="white")
    ax5.set_ylabel("Churn Rate (%)")
    ax5.set_title("Churn Rate by Contract", fontweight="bold")
    ax5.tick_params(axis="x", rotation=15)

# ── Chart 6: Monthly Charges boxplot by risk ─────────────────────────────────
ax6 = fig.add_subplot(3, 3, 9)
if "MonthlyCharges" in results_test.columns:
    box_data = [
        results_test[results_test["risk_level"] == l]["MonthlyCharges"].values
        for l in ["Low","Medium","High"]
    ]
    bp = ax6.boxplot(box_data, labels=["Low","Medium","High"],
                     patch_artist=True,
                     medianprops={"color":"black","linewidth":2})
    for patch, color in zip(bp["boxes"], ["seagreen","goldenrod","tomato"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax6.set_ylabel("Monthly Charges ($)")
    ax6.set_title("Monthly Charges by Risk Level", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.82])
plt.savefig("customerguard_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Dashboard saved as 'customerguard_dashboard.png'")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — Save Model Artifact + Download All Files
# ─────────────────────────────────────────────────────────────────────────────
import joblib

# Save the best model and scaler
joblib.dump({"model": best_model, "scaler": scaler,
             "feature_names": feature_names,
             "model_name": best_name},
            "best_model.pkl")
print("✅ Model saved as 'best_model.pkl'")

# Download all output files from Colab
try:
    from google.colab import files
    for fname in ["predictions.csv",
                  "customerguard_dashboard.png",
                  "shap_global_importance.png",
                  "shap_summary_beeswarm.png",
                  "model_evaluation.png",
                  "eda_overview.png",
                  "best_model.pkl"]:
        try:
            files.download(fname)
        except Exception:
            pass
    print("✅ Files downloaded to your local machine.")
except ImportError:
    print("ℹ️  Not running in Colab — files are saved in the current directory.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 16 — Final Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  🛡️  CUSTOMERGUARD — FINAL SUMMARY")
print("="*60)
print(f"\n  Dataset        : Telco Customer Churn")
print(f"  Total Samples  : {len(df_raw)}")
print(f"  Features       : {X.shape[1]}")
print(f"  Churn Rate     : {churn_rate:.2f}%\n")
print(f"  {'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7}")
print(f"  {'-'*55}")
for mname in results_df.index:
    r = results_df.loc[mname]
    marker = " 🏆" if mname == best_name else ""
    print(f"  {mname:<22} {r['Accuracy']:>6.4f} {r['Precision']:>6.4f} "
          f"{r['Recall']:>6.4f} {r['F1']:>6.4f} {r['AUC-ROC']:>7.4f}{marker}")
print(f"\n  Best Model     : {best_name}")
print(f"\n  Risk Levels    : Low={low_c}  Medium={med_c}  High={high_c}")
print(f"\n  Output Files:")
print(f"    • predictions.csv                 — per-customer churn probability + risk")
print(f"    • customerguard_dashboard.png     — full analysis dashboard")
print(f"    • shap_global_importance.png      — top churn drivers")
print(f"    • shap_summary_beeswarm.png       — SHAP beeswarm plot")
print(f"    • model_evaluation.png            — confusion matrices + ROC curves")
print(f"    • eda_overview.png                — EDA charts")
print(f"    • best_model.pkl                  — saved model artifact")
print("\n" + "="*60)
print("  ✅  CustomerGuard pipeline complete!")
print("="*60)
