# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier  # ✅ CHANGED
# import shap
# import joblib
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # ✅ Create models/ folder if it doesn't exist
# os.makedirs('models', exist_ok=True)

# # 1. Load Data
# df = pd.read_excel('Ecommerce_Dataset.xlsx')

# # 2. Basic Cleaning
# df = df.drop(columns=['CustomerID'])

# imputer_num = SimpleImputer(strategy='median')
# num_cols = df.select_dtypes(include=['float64', 'int64']).columns
# df[num_cols] = imputer_num.fit_transform(df[num_cols])

# imputer_cat = SimpleImputer(strategy='most_frequent')
# cat_cols = df.select_dtypes(include=['object']).columns
# df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# # 3. Encoding
# encoders = {}
# for col in cat_cols:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col].astype(str))
#     encoders[col] = le

# # 4. Split Data
# X = df.drop('Churn', axis=1)
# y = df['Churn']

# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# print(f"Data Split: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")

# # 5. Train Random Forest  ✅ CHANGED
# model = RandomForestClassifier(
#     n_estimators=200,
#     max_depth=10,
#     min_samples_split=10,
#     min_samples_leaf=4,
#     class_weight='balanced',
#     random_state=42,
#     n_jobs=-1
# )

# model.fit(X_train, y_train)

# print(f"Train Score: {model.score(X_train, y_train):.2f}")
# print(f"Val Score:   {model.score(X_val, y_val):.2f}")
# print(f"Test Score:  {model.score(X_test, y_test):.2f}")

# # 6. Save all artifacts to models/ folder  ✅ CHANGED filenames + path
# joblib.dump(model,                        'models/rf_churn_model.pkl')      # ✅ new name
# joblib.dump(encoders,                     'models/encoders.pkl')             # same
# joblib.dump(X_train.columns.tolist(),     'models/feature_names.pkl')        # same

# # ✅ NEW — save SHAP explainer separately (speeds up Streamlit)
# explainer = shap.TreeExplainer(model)
# joblib.dump(explainer,                    'models/shap_explainer.pkl')

# print("✅ All artifacts saved to models/ folder!")

# # 7. Confusion Matrix
# y_pred = model.predict(X_test)
# ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
# plt.title("Random Forest - Test Set Performance")
# plt.show()

# # 8. SHAP Summary Plot  ✅ CHANGED: shap_values[1]
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values[1], X_test)  # [1] = churn class

# # 9. Local Explanation  ✅ CHANGED: shap_values[1][idx, :]
# high_risk_indices = np.where(y_pred == 1)[0]
# if len(high_risk_indices) > 0:
#     idx = high_risk_indices[0]
#     contributions = dict(zip(X_test.columns, shap_values[1][idx, :]))  # ✅ [1][idx,:]
#     sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

#     print(f"\nTop Factors for Customer {idx}:")
#     for f, v in sorted_contribs[:3]:
#         print(f"- {f}: {v:.4f}")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

os.makedirs('models', exist_ok=True)

# 1. Load Data
df = pd.read_excel('Ecommerce_Dataset.xlsx')

# 2. Basic Cleaning
df = df.drop(columns=['CustomerID'])

imputer_num = SimpleImputer(strategy='median')
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = imputer_num.fit_transform(df[num_cols])

imputer_cat = SimpleImputer(strategy='most_frequent')
cat_cols = df.select_dtypes(include=['object']).columns
df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

# 3. Encoding
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# 4. Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Data Split: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")

# 5. Train Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print(f"Train Score: {model.score(X_train, y_train):.2f}")
print(f"Val Score:   {model.score(X_val, y_val):.2f}")
print(f"Test Score:  {model.score(X_test, y_test):.2f}")

# 6. Save Artifacts
joblib.dump(model,                     'models/rf_churn_model.pkl')
joblib.dump(encoders,                  'models/encoders.pkl')
joblib.dump(X_train.columns.tolist(),  'models/feature_names.pkl')

explainer = shap.TreeExplainer(model)
joblib.dump(explainer,                 'models/shap_explainer.pkl')

print("✅ All artifacts saved to models/ folder!")

# 7. Confusion Matrix
y_pred = model.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Blues')
plt.title("Random Forest - Test Set Performance")
plt.show()

# 8. SHAP — ✅ 3D array fix
shap_values = explainer.shap_values(X_test)

# ✅ Handle both old (list) and new (3D array) SHAP versions
if isinstance(shap_values, list):
    sv_churn = shap_values[1]           # old SHAP → list[class_1]
else:
    sv_churn = shap_values[:, :, 1]     # new SHAP → 3D [samples, features, classes]

print(f"SHAP type: {type(shap_values)} | sv_churn shape: {sv_churn.shape}")

# Summary Plot
shap.summary_plot(sv_churn, X_test)

# 9. Local Explanation
high_risk_indices = np.where(y_pred == 1)[0]
if len(high_risk_indices) > 0:
    idx = high_risk_indices[0]
    contributions = dict(zip(X_test.columns, sv_churn[idx, :]))
    sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    print(f"\nTop Factors for Customer {idx}:")
    for f, v in sorted_contribs[:3]:
        print(f"- {f}: {v:.4f}")