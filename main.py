# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder as le, StandardScaler as ss
from sklearn.linear_model import LogisticRegression as logr, LinearRegression as linr
from sklearn.metrics import accuracy_score as accs, precision_score as ps, recall_score as rs, f1_score as f1
from sklearn.metrics import confusion_matrix as cm, roc_auc_score as ras, roc_curve as rc
from sklearn.metrics import r2_score as r2, mean_absolute_error as mae, mean_squared_error as mse
import pickle as p


# Step 1: Load Dataset
data = pd.read_csv("Placement_Dataset.csv")
print("Dataset Preview:")
print(data.head())
print("="*60)



# Step 2: Basic Info & Missing Values
print("Dataset Info:")
print(data.info())
print("="*60)

print("Missing Values:")
print(data.isnull().sum())
print("="*60)

# Fill missing salary values with 0 for non-placed students
data.loc[data['status'] == 'Not Placed', 'salary'] = 0



# Step 3: Encode Categorical Variables
cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']

for col in cat_cols:
    le_col = le()
    data[col] = le_col.fit_transform(data[col])

print("Data after encoding categorical columns:")
print(data.head())
print("="*60)



# Step 4: Feature Engineering
# Academic average
data['academic_avg'] = data[['ssc_p', 'hsc_p', 'degree_p']].mean(axis=1)

# High academic indicator
data['high_academic'] = (data['academic_avg'] > 70).astype(int)

# Interaction term: Employability test x MBA percentage
data['etest_mba_interaction'] = data['etest_p'] * data['mba_p']


# Step 5: Correlation Check
new_corr = data.corr(numeric_only=True)['status'].sort_values(ascending=False)
print("Feature correlation with target 'status':")
print(new_corr)
print("="*60)



# Step 6: Feature Selection
selected_features = [
    'academic_avg',     # Combined academic strength
    'ssc_p',            # 10th %
    'hsc_p',            # 12th %
    'degree_p',         # Degree %
    'workex',           # Work experience
    'specialisation',   # MBA specialization
    'etest_p',          # Employability test
    'high_academic'     # Strong academic flag
]

X_final = data[selected_features]
y_final = data['status']

print("Final selected features:", list(X_final.columns))
print("X_final shape:", X_final.shape)
print("y_final shape:", y_final.shape)
print("="*60)


# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = tts(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("="*60)



# Step 8: Feature Scaling
scaler = ss()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling complete. Data ready for modeling.")
print("="*60)


# Step 9: Model Training & Evaluation (Placed or not)
# Initialize Logistic Regression model
logreg = logr(random_state=42, max_iter=1000)

# Train the model
logreg.fit(X_train_scaled, y_train)

# Make predictions
# Predict on test set
y_pred = logreg.predict(X_test_scaled)

# Predict probabilities (for ROC curve)
y_prob = logreg.predict_proba(X_test_scaled)[:,1]

# Evaluate the model
accuracy = accs(y_test, y_pred)
precision = ps(y_test, y_pred)
recall = rs(y_test, y_pred)
f1 = f1(y_test, y_pred)
roc_auc = ras(y_test, y_prob)

print("Model Evaluation Metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")


# Confusion matrix
cm = cm(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Placed','Placed'], yticklabels=['Not Placed','Placed'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ROC Curve
fpr, tpr, thresholds = rc(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_final.columns,
    'Coefficient': logreg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("Feature Importance (Logistic Regression):")
print(feature_importance)
print("="*60)



# Step 10: Model Training & Evaluation (Salary)
# Only consider students who got placed
placed_data = data[data['status'] == 1].copy()  # status = 1 means Placed

# Features for salary prediction (can include academics and work experience)
salary_features = [
    'ssc_p', 'hsc_p', 'degree_p', 'mba_p', 'workex', 'etest_p', 'academic_avg'
]

X_salary = placed_data[salary_features]
y_salary = placed_data['salary']

print("Salary prediction dataset shape:", X_salary.shape)
print("="*60)


X_train_sal, X_test_sal, y_train_sal, y_test_sal = tts(
    X_salary, y_salary, test_size=0.2, random_state=42
)

# Scale features
scaler_sal = ss()
X_train_sal_scaled = scaler_sal.fit_transform(X_train_sal)
X_test_sal_scaled = scaler_sal.transform(X_test_sal)


# Initialize and train
linreg = linr()
linreg.fit(X_train_sal_scaled, y_train_sal)


# Make predictions
y_pred_sal = linreg.predict(X_test_sal_scaled)


r2 = r2(y_test_sal, y_pred_sal)
mae = mae(y_test_sal, y_pred_sal)
rmse = np.sqrt(mse(y_test_sal, y_pred_sal))

print("Salary Prediction Metrics:")
print(f"R² Score      : {r2:.4f}")
print(f"MAE           : {mae:.2f}")
print(f"RMSE          : {rmse:.2f}")


# Compare actual vs predicted salaries
plt.figure(figsize=(6,4))
plt.scatter(y_test_sal, y_pred_sal, color='blue')
plt.plot([y_test_sal.min(), y_test_sal.max()], [y_test_sal.min(), y_test_sal.max()], 'r--')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()


# Feature importance for salary prediction
coeff_df = pd.DataFrame({
    'Feature': salary_features,
    'Coefficient': linreg.coef_
}).sort_values(by='Coefficient', ascending=False)

print("Feature Importance (Salary Prediction):")
print(coeff_df)
print("="*60)


# Step 11: Save Trained Models & Scalers
# Save logistic regression model and scaler
with open("logreg_model.pkl", "wb") as f:
    p.dump(logreg, f)

with open("scaler_placement.pkl", "wb") as f:
    p.dump(scaler, f)

# Save linear regression model and salary scaler
with open("linreg_model.pkl", "wb") as f:
    p.dump(linreg, f)

with open("scaler_salary.pkl", "wb") as f:
    p.dump(scaler_sal, f)

print("✅ Models and scalers saved successfully!")