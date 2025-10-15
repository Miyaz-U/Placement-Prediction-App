# Import required libraries
import streamlit as st
import numpy as np
import pickle


# Load Pre-trained Models & Scalers
with open("logreg_model.pkl", "rb") as f:
    logreg = pickle.load(f)

with open("scaler_placement.pkl", "rb") as f:
    scaler_placement = pickle.load(f)

with open("linreg_model.pkl", "rb") as f:
    linreg = pickle.load(f)

with open("scaler_salary.pkl", "rb") as f:
    scaler_salary = pickle.load(f)


# Streamlit Page Setup
st.set_page_config(page_title="Placement & Salary Predictor", layout="wide")

# Sidebar: logo + project info
st.sidebar.image("Logo.png", use_container_width=True)
st.sidebar.title("About This Project")
st.sidebar.markdown("""
### 🎓 Placement & Salary Predictor
This app predicts **whether a student will get placed** and, if placed, **estimates the expected salary**.

**Key Features**
- Predicts placement probability using Logistic Regression  
- Predicts salary using Linear Regression  
- Clean, interactive UI built with Streamlit  

**Developed by:** *Miyaz U*  
**Dataset Source:** *Campus Placement Data (Kaggle)*  
""")


# Main Interface
st.title("🎓 Interactive Placement & Salary Predictor")
st.markdown("Enter student details below to predict placement outcome and salary.")

# Ask if user studied MBA
has_mba = st.radio("Have you studied MBA?", ["Yes", "No"], index=0, horizontal=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc_p = st.slider("10th Grade %", 0, 100, 75)
    hsc_p = st.slider("12th Grade %", 0, 100, 75)
    degree_p = st.slider("Degree %", 0, 100, 70)

    if has_mba == "Yes":
        mba_p = st.slider("MBA %", 0, 100, 70)
    else:
        mba_p = 0  # default if no MBA

with col2:
    workex = st.selectbox("Work Experience", ["Yes", "No"])

    if has_mba == "Yes":
        specialisation = st.selectbox("MBA Specialisation", ["Mkt&Fin", "Mkt&HR"])
        etest_p = st.slider("Employability Test %", 0, 100, 60)
    else:
        specialisation = "None"
        etest_p = 0  # default if no MBA


# Feature Engineering for Input
academic_avg = np.mean([ssc_p, hsc_p, degree_p])
high_academic = 1 if academic_avg > 70 else 0
etest_mba_interaction = etest_p * mba_p

# Encode categorical inputs same as training
gender_val = 1 if gender == "Male" else 0
workex_val = 1 if workex == "Yes" else 0
specialisation_val = 0 if specialisation == "Mkt&Fin" else 1

# Prepare input for placement prediction
user_input = np.array([[academic_avg, ssc_p, hsc_p, degree_p, workex_val,
                        specialisation_val, etest_p, high_academic]])
user_input_scaled = scaler_placement.transform(user_input)


# Predict Placement
placement_pred = logreg.predict(user_input_scaled)[0]
placement_prob = logreg.predict_proba(user_input_scaled)[0][1]

st.subheader("📌 Placement Prediction")
col3, col4 = st.columns(2)

with col3:
    if placement_pred == 1:
        st.success(f"✅ Likely to be Placed\nProbability: {placement_prob:.2f}")
    else:
        st.error(f"❌ Unlikely to be Placed\nProbability: {1 - placement_prob:.2f}")


# Predict Salary (if placed)
if placement_pred == 1 and has_mba == "Yes":
    salary_input = np.array([[ssc_p, hsc_p, degree_p, mba_p,
                              workex_val, etest_p, academic_avg]])
    salary_input_scaled = scaler_salary.transform(salary_input)
    predicted_salary = linreg.predict(salary_input_scaled)[0]

    with col4:
        st.subheader("💰 Predicted Salary")
        st.metric(label="Estimated Salary (₹)", value=f"{predicted_salary:,.0f}")

elif placement_pred == 1 and has_mba == "No":
    with col4:

        st.warning("💡 Salary prediction is available only for MBA students.")
