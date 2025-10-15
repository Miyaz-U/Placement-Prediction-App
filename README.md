# 🎓 Placement & Salary Predictor

An end-to-end machine learning project that predicts **whether a student will get placed** and, if placed, **estimates the expected salary** based on academic and personal details.

---

## 🚀 Project Overview
This project consists of:
1. **Model Training (`main.py`)**  
   - Cleans and preprocesses campus placement data  
   - Trains two models:
     - **Logistic Regression** for predicting placement  
     - **Linear Regression** for predicting salary (for placed students)  
   - Saves trained models and scalers as `.pkl` files  

2. **Interactive Web App (`placement_app.py`)**  
   - Built using **Streamlit**  
   - Allows users to input student details  
   - Predicts placement outcome and expected salary (for MBA students)  
   - Displays clean UI with real-time predictions  

---

## 🧠 Machine Learning Pipeline
### 1. Data Preprocessing
- Missing salary values for non-placed students are filled with 0  
- Categorical variables are label-encoded  
- New features:
  - `academic_avg` → Average of SSC, HSC, and Degree %  
  - `high_academic` → Binary indicator for academic excellence (>70%)  
  - `etest_mba_interaction` → Interaction between employability test & MBA %  

### 2. Modeling
- **Placement Model** → Logistic Regression  
- **Salary Model** → Linear Regression  
- Models evaluated using Accuracy, F1, ROC-AUC (for placement) and R², MAE, RMSE (for salary)  

### 3. Saving Artifacts
Trained models and scalers are saved as:
logreg_model.pkl
scaler_placement.pkl
linreg_model.pkl
scaler_salary.pkl


---

## 💻 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/placement-salary-predictor.git
cd placement-salary-predictor
pip install -r requirements.txt
python main.py
streamlit run placement_app.py
```


📊 Input Features for Prediction
| Feature              | Description                   |
| -------------------- | ----------------------------- |
| Gender               | Male / Female                 |
| 10th Grade %         | SSC percentage                |
| 12th Grade %         | HSC percentage                |
| Degree %             | Undergraduate percentage      |
| MBA %                | Postgraduate (MBA) percentage |
| Work Experience      | Yes / No                      |
| Specialisation       | Mkt&Fin / Mkt&HR              |
| Employability Test % | E-test score                  |


🧩 Technologies Used

Python 3.x

Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn

Streamlit

🧑‍💻 Developer

Developed by: Miyaz U
Dataset Source: Campus Placement Data (Kaggle)

📁 Folder Structure
📦 placement-salary-predictor
├── main.py
├── placement_app.py
├── Placement_Dataset.csv
├── logreg_model.pkl
├── linreg_model.pkl
├── scaler_placement.pkl
├── scaler_salary.pkl
├── requirements.txt
├── README.md
└── Logo.png


🏁 Output Examples

✅ Placement Prediction: “Likely to be Placed (Probability: 0.87)”

💰 Salary Prediction: “Estimated Salary (₹ 3,50,000)”

📜 License

This project is open-source and free to use for educational purposes.
