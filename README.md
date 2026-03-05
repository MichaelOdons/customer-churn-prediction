# Telecom Customer Churn Prediction

![Churn Prediction](https://img.shields.io/badge/Status-Completed-success)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Technologies & Libraries](#technologies--libraries)
4. [Dataset](#dataset)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Key Insights](#key-insights)
7. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
8. [Modeling](#modeling)
9. [Web App Deployment](#web-app-deployment)
10. [Usage](#usage)
11. [Conclusion](#conclusion)
12. [Future Improvements](#future-improvements)
13. [Contact](#contact)

---

## Project Overview
This project predicts **customer churn** for a telecom company using machine learning. The goal is to identify customers who are likely to leave the service based on key features like **Monthly Charges, Total Charges, Contract Type, Internet Service, and Payment Method**.

The project includes:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering and encoding
- Machine learning model building
- Deployment as a **web app/API** using **Flask** and **Railway**

---

## Problem Statement
Customer churn is a major concern for telecom companies as it directly impacts revenue. By predicting which customers are likely to churn, businesses can take proactive steps to **retain high-risk customers**.

---

## Technologies & Libraries
- Python 3.11
- Flask (Web Framework)
- Pandas, NumPy (Data Processing)
- Seaborn, Matplotlib (Data Visualization)
- Scikit-learn (Machine Learning)
- Pickle (Model Serialization)
- Railway (Cloud Deployment)

---

## Dataset
Dataset used: `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
- **Rows:** ~7,000  
- **Columns:** 21 features including customer demographics, subscription details, and usage metrics.  

**Key Features for This Model:**
- `MonthlyCharges` – monthly billing amount
- `TotalCharges` – total charges incurred
- `Contract` – contract type (month-to-month, one year, two year)
- `InternetService` – type of internet service (DSL, Fiber optic, No)
- `PaymentMethod` – method of payment (electronic check, mailed check, etc.)

---

## Exploratory Data Analysis (EDA)
- Checked missing values and cleaned `TotalCharges`
- Converted categorical variables to dummy variables
- Visualized distributions of numerical and categorical features
- Explored relationships between predictors and churn
- Identified key features affecting churn

---

## Key Insights
- **Electronic check** payment users are more likely to churn
- Customers on **month-to-month contracts** churn more frequently
- Customers with **no online security** or **no tech support** are at higher risk of churn
- **Non-senior citizens** tend to churn more than senior citizens
- **Gender** and **number of phone lines** have minimal impact on churn
- High **Monthly Charges** combined with low **Tenure** leads to higher churn
- Low **Total Charges** also correlates with higher churn for new customers

---

## Data Preprocessing & Feature Engineering
- Removed missing values in `TotalCharges`
- Converted categorical features using `pd.get_dummies()`
- Created `tenure_group` bins for customer tenure
- Ensured consistent columns between training and input data for model predictions

---

## Modeling
- **Model Used:** Random Forest Classifier (also tried Decision Tree)
- Trained on preprocessed dataset
- Pickled the trained model for deployment (`model.pkl`)
- Performance: Achieved high accuracy on training/testing split (specific metrics can be added)

---

## Web App Deployment
- Built **Flask Web App** with two routes:
  - `/` – homepage with input form
  - `/predict` – POST request for predictions
- Hosted on **Railway** cloud platform: [Churn Prediction App](https://web-production-e2af2.up.railway.app/predict)
- Input form captures:
  - Monthly Charges
  - Total Charges
  - Contract type
  - Internet service
  - Payment method
- Returns prediction:
  - **Customer is Likely to Churn ❌**
  - **Customer is Not Likely to Churn ✅**

---

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction
