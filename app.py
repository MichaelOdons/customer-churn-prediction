from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load training data to get column structure
data = pd.read_csv("tel_churn.csv")
data.drop("Unnamed: 0", axis=1, inplace=True)

# Save feature column order
feature_columns = data.drop("Churn", axis=1).columns


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get form data
    input_data = request.form.to_dict()

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert numeric columns
    numeric_cols = ["MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        input_df[col] = pd.to_numeric(input_df[col])

    # Apply same dummy encoding
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure same column order
    input_df = input_df[feature_columns]

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        result = "Customer is Likely to Churn ❌"
    else:
        result = "Customer is Not Likely to Churn ✅"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)