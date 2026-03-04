from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training feature columns
data = pd.read_csv("tel_churn.csv")
data.drop("Unnamed: 0", axis=1, inplace=True, errors='ignore')
feature_columns = data.drop("Churn", axis=1).columns

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        input_data = request.form.to_dict()
        for k, v in input_data.items():
            if v.strip() == "":
                input_data[k] = 0

        input_df = pd.DataFrame([input_data])

        # Convert numeric columns
        for col in ["MonthlyCharges", "TotalCharges"]:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)

        # Dummy encode categorical columns
        input_df = pd.get_dummies(input_df)

        # Reindex to match training columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prediction = int(model.predict(input_df)[0])
        result = "Customer is Likely to Churn ❌" if prediction == 1 else "Customer is Not Likely to Churn ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error in predict route: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)