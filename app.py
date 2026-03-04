from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# -----------------------------
# Load trained model safely
# -----------------------------
try:
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    model = None
    print(f"Error loading model.pkl: {e}")

# -----------------------------
# Load training data to get feature columns
# -----------------------------
try:
    data = pd.read_csv("tel_churn.csv")
    data.drop("Unnamed: 0", axis=1, inplace=True, errors='ignore')
    feature_columns = data.drop("Churn", axis=1).columns
except Exception as e:
    data = None
    feature_columns = []
    print(f"Error loading tel_churn.csv: {e}")

# -----------------------------
# Home route
# -----------------------------
@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return f"Error in home route: {e}"

# -----------------------------
# Predict route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None or not feature_columns:
            return "Model or feature columns not loaded correctly."

        # Get form data
        input_data = request.form.to_dict()
        input_df = pd.DataFrame([input_data])

        # Convert numeric columns safely
        numeric_cols = ["MonthlyCharges", "TotalCharges"]
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Apply dummy encoding
        input_df = pd.get_dummies(input_df)

        # **Reindex to match training columns safely**
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        # Build result string
        result = "Customer is Likely to Churn ❌" if prediction == 1 else "Customer is Not Likely to Churn ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"Error in predict route: {e}"

# -----------------------------
# Run app (for Railway, use PORT env variable)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)