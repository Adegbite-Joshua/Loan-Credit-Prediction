from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import sys
import preprocessing



app = Flask(__name__)
CORS(app)





sys.modules['__main__'].FeatureEngineering = preprocessing.FeatureEngineering
sys.modules['__main__'].NumericalTransformer = preprocessing.NumericalTransformer
sys.modules['__main__'].iqr_cap = preprocessing.iqr_cap

# Load the model pipeline
artifact = joblib.load("credit_scoring_model.pkl")
model = artifact["model"]

# Feature columns expected by the model
feature_columns = [
    'Age','Occupation','Annual_Income','Monthly_Inhand_Salary',
    'Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan',
    'Delay_from_due_date','Num_of_Delayed_Payment','Changed_Credit_Limit',
    'Num_Credit_Inquiries','Credit_Mix','Outstanding_Debt',
    'Credit_Utilization_Ratio','Credit_History_Age','Payment_of_Min_Amount',
    'Total_EMI_per_month','Amount_invested_monthly','Payment_Behaviour','Monthly_Balance'
]

# Helper to convert numpy types to native Python types
def to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, pd.Series)):
        return [to_serializable(x) for x in obj]
    else:
        return obj

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Accept both single dict or list of dicts
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)
        df = df.reindex(columns=feature_columns)

        # Ensure object columns are strings (for text processing)
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype(str)

        # Ensure numeric columns are floats
        for col in df.select_dtypes(include=["int64", "float64"]):
            df[col] = df[col].astype(float)

        # Make predictions
        predictions = model.predict(df)
        predictions = to_serializable(predictions)

        # Get probabilities if available
        prob_list = []
        if hasattr(model, "predict_proba"):
            prob_array = model.predict_proba(df)
            for row in prob_array:
                prob_dict = {str(cls): to_serializable(row[i]) for i, cls in enumerate(model.classes_)}
                prob_list.append(prob_dict)

        # Return predictions for all rows
        results = []
        for i in range(len(df)):
            results.append({
                "prediction": predictions[i],
                "probability": prob_list[i] if prob_list else {}
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health')
def health():
    return {"status": "running"}

if __name__ == "__main__":
    print("Model Loaded Successfully")
    app.run(debug=True, port=8080)