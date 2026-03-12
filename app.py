from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and preprocessors
model = None
label_encoders = {}
scaler = None
feature_columns = ['Age', 'Income', 'Debt', 'Credit_Score', 'Loan_Amount', 
                   'Loan_Term', 'Num_Credit_Cards', 'Gender', 'Education', 
                   'Payment_History', 'Employment_Status', 'Residence_Type', 
                   'Marital_Status']

def create_sample_model():
    """Create a sample model for demonstration purposes"""
    # Create synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    data = {
        'Age': np.random.randint(18, 70, n_samples),
        'Income': np.random.randint(20000, 200000, n_samples),
        'Debt': np.random.randint(0, 100000, n_samples),
        'Credit_Score': np.random.randint(300, 850, n_samples),
        'Loan_Amount': np.random.randint(5000, 100000, n_samples),
        'Loan_Term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'Num_Credit_Cards': np.random.randint(0, 10, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'Payment_History': np.random.choice(['Good', 'Bad'], n_samples),
        'Employment_Status': np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], n_samples),
        'Residence_Type': np.random.choice(['Rented', 'Owned', 'Mortgaged'], n_samples),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on some rules
    # Higher credit score, good payment history, and higher income increase chances of approval
    df['Creditworthiness'] = (
        (df['Credit_Score'] > 650) & 
        (df['Payment_History'] == 'Good') & 
        (df['Income'] > df['Loan_Amount'] * 0.3) &
        (df['Debt'] < df['Income'] * 0.4)
    ).astype(int)
    
    # Add some randomness
    df['Creditworthiness'] = (df['Creditworthiness'] + np.random.random(n_samples) > 1.2).astype(int)
    
    return df

def train_model():
    """Train a model using synthetic data"""
    global label_encoders, scaler, model
    
    # Create sample data
    df = create_sample_model()
    
    # Encode categorical variables
    categorical_columns = ['Gender', 'Education', 'Payment_History', 
                          'Employment_Status', 'Residence_Type', 'Marital_Status']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['Creditworthiness']
    
    # Scale numerical features
    numerical_columns = ['Age', 'Income', 'Debt', 'Credit_Score', 'Loan_Amount', 
                        'Loan_Term', 'Num_Credit_Cards']
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Model trained successfully!")

def preprocess_input(data):
    """Preprocess input data for prediction"""
    global label_encoders, scaler, feature_columns
    
    # Convert input data to DataFrame
    df = pd.DataFrame([data])
    
    # Rename columns to match training data
    column_mapping = {
        'age': 'Age',
        'gender': 'Gender',
        'education': 'Education',
        'income': 'Income',
        'debt': 'Debt',
        'credit_score': 'Credit_Score',
        'loan_amount': 'Loan_Amount',
        'loan_term': 'Loan_Term',
        'num_credit_cards': 'Num_Credit_Cards',
        'payment_history': 'Payment_History',
        'employment_status': 'Employment_Status',
        'residence_type': 'Residence_Type',
        'marital_status': 'Marital_Status'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Encode categorical variables
    categorical_columns = ['Gender', 'Education', 'Payment_History', 
                          'Employment_Status', 'Residence_Type', 'Marital_Status']
    
    for col in categorical_columns:
        if col in df.columns and col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col])
            except ValueError as e:
                # Handle unknown categories
                print(f"Warning: Unknown category in {col}")
                # Use the most frequent category as fallback
                df[col] = 0
    
    # Scale numerical features
    numerical_columns = ['Age', 'Income', 'Debt', 'Credit_Score', 'Loan_Amount', 
                        'Loan_Term', 'Num_Credit_Cards']
    
    if scaler is not None:
        df[numerical_columns] = scaler.transform(df[numerical_columns])
    
    # Ensure all required columns are present and in correct order
    df = df[feature_columns]
    
    return df

@app.route('/')
def index():
    """Serve the HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input data"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if required fields are present
        required_fields = ['age', 'gender', 'education', 'income', 'debt', 
                          'credit_score', 'loan_amount', 'loan_term', 
                          'num_credit_cards', 'payment_history', 
                          'employment_status', 'residence_type', 'marital_status']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Preprocess input data
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': {
                'approved': float(probability[1]),
                'declined': float(probability[0])
            },
            'message': 'Loan Approved' if prediction == 1 else 'Loan Declined'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

if __name__ == '__main__':
    # Train the model when starting the app
    print("Training model...")
    train_model()
    print("Model ready!")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)