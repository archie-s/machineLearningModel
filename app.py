from flask import Flask, request, jsonify, render_template
import joblib
from tensorflow import keras
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, scaler, encoders, and feature columns
model = keras.models.load_model('multi_disease_prediction_model.keras')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

def preprocess_input(new_data, encoders, scaler, feature_columns):
    """
    Preprocess user input for prediction.
    - new_data: Dictionary containing feature values from the user.
    - encoders: Dictionary of fitted LabelEncoders for categorical variables.
    - scaler: Fitted MinMaxScaler for numerical variables.
    - feature_columns: List of feature column names in the correct order.
    """
    # Map binary columns ('yes'/'no') to 1/0
    binary_columns = ['temp_gradient', 'pallor', 'indrawing', 'resp_deep',
                      'fever_high', 'jaundice']
    for col in binary_columns:
        if col in new_data:
            new_data[col] = 1 if new_data[col].lower() == 'yes' else 0

    # Encode categorical variables
    for col, encoder in encoders.items():
        if col in new_data:
            try:
                new_data[col] = encoder.transform([new_data[col]])[0]
            except ValueError:
                new_data[col] = 0  # Assign default value for unseen labels

    # Create a DataFrame in the correct feature order
    data_frame = pd.DataFrame([new_data], columns=feature_columns)

    # Scale numerical features
    if hasattr(scaler, 'feature_names_in_'):
        numerical_columns = scaler.feature_names_in_
        data_frame[numerical_columns] = scaler.transform(data_frame[numerical_columns])

    return data_frame

@app.route('/')
def index():
    """Render the HTML form for user input."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    """
    try:
        # Get user input from the form or API
        user_input = request.get_json()
        if not user_input:
            return jsonify({'error': 'No input provided'})

        # Preprocess the input
        preprocessed_data = preprocess_input(user_input, encoders, scaler, feature_columns)

        # Make predictions
        predictions = model.predict(preprocessed_data)

        # Construct response
        result = {
            'died': 'Yes' if predictions[0][0][0] > 0.5 else 'No',
            'hbgrp': int(predictions[1][0].argmax()),
            'oxysat': int(predictions[2][0].argmax()),
            'parasitaemia': 'Yes' if predictions[3][0][0] > 0.5 else 'No'
        }

        return jsonify(result)

    except Exception as e:
        # Handle errors gracefully
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
