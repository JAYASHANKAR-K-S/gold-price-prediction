import os
import joblib
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template
from datetime import datetime
# --- Fix: Set up correct model directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR  # .pkl files are in the same folder as app.py

# --- Load pre-trained models safely ---
try:
    random_forest = joblib.load(os.path.join(MODEL_DIR, 'randomforest.pkl'))
    knn_ = joblib.load(os.path.join(MODEL_DIR, 'knn.pkl'))
    decision_tree = joblib.load(os.path.join(MODEL_DIR, 'decisiontree.pkl'))
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Model file missing: {e.filename}. Please check model paths."
    )


# --- Model accuracies (static display values) ---
model_accuracy1 = 99.79  # Random Forest
model_accuracy2 = 94.92  # KNN
model_accuracy3 = 100.00  # Decision Tree

# --- External APIs ---
api_key = 'goldapi-1b9bbslywo2n1w-io'  # replace with your valid GoldAPI key
api_url = 'https://www.goldapi.io/api/XAU/INR'
usd_to_inr_api_url = 'https://api.exchangerate-api.com/v4/latest/USD'

# --- Flask app setup ---
app = Flask(__name__)

# --- Helper: Convert date to features ---
def generate_features(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    features = np.array([date.year, date.month, date.day, date.weekday()]).reshape(1, -1)
    return features

# --- Helper: Fetch live gold price ---
def get_real_time_gold_price():
    headers = {'x-access-token': api_key, 'Content-Type': 'application/json'}
    try:
        response = requests.get(api_url, headers=headers)
        data = response.json()
        return data.get('price')
    except Exception:
        return None

# --- Helper: USD → INR ---
def convert_usd_to_inr(usd_value):
    try:
        response = requests.get(usd_to_inr_api_url)
        data = response.json()
        return usd_value * data['rates']['INR'] if 'INR' in data['rates'] else None
    except Exception:
        return None

# --- Helper: Troy ounce → pennyweight ---
def convert_ounce_to_pennyweight(ounce_price):
    return ounce_price / 20 if ounce_price else None

# --- Prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form.get('date')
    features = generate_features(date_str)

    # Predictions in USD
    prediction1_usd = random_forest.predict(features)[0]
    prediction2_usd = knn_.predict(features)[0]
    prediction3_usd = decision_tree.predict(features)[0]

    # Convert to INR
    prediction1_inr = convert_usd_to_inr(prediction1_usd)
    prediction2_inr = convert_usd_to_inr(prediction2_usd)
    prediction3_inr = convert_usd_to_inr(prediction3_usd)

    # Get actual gold price (per pennyweight)
    actual_gold_price_per_ounce_inr = get_real_time_gold_price()
    actual_gold_price_per_pennyweight_inr = convert_ounce_to_pennyweight(actual_gold_price_per_ounce_inr)

    if not actual_gold_price_per_pennyweight_inr:
        return jsonify({'error': 'Failed to fetch real-time data'}), 500

    # Compare prediction accuracy
    accuracy1 = 100 - abs((actual_gold_price_per_pennyweight_inr - prediction1_inr) / actual_gold_price_per_pennyweight_inr) * 100
    accuracy2 = 100 - abs((actual_gold_price_per_pennyweight_inr - prediction2_inr) / actual_gold_price_per_pennyweight_inr) * 100
    accuracy3 = 100 - abs((actual_gold_price_per_pennyweight_inr - prediction3_inr) / actual_gold_price_per_pennyweight_inr) * 100

    return jsonify({
        'gold_price_model1_usd': prediction1_usd,
        'gold_price_model1_inr': prediction1_inr,
        'gold_price_model2_usd': prediction2_usd,
        'gold_price_model2_inr': prediction2_inr,
        'gold_price_model3_usd': prediction3_usd,
        'gold_price_model3_inr': prediction3_inr,
        'actual_value_per_pennyweight_inr': actual_gold_price_per_pennyweight_inr,
        'model_accuracy1': model_accuracy1,
        'model_accuracy2': model_accuracy2,
        'model_accuracy3': model_accuracy3,
        'accuracy1': accuracy1,
        'accuracy2': accuracy2,
        'accuracy3': accuracy3
    })

# --- Home route ---
@app.route('/')
def home():
    return render_template('index.html')

# --- Run app ---
if __name__ == '__main__':
    app.run(debug=True)
