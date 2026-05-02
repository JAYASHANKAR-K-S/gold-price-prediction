# 📈 Gold Price Prediction

A Machine Learning web application built with Flask that predicts future gold prices using multiple regression models. It integrates with live APIs to compare model predictions against real-time market values, offering users dynamic accuracy metrics and price conversions.

## 🚀 Features

- **Multi-Model Prediction**: Utilizes three different machine learning models to forecast gold prices:
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- **Live Market Comparison**: Fetches real-time gold prices via [GoldAPI.io](https://www.goldapi.io/) to calculate and display the live percentage accuracy of the predictions.
- **Currency Conversion**: Uses [ExchangeRate-API](https://www.exchangerate-api.com/) to automatically convert predicted and actual USD prices into Indian Rupees (INR).
- **Interactive UI**: Clean and intuitive web interface built with HTML/CSS and Flask templating.

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, NumPy, Joblib
- **Frontend**: HTML, CSS, Jinja2 Templates
- **External APIs**: GoldAPI (Live Gold Prices), ExchangeRate-API (Live Currency Rates)

## 📁 Project Structure

```text
gold-price-prediction/
│
├── app.py                 # Main Flask application and API routes
├── decisiontree.pkl       # Trained Decision Tree model
├── knn.pkl                # Trained K-Nearest Neighbors model
├── randomforest.pkl       # Trained Random Forest model
└── templates/
    └── index.html         # Frontend user interface
```

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/JAYASHANKAR-K-S/gold-price-prediction.git
   cd gold-price-prediction
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Install the required Python packages:
   ```bash
   pip install flask numpy joblib requests scikit-learn
   ```

4. **Set up API Keys**
   - Get a free API key from [GoldAPI.io](https://www.goldapi.io/)
   - Open `app.py` and replace the `api_key` variable with your actual API key.

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **View in Browser**
   Open your web browser and navigate to `http://127.0.0.1:5000/`

## 🧠 How It Works

1. The user inputs a future target date via the web interface.
2. The date is parsed and converted into usable features (`Year`, `Month`, `Day`, `Weekday`).
3. The features are fed into the three pre-trained ML models to predict the price in USD per ounce.
4. The backend converts the predicted USD value to INR and converts ounces to pennyweights.
5. Real-time data is fetched from external APIs to calculate the precise variance and prediction accuracy.
6. Results are returned and displayed on the frontend.
