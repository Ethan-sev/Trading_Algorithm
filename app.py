from flask import Flask, render_template, request
from pymongo import MongoClient
import joblib
import numpy as np

app = Flask(__name__)

# Connect to MongoDB (ensure your MongoDB is running)
client = MongoClient("mongodb://localhost:27017/")
db = client["options_db"]
collection = db["historical_with_premiums"]

# Load the trained regression model and scaler
model = joblib.load("trained_model.pkl")
scaler = joblib.load("scaler.pkl")

def get_stock_data(ticker):
    """
    Retrieve the latest stock data record from MongoDB for the given ticker.
    Expected fields in the record:
      - "Ticker"
      - "Stock Price"
      - "OTM1", "OTM2", "OTM3", "OTM4", "OTM5"
      - "Pred_Premium_OTM1", "Pred_Premium_OTM2", "Pred_Premium_OTM3", "Pred_Premium_OTM4"
      - "sma_50", "sma_200", "annualized_volatility"
    """
    return collection.find_one({"Ticker": ticker}, sort=[("Date", -1)])

def get_strike_prices(ticker):
    """
    Retrieve a list of available strike prices (OTM1-OTM5) for the given ticker.
    """
    data = get_stock_data(ticker)
    if not data:
        return []
    return [
        data.get("OTM1"),
        data.get("OTM2"),
        data.get("OTM3"),
        data.get("OTM4"),
        data.get("OTM5")
    ]

@app.route("/", methods=["GET", "POST"])
def index():
    # Get all available tickers from MongoDB
    tickers = collection.distinct("Ticker")
    # On GET, display the main form (Stock.html)
    # On POST, if a ticker is submitted, re-render Stock.html with available strike prices.
    if request.method == "POST":
        ticker = request.form.get("ticker")
        if ticker:
            strike_prices = get_strike_prices(ticker)
            return render_template("select_strike.html", ticker=ticker, strike_prices=strike_prices, tickers=tickers)
    return render_template("stock.html", tickers=tickers)

@app.route("/assess", methods=["POST"])
def assess():
    # Retrieve user inputs from the form
    ticker = request.form.get("ticker", "").upper()
    strike_str = request.form.get("strike_price", "")
    
    if not ticker or not strike_str:
        return render_template("select_strike.html", error="Please select both a ticker and a strike price.", ticker=ticker)
    
    try:
        user_strike = float(strike_str)
    except ValueError:
        return render_template("select_strike.html", error="Invalid strike price format.", ticker=ticker)
    
    # Retrieve the latest stock data from MongoDB
    stock_data = get_stock_data(ticker)
    if not stock_data:
        return render_template("select_strike.html", error=f"Ticker '{ticker}' not found in database.", ticker=ticker)
    
    stock_price = stock_data.get("Stock Price")
    sma_50 = stock_data.get("sma_50")
    sma_200 = stock_data.get("sma_200")
    volatility = stock_data.get("annualized_volatility")
    
    # Get the available OTM strike prices and corresponding premiums
    strikes = [
        stock_data.get("OTM1"),
        stock_data.get("OTM2"),
        stock_data.get("OTM3"),
        stock_data.get("OTM4")
    ]
    premiums = [
        stock_data.get("Pred_Premium_OTM1"),
        stock_data.get("Pred_Premium_OTM2"),
        stock_data.get("Pred_Premium_OTM3"),
        stock_data.get("Pred_Premium_OTM4")
    ]
    
    predictions = {}
    best_profit = -np.inf
    best_option = None
    
    # Build the feature vector for each strike option.
    # Our training features are: [Stock Price, sma_50, sma_200, annualized_volatility, Strike Price, Premium]
    for i in range(4):
        strike = strikes[i]
        premium = premiums[i]
        feature_vector = np.array([stock_price, sma_50, sma_200, volatility, strike, premium]).reshape(1, -1)
        features_scaled = scaler.transform(feature_vector)
        predicted_profit = model.predict(features_scaled)[0]  # Regression output (profit in dollars)
        predictions[f"OTM{i+1}"] = {"strike": strike, "predicted_profit": predicted_profit}
        if predicted_profit > best_profit:
            best_profit = predicted_profit
            best_option = f"OTM{i+1}"
    
    # Final recommendation: if the best predicted profit is positive, recommend selling at that strike.
    if best_profit > 0:
        final_recommendation = f"Sell the put at strike {predictions[best_option]['strike']} (Expected profit: ${best_profit:.2f})"
    else:
        final_recommendation = "Do not sell the put (expected profit negative)"
    
    return render_template("result.html",
                           ticker=ticker,
                           stock_price=stock_price,
                           strike_price=user_strike,
                           predictions=predictions,
                           final_recommendation=final_recommendation)

if __name__ == "__main__":
    app.run(debug=True)