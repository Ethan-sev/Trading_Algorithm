from flask import Flask, render_template, request
from pymongo import MongoClient

# Initialize Flask app
app = Flask(__name__)

# Initialize MongoDB client
client = MongoClient("mongodb://localhost:27017")  # Make sure this is correct
db = client['Stock_Data']  # Replace with your actual database name
collection = db['Stock_5year']  # Replace with your actual collection name

@app.route('/')
def home():
    # Get a distinct list of stock tickers from the MongoDB collection
    stocks = collection.distinct("Ticker")
    return render_template('Stock.html', stocks=stocks)

@app.route('/stock', methods=['POST'])
def stock():
    # Get the selected stock ticker from the form
    selected_stock = request.form['stock']
    
    # Fetch stock data from MongoDB for the selected stock
    stock_data = collection.find_one({"Ticker": selected_stock}, sort=[("Date", -1)])
    
    # Return the stock data to display in HTML
    return render_template('Stock.html', stock_data=stock_data, stocks=collection.distinct("Ticker"))

if __name__ == '__main__':
    app.run(debug=True)