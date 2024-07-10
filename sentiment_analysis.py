import csv
import praw
import tweepy
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import sqlite3
import requests
import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# Set up NLTK stopwords
nltk.download('stopwords')

# Reddit setup
reddit = praw.Reddit(client_id='YOUR_REDDIT_CLIENT_ID',
                     client_secret='YOUR_REDDIT_CLIENT_SECRET',
                     user_agent='YOUR_REDDIT_USER_AGENT')

# Twitter setup
auth = tweepy.OAuth1UserHandler('YOUR_TWITTER_CONSUMER_KEY', 'YOUR_TWITTER_CONSUMER_SECRET', 'YOUR_TWITTER_ACCESS_TOKEN', 'YOUR_TWITTER_ACCESS_SECRET')
api = tweepy.API(auth)

# Yahoo Finance setup
YAHOO_FINANCE_API_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"

# Read energy stocks from CSV
def read_energy_stocks(csv_file):
    energy_stocks = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            energy_stocks.append({'company': row['Company'], 'ticker': row['Ticker']})
    return energy_stocks

# Function to get Reddit data
def get_reddit_data(energy_stocks, limit=100):
    reddit_data = []
    for stock in energy_stocks:
        subreddit = reddit.subreddit('all')
        query = f"{stock['company']} OR {stock['ticker']}"
        for submission in subreddit.search(query, limit=limit):
            reddit_data.append((stock['ticker'], submission.title + ' ' + submission.selftext))
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                reddit_data.append((stock['ticker'], comment.body))
    return reddit_data

# Function to get Twitter data
def get_twitter_data(energy_stocks, count=100):
    twitter_data = []
    for stock in energy_stocks:
        query = f"{stock['company']} OR {stock['ticker']}"
        tweets = api.search(q=query, lang='en', count=count)
        twitter_data.extend([(stock['ticker'], tweet.text) for tweet in tweets])
    return twitter_data

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# VADER Sentiment Analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Transformer-based sentiment analysis
transformer_analyzer = pipeline('sentiment-analysis')

# Function to get VADER sentiment
def get_vader_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

# Function to get Transformer-based sentiment
def get_transformer_sentiment(text):
    analysis = transformer_analyzer(text)
    if analysis[0]['label'] == 'NEGATIVE':
        return -analysis[0]['score']
    else:
        return analysis[0]['score']

# Fetch financial data from Yahoo Finance
def fetch_financial_data(symbol):
    end_date = int(datetime.datetime.now().timestamp())
    start_date = end_date - (30 * 24 * 60 * 60)  # Last 30 days
    url = f"{YAHOO_FINANCE_API_URL}{symbol}?period1={start_date}&period2={end_date}&interval=1d"
    response = requests.get(url)
    data = response.json()['chart']['result'][0]
    return data

# Save sentiment data to the database
def save_sentiment_data(source, symbol, sentiment_score):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO SentimentData (source, symbol, sentiment_score)
        VALUES (?, ?, ?)
    ''', (source, symbol, sentiment_score))
    conn.commit()
    conn.close()

# Save financial data to the database
def save_financial_data(symbol, financial_data):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    for i in range(len(financial_data['timestamp'])):
        cursor.execute('''
            INSERT INTO FinancialData (symbol, date, open, high, low, close, volume, adj_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            datetime.datetime.fromtimestamp(financial_data['timestamp'][i]).strftime('%Y-%m-%d'),
            financial_data['indicators']['quote'][0]['open'][i],
            financial_data['indicators']['quote'][0]['high'][i],
            financial_data['indicators']['quote'][0]['low'][i],
            financial_data['indicators']['quote'][0]['close'][i],
            financial_data['indicators']['quote'][0]['volume'][i],
            financial_data['indicators']['adjclose'][0]['adjclose'][i]
        ))
    conn.commit()
    conn.close()

# Train a model based on historical sentiment and price data
def train_model(symbol):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT sd.sentiment_score, fd.close
        FROM SentimentData sd
        JOIN FinancialData fd ON sd.symbol = fd.symbol AND DATE(sd.timestamp) = DATE(fd.date)
        WHERE sd.symbol = ?
    ''', (symbol,))
    data = cursor.fetchall()

    if len(data) > 1:
        X = np.array([d[0] for d in data[:-1]]).reshape(-1, 1)
        y = np.array([d[1] for d in data[1:]])
        model = LinearRegression()
        model.fit(X, y)
        return model
    return None

# Backtest the model on historical data
def backtest_model(symbol, model):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT sd.sentiment_score, fd.close
        FROM SentimentData sd
        JOIN FinancialData fd ON sd.symbol = fd.symbol AND DATE(sd.timestamp) = DATE(fd.date)
        WHERE sd.symbol = ?
    ''', (symbol,))
    data = cursor.fetchall()

    if len(data) > 1:
        X_test = np.array([d[0] for d in data[:-1]]).reshape(-1, 1)
        y_test = np.array([d[1] for d in data[1:]])
        predictions = model.predict(X_test)
        return predictions, y_test
    return None, None

# Function to analyze sentiment
def analyze_sentiment(energy_stocks):
    reddit_data = get_reddit_data(energy_stocks)
    twitter_data = get_twitter_data(energy_stocks)
    
    combined_data = reddit_data + twitter_data
    preprocessed_data = [(symbol, preprocess_text(text)) for symbol, text in combined_data]
    
    sentiment_scores = {}
    
    for symbol, text in preprocessed_data:
        vader_sentiment = get_vader_sentiment(text)
        transformer_sentiment = get_transformer_sentiment(text)
        
        # Save sentiment data
        save_sentiment_data('Reddit', symbol, vader_sentiment)
        save_sentiment_data('Twitter', symbol, transformer_sentiment)
        
        # Combine both sentiments (you can adjust weights based on your preference)
        avg_sentiment = (vader_sentiment + transformer_sentiment) / 2
        if symbol in sentiment_scores:
            sentiment_scores[symbol] = (sentiment_scores[symbol] + avg_sentiment) / 2
        else:
            sentiment_scores[symbol] = avg_sentiment
    
    # Save sentiment scores to file
    save_sentiment_to_file(sentiment_scores)
    
    return sentiment_scores

# Load energy stocks
energy_stocks = read_energy_stocks('energy_stocks.csv')

# Analyze sentiment for energy stocks
average_sentiment = analyze_sentiment(energy_stocks)
print(f"Average sentiment for energy stocks: {average_sentiment}")

# Train and backtest models for each stock
for stock in energy_stocks:
    symbol = stock['ticker']
    financial_data = fetch_financial_data(symbol)
    save_financial_data(symbol, financial_data)

    model = train_model(symbol)
    if model:
        predictions, actuals = backtest_model(symbol, model)
        if predictions is not None and actuals is not None:
            print(f"Backtest results for {symbol}:")
            for pred, act in zip(predictions, actuals):
                print(f"Predicted: {pred}, Actual: {act}")
