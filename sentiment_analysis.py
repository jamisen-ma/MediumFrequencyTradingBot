import csv
import praw
import tweepy
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Set up NLTK stopwords
nltk.download('stopwords')

# Reddit setup
reddit = praw.Reddit(client_id='YOUR_REDDIT_CLIENT_ID',
                     client_secret='YOUR_REDDIT_CLIENT_SECRET',
                     user_agent='YOUR_REDDIT_USER_AGENT')

# Twitter setup
auth = tweepy.OAuth1UserHandler('YOUR_TWITTER_CONSUMER_KEY', 'YOUR_TWITTER_CONSUMER_SECRET', 'YOUR_TWITTER_ACCESS_TOKEN', 'YOUR_TWITTER_ACCESS_SECRET')
api = tweepy.API(auth)

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
            reddit_data.append(submission.title + ' ' + submission.selftext)
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                reddit_data.append(comment.body)
    return reddit_data

# Function to get Twitter data
def get_twitter_data(energy_stocks, count=100):
    twitter_data = []
    for stock in energy_stocks:
        query = f"{stock['company']} OR {stock['ticker']}"
        tweets = api.search(q=query, lang='en', count=count)
        twitter_data.extend([tweet.text for tweet in tweets])
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

# Function to analyze sentiment
def analyze_sentiment(energy_stocks):
    reddit_data = get_reddit_data(energy_stocks)
    twitter_data = get_twitter_data(energy_stocks)
    
    combined_data = reddit_data + twitter_data
    preprocessed_data = [preprocess_text(text) for text in combined_data]
    
    vader_sentiments = [get_vader_sentiment(text) for text in preprocessed_data]
    transformer_sentiments = [get_transformer_sentiment(text) for text in preprocessed_data]
    
    avg_vader_sentiment = sum(vader_sentiments) / len(vader_sentiments)
    avg_transformer_sentiment = sum(transformer_sentiments) / len(transformer_sentiments)
    
    # Combine both sentiments (you can adjust weights based on your preference)
    avg_sentiment = (avg_vader_sentiment + avg_transformer_sentiment) / 2
    
    return avg_sentiment

# Load energy stocks
energy_stocks = read_energy_stocks('energy_stocks.csv')

# Analyze sentiment for energy stocks
average_sentiment = analyze_sentiment(energy_stocks)
print(f"Average sentiment for energy stocks: {average_sentiment}")
